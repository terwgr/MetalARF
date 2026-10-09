"""The balance package of 2026-09-06: the linearly balanced perturbation
family, the mass-wind consistency instrument, the observation-error
calibration, the per-class vertical localisation and its derivation, and
the incremental analysis update as the default.

What is held: a balanced draw's geopotential carries a geostrophic wind
that IS the drawn wind in the extratropics (correlation above 0.9 at
T21) where the independent family's reads noise; the instrument that
says so is the one the control record carries; the Desroziers table is
laid over a stream's assigned errors before quality control and both are
recorded; every report class carries its own cutoff and a surface
pressure report no longer reaches the whole column; the cutoff fit
recovers a planted Gaspari-Cohn support; the defaults are what the door
runs.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math

import numpy as np
import pytest

from woof.globe.configs_dir import config_root as _shipped_configs
from woof.globe.config import load_config
from woof.globe.constants import SPECTRAL_FIELDS
from woof.globe.da import (
    DESROZIERS_ERROR_TABLE,
    EnsembleOptions,
    FilterOptions,
    GlobalEnsemble,
    analyze_ensemble,
    ensemble_config,
)
from woof.globe.da.analysis import BALANCE_RULE, ControlBackground, increment_balance_record
from woof.globe.da.ensemble import recut_config
from woof.globe.da.localisation import (
    REPORT_CLASSES,
    derive_cutoffs,
    fit_gaspari_cohn_cutoff,
    vertical_correlation_profiles,
)
from woof.globe.da.observation_errors import calibrated_error
from woof.globe.da.operators import MemberOperators, batches_from_rows
from woof.globe.da.perturbations import (
    balanced_mass_from_vorticity,
    draw_perturbation,
    internal_mode_basis,
    latitude_envelope,
    member_rng,
    perturbed_state,
    vertical_mode_basis,
)
from woof.globe.da_control import ControlOptions
from woof.globe.obs_table import ObsRow
from woof.globe.runner import build_model_and_cold_state, build_transform
from woof.da.letkf import gaspari_cohn
from test_arwen_global_assimilate import spun_up  # noqa: F401 - the door fixture rides the import

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
WHEN = dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture(scope="module")
def world_t21():
    """The smoke configuration recut at T31 on twenty surface-stretched
    levels: columns enough for a correlation in the extratropics and
    layers thin enough for the hydrostatic inversion of the balance."""
    from woof.globe.vertical import HybridCoordinate

    vertical = HybridCoordinate.surface_stretched(20, 100.0)
    cfg = dataclasses.replace(
        recut_config(load_config(CONFIG), 31, name="balance-t31"),
        a_half_pa=tuple(float(v) for v in vertical.a_half_pa), b_half=tuple(float(v) for v in vertical.b_half))
    transform = build_transform(cfg)
    model, cold = build_model_and_cold_state(cfg, transform)
    return cfg, transform, model, cold


@pytest.fixture(scope="module")
def world_t3():
    cfg = load_config(CONFIG)
    options = EnsembleOptions(members=6, truncation=cfg.truncation, seed=3, additive_inflation_fraction=0.0)
    ecfg = ensemble_config(cfg, options)
    transform = build_transform(ecfg)
    model, cold = build_model_and_cold_state(ecfg, transform)
    return cfg, ecfg, options, transform, model, cold


def _balance_at_500(model, transform, base, increments):
    record = increment_balance_record(model, transform, base.atmosphere, increments, levels_hpa=(500.0,),
                                      latitude_bound_deg=25.0)
    return record["levels"]["500"]


def test_a_balanced_draw_carries_the_geostrophic_wind_of_its_own_geopotential(world_t21):
    cfg, transform, model, cold = world_t21
    balanced = EnsembleOptions(members=4, truncation=31, seed=11, perturbation_balance="linear",
                               perturbation_unbalanced_fraction=0.0, perturbation_peak_degree=10.0)
    inc, record = draw_perturbation(model, transform, cold.atmosphere, balanced, member_rng(11, 0, "initial"))
    assert record["balance"] == "linear" and record["unbalanced_fraction"] == 0.0
    assert record["fields"]["balanced"]["ln_surface_pressure_rms"] > 0.0
    assert all(v > 0.0 for v in record["fields"]["balanced"]["temperature_rms_per_level"])
    row = _balance_at_500(model, transform, cold, inc)
    assert row["correlation_wind_vs_geostrophic"] > 0.9, row
    assert row["ageostrophic_share"] < 0.5, row
    # The same seed under the independent family: the mass has nothing to
    # do with the wind, and the instrument says so.
    independent = dataclasses.replace(balanced, perturbation_balance="none")
    inc_none, record_none = draw_perturbation(model, transform, cold.atmosphere, independent, member_rng(11, 0, "initial"))
    assert record_none["balance"] == "none" and "balanced" not in record_none["fields"]
    row_none = _balance_at_500(model, transform, cold, inc_none)
    assert abs(row_none["correlation_wind_vs_geostrophic"]) < 0.5, row_none
    # Both families draw a rotational wind: the divergence stays zero.
    assert not np.asarray(inc["divergence"]).any() and not np.asarray(inc_none["divergence"]).any()


def test_the_balanced_mass_is_hydrostatic_and_zero_mean(world_t21):
    cfg, transform, model, cold = world_t21
    options = EnsembleOptions(members=4, truncation=31, seed=5, perturbation_peak_degree=10.0)
    inc, _ = draw_perturbation(model, transform, cold.atmosphere, options, member_rng(5, 0, "initial"))
    t_bal, lnps_bal, phi_bal = balanced_mass_from_vorticity(model, transform, cold.atmosphere, inc["vorticity"])
    phi = np.asarray(phi_bal)
    weights = np.asarray(transform.grid.quadrature_weights)
    # The balanced geopotential has no global mean (the inverse Laplacian
    # leaves n = 0 at zero) and the surface pressure follows its bottom.
    for k in range(phi.shape[0]):
        assert abs(np.sum(weights[:, None] * phi[k]) / (np.sum(weights) * phi.shape[-1])) < 1e-6 * np.abs(phi[k]).max()
    assert np.corrcoef(np.asarray(lnps_bal).ravel(), phi[-1].ravel())[0, 1] > 0.99
    # In the tropics the balanced mass is small against the extratropics.
    lat = np.asarray(transform.grid.latitude_deg)
    tropics = np.abs(lat) < 10.0
    extra = np.abs(lat) > 40.0
    assert np.sqrt(np.mean(phi[-1][tropics] ** 2)) < 0.5 * np.sqrt(np.mean(phi[-1][extra] ** 2))
    basis = vertical_mode_basis(np.log([100.0, 300.0, 500.0, 850.0, 1000.0]), 4)
    assert basis.shape == (5, 4) and np.allclose(np.sum(basis ** 2, axis=1), 1.0)
    assert basis[0, 1] > 0.0 and basis[-1, 1] < 0.0 and np.all(basis[:, 0] > 0.0)
    # The internal basis: zero above its top and at the surface, a zero
    # gradient at the surface (the level above it a few thousandths of the
    # peak), the first mode peaking near the jet, unit mean square over
    # the column, and nothing at all when no modes are asked for.
    lnp = np.linspace(math.log(100.0), math.log(100000.0), 60)
    internal = internal_mode_basis(lnp, 4)
    p_hpa = np.exp(lnp) / 100.0
    assert internal.shape == (60, 5), "four jet-level modes and the low-level mode"
    assert np.all(internal[p_hpa < 70.0] == 0.0) and np.allclose(internal[-1], 0.0, atol=1e-12)
    # The zero gradient at the surface, read on a column fine enough to
    # resolve it (the 60-level one samples a tenth of a scale height up).
    fine = internal_mode_basis(np.linspace(math.log(100.0), math.log(100000.0), 3000), 4)
    assert np.all(np.abs(fine[-2]) < 1.0e-3 * np.abs(fine).max(axis=0))
    assert 200.0 < p_hpa[np.argmax(internal[:, 0])] < 330.0
    assert 620.0 < p_hpa[np.argmax(internal[:, 4])] < 780.0 and np.all(internal[p_hpa < 500.0, 4] == 0.0)
    assert np.mean(np.sum(internal ** 2, axis=1)) == pytest.approx(1.0)
    assert internal_mode_basis(lnp, 0).shape == (60, 0)
    with pytest.raises(ValueError, match="perturbation_internal_modes"):
        EnsembleOptions(perturbation_internal_modes=-1)
    with pytest.raises(ValueError, match="perturbation_tropical_wind_fraction"):
        EnsembleOptions(perturbation_tropical_wind_fraction=0.0)
    envelope = latitude_envelope(np.sin(np.deg2rad([0.0, 30.0, 45.0, 60.0, 90.0])), 0.5)
    assert envelope == pytest.approx([0.5, 0.661, 0.791, 0.901, 1.0], abs=0.002)
    with pytest.raises(ValueError, match="perturbation_vertical_modes"):
        EnsembleOptions(perturbation_vertical_modes=0)
    with pytest.raises(ValueError, match="perturbation_balance"):
        EnsembleOptions(perturbation_balance="geostrophic")
    with pytest.raises(ValueError, match="perturbation_unbalanced_fraction"):
        EnsembleOptions(perturbation_unbalanced_fraction=2.0)


def test_the_potentials_arm_is_selectable_and_the_components_are_the_default():
    assert FilterOptions().analysis_fields == ("u", "v", "theta", "qv", "lnps")
    arm = FilterOptions(analysis_fields=("psi", "chi", "theta", "qv", "lnps"))
    assert arm.analysis_fields[:2] == ("psi", "chi")
    with pytest.raises(ValueError, match="psi and chi"):
        FilterOptions(analysis_fields=("psi", "theta", "qv", "lnps"))
    with pytest.raises(ValueError, match="analysed once"):
        FilterOptions(analysis_fields=("psi", "chi", "u", "v", "theta", "qv", "lnps"))
    assert FilterOptions().max_local_obs == 3000 and FilterOptions().memory_budget_mib == 2048.0


def test_the_defaults_are_the_balanced_family_and_the_incremental_update():
    ensemble = EnsembleOptions()
    assert ensemble.perturbation_balance == "linear"
    assert ensemble.identity()["perturbation_balance"] == "linear"
    filter_options = FilterOptions()
    assert filter_options.increment_application == "iau"
    assert ControlOptions().increment_application == "iau"
    assert filter_options.observation_error_calibration == "desroziers-2026-09-06"
    assert filter_options.pressure_vertical_cutoff_lnp is not None
    ident = filter_options.identity()
    for key in ("aloft_wind_vertical_cutoff_lnp", "aloft_humidity_vertical_cutoff_lnp",
                "surface_wind_vertical_cutoff_lnp", "pressure_vertical_cutoff_lnp",
                "observation_error_calibration", "spread_ratio_band"):
        assert key in ident
    with pytest.raises(ValueError, match="observation_error_calibration"):
        FilterOptions(observation_error_calibration="hollingsworth")
    with pytest.raises(ValueError, match="spread_ratio_band"):
        FilterOptions(spread_ratio_band=(2.0, 1.0))


def test_every_report_class_carries_its_own_vertical_cutoff():
    options = FilterOptions(vertical_cutoff_lnp=1.1, aloft_wind_vertical_cutoff_lnp=0.9,
                            aloft_humidity_vertical_cutoff_lnp=0.7, surface_vertical_cutoff_lnp=0.5,
                            surface_wind_vertical_cutoff_lnp=0.4, pressure_vertical_cutoff_lnp=0.8)
    assert options.vertical_cutoff_for("surface_pressure_pa", True) == 0.8
    assert options.vertical_cutoff_for("temperature_k", True) == 0.5
    assert options.vertical_cutoff_for("dewpoint_k", True) == 0.5
    assert options.vertical_cutoff_for("wind_u_m_s", True) == 0.4
    assert options.vertical_cutoff_for("wind_v_m_s", False) == 0.9
    assert options.vertical_cutoff_for("temperature_k", False) == 1.1
    assert options.vertical_cutoff_for("dewpoint_k", False) == 0.7
    assert FilterOptions(pressure_vertical_cutoff_lnp=None).vertical_cutoff_for("surface_pressure_pa", True) is None


def test_the_calibration_is_laid_over_the_assigned_errors_and_recorded(world_t3):
    cfg, ecfg, options, transform, model, cold = world_t3
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    operators = MemberOperators.for_model(model, transform, ecfg)
    rng = np.random.default_rng(0)
    n = 40
    lat = rng.uniform(-60.0, 60.0, n)
    lon = rng.uniform(0.0, 360.0, n)
    values, _ = operators.evaluate(ensemble.members[:1], lat, lon, np.zeros(n), np.full(n, np.nan))
    rows = [
        ObsRow("iem-metar", f"S{k}", lat[k], lon[k], 0.0, None, WHEN, "temperature_k",
               float(values["temperature_k"][0, k]) + rng.normal(0.0, 1.0), 1.5)
        for k in range(n)
    ] + [
        ObsRow("probe", f"P{k}", lat[k], lon[k], 0.0, None, WHEN, "surface_pressure_pa",
               float(values["surface_pressure_pa"][0, k]) + rng.normal(0.0, 50.0), 100.0)
        for k in range(n)
    ]
    batches = batches_from_rows(rows, operators, ensemble.members)
    filter_options = FilterOptions(horizontal_cutoff_km=6000.0, thinning=False, gate_minimum_count=1000)
    result = analyze_ensemble(ensemble, batches, filter_options, analysis_time=WHEN, additive_inflation=False)
    record = result.report["observation_error_calibration"]
    assert record["name"] == "desroziers-2026-09-06"
    metar = record["streams"]["iem-metar"]["temperature_k"]
    assert metar["assigned_error_rms"] == pytest.approx(1.5)
    assert metar["calibrated_error"] == DESROZIERS_ERROR_TABLE[("iem-metar", "temperature_k")] == 1.75
    assert metar["applied"] is True
    probe = record["streams"]["probe"]["surface_pressure_pa"]
    assert probe["calibrated_error"] is None and probe["applied"] is False
    des = result.report["streams"]["iem-metar"]["temperature_k"]["desroziers"]
    assert des["assigned_error_variance"] == pytest.approx(1.75 ** 2)
    assert "spread_ratio" in des and "background_error_variance_from_innovations" in des
    stat = result.report["assessments"]["statistical_consistency"]["spread_calibration"]
    assert stat["band"] == [0.5, 2.0] and "iem-metar/temperature_k" in stat["cells"]
    # Off by name: the rows keep their own errors.
    errors, value = calibrated_error(None, "iem-metar", "temperature_k", np.full(3, 1.5))
    assert value is None and errors.tolist() == [1.5] * 3
    with pytest.raises(ValueError, match="unknown observation-error calibration"):
        calibrated_error("nmc-1992", "iem-metar", "temperature_k", np.full(3, 1.5))


def test_the_control_record_carries_the_balance_reading(world_t3):
    cfg, ecfg, options, transform, model, cold = world_t3
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    operators = MemberOperators.for_model(model, transform, ecfg)
    rng = np.random.default_rng(4)
    n = 30
    lat = rng.uniform(-60.0, 60.0, n)
    lon = rng.uniform(0.0, 360.0, n)
    values, _ = operators.evaluate(ensemble.members[:1], lat, lon, np.zeros(n), np.full(n, np.nan))
    rows = [ObsRow("probe", f"S{k}", lat[k], lon[k], 0.0, None, WHEN, "surface_pressure_pa",
                   float(values["surface_pressure_pa"][0, k]) + rng.normal(0.0, 50.0), 100.0) for k in range(n)]
    batches = batches_from_rows(rows, operators, ensemble.members)
    control = ControlBackground(cold, model, transform, ecfg, operators=operators)
    result = analyze_ensemble(ensemble, batches, FilterOptions(horizontal_cutoff_km=6000.0, thinning=False,
                                                                gate_minimum_count=1000),
                              analysis_time=WHEN, additive_inflation=False, control=control)
    balance = result.control_record["balance"]
    assert balance["rule"] == BALANCE_RULE
    assert set(balance["levels"]) == {"850", "500", "250"}
    for row in balance["levels"].values():
        assert row["wind_increment_rms_m_s"] is not None and row["geopotential_increment_rms_m"] >= 0.0
    assert balance["surface_pressure_increment_rms_pa"] > 0.0
    assert "divergent_share_of_wind_increment_ke" in balance
    assert result.report["increment"]["control"]["balance"]["latitude_bound_deg"] == 25.0


def test_the_cutoff_fit_recovers_a_planted_support_and_the_profiles_decay():
    separation = np.linspace(0.0, 3.0, 40)
    for planted in (0.35, 0.8, 1.6):
        signal = 0.6 * np.asarray(gaspari_cohn(separation / planted, 1.0))
        fit = fit_gaspari_cohn_cutoff(separation, signal)
        assert fit["cutoff_lnp"] == pytest.approx(planted, rel=0.03), fit
        assert fit["signal_at_source"] == pytest.approx(0.6)
    none = fit_gaspari_cohn_cutoff(separation, np.zeros_like(separation))
    assert none["no_signal_above_floor"] is True and none["cutoff_lnp"] == 0.25
    # A synthetic member set whose level fields decorrelate in ln p over
    # 0.5 scale heights: every class's profile falls with separation and
    # the derived cutoffs sit inside the bounds.
    rng = np.random.default_rng(1)
    members, nlev, nlat, nlon = 40, 12, 8, 16
    lnp = np.linspace(math.log(10000.0), math.log(100000.0), nlev)
    rho = np.exp(-np.abs(np.diff(lnp)) / 0.5)

    def draw():
        out = np.empty((members, nlev, nlat, nlon))
        out[:, 0] = rng.standard_normal((members, nlat, nlon))
        for k in range(1, nlev):
            out[:, k] = rho[k - 1] * out[:, k - 1] + math.sqrt(1.0 - rho[k - 1] ** 2) * rng.standard_normal((members, nlat, nlon))
        return out.astype(np.float32)

    t = draw()
    u = draw()
    # The streamfunction shares the wind's vertical structure and, as a
    # balanced perturbation would, carries the mass field's too.
    psi = 0.6 * u + 0.5 * t + 0.6 * draw()
    fields = {"t": t, "u": u, "v": 0.7 * u + 0.7 * draw(), "psi": psi, "q": 0.5 * t + 0.8 * draw(),
              "lnps": (lnp[-1] + 0.01 * (0.8 * t[:, -1] + 0.6 * rng.standard_normal((members, nlat, nlon)))).astype(np.float32),
              "lnpf": np.broadcast_to(lnp[None, :, None, None], (members, nlev, nlat, nlon)).astype(np.float32)}
    lat = np.linspace(-80.0, 80.0, nlat)
    weights = np.cos(np.deg2rad(lat))
    profiles = vertical_correlation_profiles(fields, lat, weights)
    assert set(profiles["classes"]) == set(REPORT_CLASSES)
    for name, per_region in profiles["classes"].items():
        row = per_region["global"]
        assert row["pairs"] == len(REPORT_CLASSES[name]["pairs"]) * (1 if REPORT_CLASSES[name]["level"] == "surface" else 3)
        assert all(("u" not in pair and "v" not in pair) or pair[0] == pair[1] for pair in row["pair_fields"]), (
            "a wind component pairs only with itself; the cross pairs go through the streamfunction")
        signal = np.asarray(row["signal"])
        separation = np.asarray(row["separation"])
        assert signal[np.argmin(separation)] > 0.5 * signal.max(), name
        assert signal[np.argmax(separation)] < signal[np.argmin(separation)], name
    cutoffs = derive_cutoffs(profiles, region="global")
    assert set(cutoffs) == {spec["option"] for spec in REPORT_CLASSES.values()}
    for option, row in cutoffs.items():
        assert 0.05 <= row["cutoff_lnp"] <= 6.0 and row["fit_rms_residual"] < 0.3, (option, row)


# ---------------------------------------------------------------------------
# The refutation of 2026-09-07: the card judges the error the filter used,
# and a cutoff that is the scan's bound says so
# ---------------------------------------------------------------------------

def _departure_rows(source: str, variable: str, error: float, n: int = 60):
    rng = np.random.default_rng(7)
    lat = rng.uniform(-60.0, 60.0, n)
    lon = rng.uniform(0.0, 360.0, n)
    truth = rng.normal(0.0, 1.0, n)
    background = truth + rng.normal(0.0, 1.0, n)
    analysis = truth + 0.5 * (background - truth)
    reports = truth + rng.normal(0.0, error, n)
    rows = [ObsRow(source, f"S{k}", lat[k], lon[k], 0.0, None, WHEN, variable, float(reports[k]), error)
            for k in range(n)]
    return rows, background, analysis


def test_the_card_reads_desroziers_against_the_error_the_filter_weighted_by():
    """The 2026-09-06 record's card read the motion vectors' Desroziers
    ratio at 0.66 of an assigned 4.0 m/s while the solve had weighted them
    at the calibrated 2.8: the reading judged an error the analysis never
    used.  The card now takes the error the filter weighted by and records
    the door's figure beside it."""
    from woof.globe.da_scorecard import Departures, scorecard

    rows, hb, ha = _departure_rows("iem-metar", "temperature_k", 1.5)
    door = np.array([row.error for row in rows])
    weighted, table_value = calibrated_error("desroziers-2026-09-06", "iem-metar", "temperature_k", door)
    assert table_value == 1.75
    dep = Departures.from_rows(rows, hb, ha, withheld=False, error=weighted)
    assert dep.door_error.tolist() == [1.5] * len(rows) and dep.error.tolist() == [1.75] * len(rows)
    cell = scorecard(dep)["streams"]["iem-metar"]["variables"]["temperature_k"]["regions"]["global"]["consistency"]
    assert cell["assigned_sigma_o"] == pytest.approx(1.75)
    assert cell["door_sigma_o"] == pytest.approx(1.5)
    assert cell["calibrated"] is True
    assert cell["desroziers_ratio"] == pytest.approx(cell["desroziers_sigma_o"] / 1.75)
    assert "calibrated error 1.75" in cell["reading"] and "doors assigned 1.5" in cell["reading"]
    # Without a calibration the two figures are one and the reading says "assigned".
    plain = Departures.from_rows(rows, hb, ha, withheld=False)
    cell = scorecard(plain)["streams"]["iem-metar"]["variables"]["temperature_k"]["regions"]["global"]["consistency"]
    assert cell["assigned_sigma_o"] == pytest.approx(1.5) and cell["door_sigma_o"] == pytest.approx(1.5)
    assert cell["calibrated"] is False and "against assigned 1.5" in cell["reading"]
    # Concatenation carries the door's column with the rest.
    both = Departures.concatenate([dep, plain])
    assert both.count == 2 * len(rows) and np.isfinite(both.door_error).all()
    with pytest.raises(ValueError, match="one weighted error per row"):
        Departures.from_rows(rows, hb, ha, withheld=False, error=weighted[:-1])


def test_the_control_card_of_the_door_lays_the_calibration_over_the_rows(spun_up, tmp_path, monkeypatch):
    """Through the door: a cell the table names is judged at the table's
    error with the door's figure beside it, a cell it does not name at the
    door's, and both say which."""
    from woof.globe import da_door
    from woof.globe.da import observation_errors
    from woof.globe.da_filter import ENSEMBLE_MANIFEST_NAME
    from test_arwen_global_assimilate import _synthetic_obs_files
    from test_arwen_global_cycle import OPTIONS, START_TEXT

    monkeypatch.setitem(observation_errors.DESROZIERS_ERROR_TABLE, ("awc-metar-cache", "temperature_k"), 1.75)
    cfg, checkpoint = spun_up
    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    out = tmp_path / "card"
    da_door.init(cfg, out, filter_name="letkf", members=3, ensemble_truncation=3,
                 analysis_time_utc=START_TEXT, options=OPTIONS)
    receipt = da_door.cycle(
        cfg, out, stream_specs=["local-tables:paths=" + ",".join(map(str, obs))], cycles=1,
        start_utc=START_TEXT, interval_s=20.0, ensemble=out / ENSEMBLE_MANIFEST_NAME, options=OPTIONS,
    )
    assert receipt["status"] == "pass"
    import json as _json
    report = _json.loads((out / "assimilation-report-step00000002.json").read_text())
    calibration = report["ensemble_report"]["observation_error_calibration"]
    assert calibration["name"] == "desroziers-2026-09-06"
    assert calibration["streams"]["awc-metar-cache"]["temperature_k"]["applied"] is True
    named = report["scorecard"]["streams"]["awc-metar-cache"]["variables"]["temperature_k"]["regions"]["global"]["consistency"]
    assert named["calibrated"] is True and named["assigned_sigma_o"] == pytest.approx(1.75)
    assert named["door_sigma_o"] == pytest.approx(calibration["streams"]["awc-metar-cache"]["temperature_k"]["assigned_error_rms"])
    unnamed = report["scorecard"]["streams"]["awc-metar-cache"]["variables"]["surface_pressure_pa"]["regions"]["global"]["consistency"]
    assert unnamed["calibrated"] is False and unnamed["door_sigma_o"] == pytest.approx(unnamed["assigned_sigma_o"])


def test_the_cutoff_fit_names_a_cutoff_that_is_the_scan_bound():
    """The 2026-09-06 derivation's surface-pressure class returned 6.0,
    which is the scan's upper bound and not a value the profile chose; the
    fit now says so."""
    separation = np.linspace(0.0, 3.0, 40)
    deep = 0.5 * np.asarray(gaspari_cohn(separation / 12.0, 1.0))
    fit = fit_gaspari_cohn_cutoff(separation, deep)
    assert fit["cutoff_lnp"] == pytest.approx(6.0) and fit["cutoff_at_bound"] == "upper"
    inside = fit_gaspari_cohn_cutoff(separation, 0.5 * np.asarray(gaspari_cohn(separation / 0.8, 1.0)))
    assert inside["cutoff_at_bound"] is None and inside["cutoff_lnp"] == pytest.approx(0.8, rel=0.03)
    shallow = fit_gaspari_cohn_cutoff(separation, 0.5 * np.asarray(gaspari_cohn(separation / 0.01, 1.0)))
    assert shallow["cutoff_at_bound"] == "lower" and shallow["cutoff_lnp"] == pytest.approx(0.05)
    assert fit_gaspari_cohn_cutoff(separation, np.zeros_like(separation))["cutoff_at_bound"] is None
