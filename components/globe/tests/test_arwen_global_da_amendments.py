"""The design amendments of 2026-09-06 on the T3 / T7 smoke configurations
(numpy, reference physics).

What is held: the control gets its own analysis (A): a planted report
produces the analytic localised gain applied to the CONTROL's innovation
at every gridpoint, a control whose innovation equals the ensemble-mean
innovation receives the ensemble-mean increment to rounding, a T7 control
analysed through a T3 ensemble carries no increment power above T3 and
keeps its global-mean surface pressure, and recentering lands the mean
on the control at the stated fraction; the transfer taper (C) is one
below its start, zero at its end and Parseval holds for the degree power;
observations are compared at their own times (B): a bin of the window's
length reproduces the instantaneous operators bitwise and a shorter bin
observes a report at the step nearest its bin's centre; the incremental
analysis update (D) applies the whole increment over the window and
folds a recentering shift into the pending increments; the Desroziers
ratios read one on a sample built from the optimal scalar gain and the
receipt separates the four assessments (G); the options refuse what they
cannot run.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import dataclasses
import datetime as dt
import math

import numpy as np
import pytest

from woof.globe.config import load_config
from woof.globe.constants import SPECTRAL_FIELDS
from woof.globe.da import (
    ControlBackground,
    EnsembleOptions,
    FilterOptions,
    GlobalEnsemble,
    ObservationWindow,
    PointObs,
    analyze_ensemble,
    ensemble_config,
    recenter,
    taper_weights,
    truncate_spectral,
)
from woof.globe.da.analysis import (
    apply_mean_increment,
    band_summary,
    desroziers,
    spectral_power_by_degree,
    wind_power_by_degree,
)
from woof.globe.da.letkf_point import (
    ColumnGeometry,
    PointLetkfConfig,
    PointLetkfDiagnostics,
    analyze_points_with_control,
    flatten_batches,
    single_observation_increment,
)
from woof.globe.da.operators import MemberOperators, batches_from_rows
from woof.globe.da.perturbations import draw_perturbation, member_rng, perturbed_state
from woof.globe.da.window import batches_unevaluated
from woof.globe.obs_table import ObsRow
from woof.globe.runner import build_model_and_cold_state, build_transform
from woof.da.letkf import gaspari_cohn

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
EPOCH = dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture(scope="module")
def dual():
    """A T7 control beside a T3 ensemble on the smoke physics."""
    base = load_config(CONFIG)
    det_cfg = dataclasses.replace(base, truncation=7, nlat=None, nlon=None, name="smoke-t7")
    det_transform = build_transform(det_cfg)
    det_model, det_cold = build_model_and_cold_state(det_cfg, det_transform)
    options = EnsembleOptions(members=8, truncation=3, seed=11)
    ecfg = ensemble_config(det_cfg, options)
    transform = build_transform(ecfg)
    model, cold = build_model_and_cold_state(ecfg, transform)
    return det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options


def _rows(operators, truth, n=40, seed=0, noisy=True, when=EPOCH):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-70.0, 70.0, n)
    lon = rng.uniform(0.0, 360.0, n)
    values, _ = operators.evaluate([truth], lat, lon, np.zeros(n), np.full(n, np.nan))
    rows = []
    for k in range(n):
        for variable, error in (("surface_pressure_pa", 100.0), ("temperature_k", 1.0),
                                ("wind_u_m_s", 1.5), ("wind_v_m_s", 1.5)):
            noise = rng.normal(0.0, error) if noisy else 0.0
            rows.append(ObsRow("probe", f"S{k}", lat[k], lon[k], 0.0, None, when, variable,
                               float(values[variable][0, k]) + noise, error))
    return rows


def _global_ps(transform, state):
    xp = transform.backend.xp
    return transform.grid.global_mean(np.asarray(xp.exp(transform.inverse(state.atmosphere.log_surface_pressure))))


# ---------------------------------------------------------------------------
# A. the control's own analysis
# ---------------------------------------------------------------------------

def test_a_planted_report_gives_the_control_the_analytic_gain_on_its_own_innovation(dual):
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = dual
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    filter_options = FilterOptions(horizontal_cutoff_km=5000.0, thinning=False)
    fields, ln_p_full, ln_ps = ensemble.grid_fields(tuple(filter_options.analysis_fields))
    grid = transform.grid
    geometry = ColumnGeometry(latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg,
                              ln_p_full=ln_p_full, ln_ps=ln_ps, radius_m=float(grid.radius_m))
    operators = MemberOperators.for_model(model, transform, ecfg)
    lat, lon, level = 30.0, 100.0, 50000.0
    values, _ = operators.evaluate(ensemble.members, np.array([lat]), np.array([lon]), np.zeros(1), np.array([level]))
    sim = values["temperature_k"]
    # The control is a third state: the cold start stepped once (its own H).
    control_state = cold.copy()
    mass = model.initialize_mass_target(control_state.atmosphere)
    water = model.initialize_water_target(control_state)
    model.set_conservation_targets(mass, water)
    control_state, _ = model.step(control_state, ecfg.dt_s)
    model.release_syntheses()
    ctl_values, _ = operators.evaluate([control_state], np.array([lat]), np.array([lon]), np.zeros(1), np.array([level]))
    h_control = float(ctl_values["temperature_k"][0, 0])
    y = float(sim.mean()) + 2.0
    batch = PointObs("planted", "temperature_k", [lat], [lon], [math.log(level)], [False], [y], [1.0],
                     simulated=sim, control_simulated=ctl_values["temperature_k"])
    xp = transform.backend.xp
    hcut_m = filter_options.horizontal_cutoff_km * 1000.0
    flat = flatten_batches([batch], xp, horizontal_cutoff_m=hcut_m,
                           vertical_cutoff_for=lambda b, s: np.full(b.count, filter_options.vertical_cutoff_lnp),
                           solve_dtype="float64", control=True)
    assert flat.control_innovation is not None and float(flat.control_innovation[0]) == pytest.approx(y - h_control)
    diagnostics = PointLetkfDiagnostics()
    solved = analyze_points_with_control(fields, flat, geometry, PointLetkfConfig(rtps_alpha=0.0), diagnostics)
    assert solved.control_increment is not None
    from woof.globe.da.letkf_point import _geodesic
    lat_g, lon_g = np.meshgrid(np.deg2rad(grid.latitude_deg), np.deg2rad(grid.longitude_deg), indexing="ij")
    dist = _geodesic(math.radians(lat), math.radians(lon), lat_g, lon_g, float(grid.radius_m), np)
    wh = np.asarray(gaspari_cohn(dist / hcut_m, 1.0))
    vcut = filter_options.vertical_cutoff_lnp
    w3 = wh[None] * np.asarray(gaspari_cohn(np.abs(np.asarray(ln_p_full) - math.log(level)) / vcut, 1.0))
    w2 = wh * np.asarray(gaspari_cohn(np.abs(np.asarray(ln_ps) - math.log(level)) / vcut, 1.0))
    sim_np = np.asarray(sim[:, 0])
    worst = 0.0
    for name, inc in solved.control_increment.items():
        prior_np = np.asarray(fields[name])
        if name in diagnostics.zero_spread_fields:
            # The velocity potential of an unstepped rotational draw: the
            # members agree to rounding, the increment is zero by name.
            assert not np.asarray(inc).any(), name
            continue
        w = w3 if prior_np.ndim == 4 else w2
        # The analytic gain applied to the CONTROL innovation: K (y - H(x_H)).
        analytic_mean = single_observation_increment(prior_np, sim_np, y, 1.0, w)
        d_mean = y - float(sim_np.mean())
        analytic_control = analytic_mean * ((y - h_control) / d_mean)
        got = np.asarray(inc)
        scale = float(np.max(np.abs(got))) or 1.0
        worst = max(worst, float(np.max(np.abs(analytic_control - got)) / scale))
        assert np.all(got[w == 0.0] == 0.0), name
        # And it is NOT the ensemble-mean increment (the innovations differ).
        mean_inc = np.asarray(solved.increments[name]).mean(axis=0)
        assert not np.allclose(mean_inc, got, rtol=1e-6, atol=1e-12) or abs(d_mean - (y - h_control)) < 1e-9
    assert worst <= 1.0e-9, worst


def test_a_control_whose_innovation_is_the_ensemble_means_receives_the_mean_increment(dual):
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = dual
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    operators = MemberOperators.for_model(model, transform, ecfg)
    filter_options = FilterOptions(horizontal_cutoff_km=5000.0, thinning=False, rtps_alpha=0.0)
    rows = _rows(operators, ensemble.members[1], n=30, seed=4)
    batches = batches_from_rows(rows, operators, ensemble.members)
    for b in batches:
        b.control_simulated = b.simulated.mean(axis=0)[None, :]
    fields, ln_p_full, ln_ps = ensemble.grid_fields(tuple(filter_options.analysis_fields))
    grid = transform.grid
    geometry = ColumnGeometry(latitude_deg=grid.latitude_deg, longitude_deg=grid.longitude_deg,
                              ln_p_full=ln_p_full, ln_ps=ln_ps, radius_m=float(grid.radius_m))
    flat = flatten_batches(batches, transform.backend.xp, horizontal_cutoff_m=5.0e6,
                           vertical_cutoff_for=lambda b, s: np.where(s, 0.6, 1.5), solve_dtype="float64", control=True)
    diagnostics = PointLetkfDiagnostics()
    solved = analyze_points_with_control(fields, flat, geometry, PointLetkfConfig(rtps_alpha=0.0), diagnostics)
    for name, inc in solved.increments.items():
        mean_inc = np.asarray(inc).mean(axis=0)
        ctl = np.asarray(solved.control_increment[name])
        scale = float(np.max(np.abs(mean_inc))) or 1.0
        assert np.max(np.abs(mean_inc - ctl)) / scale < 1.0e-9, name


def test_the_control_analysis_carries_no_power_above_the_ensemble_truncation_and_keeps_its_mass(dual):
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = dual
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    operators = MemberOperators.for_model(model, transform, ecfg)
    truth_inc, _ = draw_perturbation(model, transform, cold.atmosphere, options, member_rng(7, 0, "truth"), amplitude_scale=1.5)
    truth = perturbed_state(model, transform, cold, truth_inc)
    rows = _rows(operators, truth, n=40, seed=2, noisy=False)
    batches = batches_from_rows(rows, operators, ensemble.members)
    control = ControlBackground(det_cold, det_model, det_transform, det_cfg)
    filter_options = FilterOptions(horizontal_cutoff_km=6000.0, thinning=False, gate_minimum_count=1000,
                                   transfer_taper_start_degree=2, transfer_taper_end_degree=3,
                                   increment_application="direct")
    ps_before = _global_ps(det_transform, det_cold)
    result = analyze_ensemble(ensemble, batches, filter_options, analysis_time=EPOCH, control=control)
    assert result.status == "pass", result.report["assessments"]["engineering_validity"]
    assert result.control_analysis is not None and result.control_record is not None
    # No increment power above T3 in the T7 control (the embedding is exact).
    for name in SPECTRAL_FIELDS:
        inc = np.asarray(getattr(result.control_analysis.atmosphere, name)) - np.asarray(getattr(det_cold.atmosphere, name))
        power = spectral_power_by_degree(inc)
        assert power.size == 8
        if name == "qv":
            # The model's positivity repair runs on the T7 grid after the
            # embedded increment lands, so the vapor may carry the repair's
            # residual above T3 (measured 1e-39 against a field of 1e-5).
            assert np.all(power[4:] <= 1.0e-30), name
        else:
            assert np.all(power[4:] == 0.0), name
        if name in ("theta", "vorticity"):
            assert power[:3].sum() > 0.0, name
            # The taper: degree 3 is the taper's end, so its weight is zero.
            assert power[3] == 0.0, name
    assert _global_ps(det_transform, result.control_analysis) == pytest.approx(ps_before, rel=1e-12)
    record = result.control_record
    assert record["taper"] == {**record["taper"], "start_degree": 2, "end_degree": 3}
    assert record["taper"]["weights_by_degree"][:3] == [1.0, 1.0, 1.0] and record["taper"]["weights_by_degree"][3] == 0.0
    assert record["spectrum_after_taper"]["temperature_k2"]["bands"]["above_ensemble_T"]["power"] == 0.0
    # The receipt carries the control's own O-B and O-A beside the ensemble's.
    entry = result.report["streams"]["probe"]["temperature_k"]
    ctl = entry["regions"]["global"]["control"]["assimilated"]
    assert ctl["o_minus_b"]["count"] == 40 and ctl["o_minus_a"]["count"] == 40
    assert ctl["o_minus_a"]["rms"] < ctl["o_minus_b"]["rms"]
    assert result.report["control_truncation"] == 7 and result.report["truncation"] == 3
    # Recentering at fraction one lands the ensemble mean on the control truncated to T3.
    # Increment recentring (the default): the mean's increment becomes the
    # control's increment at T3; the members keep their own background.
    mean_before_recentring = ensemble.mean_spectral()
    shift = recenter(ensemble, result.control_analysis, det_transform, fraction=1.0,
                     control_increment=result.control_increment_spectral,
                     mean_increment=result.mean_increment_spectral)
    mean = ensemble.mean_spectral()
    for name in ("vorticity", "divergence", "theta", "log_surface_pressure"):
        expected = (np.asarray(mean_before_recentring[name]) - np.asarray(result.mean_increment_spectral[name])
                    + np.asarray(result.control_increment_spectral[name]))
        assert np.allclose(np.asarray(mean[name]), expected, atol=1e-9), name
    assert shift["fraction"] == 1.0 and shift["mode"] == "increment"
    # State recentring lands the mean on the control truncated to T3.
    ensemble_s = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    recenter(ensemble_s, result.control_analysis, det_transform, fraction=1.0, mode="state")
    mean_s = ensemble_s.mean_spectral()
    for name in ("vorticity", "divergence", "theta", "log_surface_pressure"):
        target = truncate_spectral(np.asarray(getattr(result.control_analysis.atmosphere, name)), 3)
        assert np.allclose(np.asarray(mean_s[name]), target, atol=1e-9), name
    # Partial state recentering moves the mean halfway.
    ensemble2 = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    before = ensemble2.mean_spectral()["theta"]
    recenter(ensemble2, result.control_analysis, det_transform, fraction=0.5, mode="state")
    after = ensemble2.mean_spectral()["theta"]
    target = truncate_spectral(np.asarray(result.control_analysis.atmosphere.theta), 3)
    assert np.allclose(np.asarray(after) - np.asarray(before), 0.5 * (target - np.asarray(before)), atol=1e-9)
    # Increment recentring without the increments is refused by name.
    with pytest.raises(ValueError, match="increment recentring needs"):
        recenter(ensemble2, result.control_analysis, det_transform)
    # The comparison experiment still runs and is labelled as such.
    det2, rec = apply_mean_increment(det_cold, det_model, det_transform, result.mean_increment_spectral, options=filter_options)
    assert "comparison experiment" in rec["embedding"]
    assert _global_ps(det_transform, det2) == pytest.approx(ps_before, rel=1e-12)


def test_a_member_above_the_report_bound_keeps_its_increment_and_one_above_the_radiation_ceiling_is_named(dual):
    """A member's surface pressure at a truncated orography legitimately
    exceeds the observation vocabulary's gross bound for a REPORT (a T63
    member carries 109.5 kPa at the Andes' Pacific foot by construction),
    so refusing it there froze the ensemble's own update on every cycle of
    the reference twin (2026-09-06); the state bound is the radiation
    tables' ceiling, a member above it keeps its increment and is named
    with the breakage, direct and IAU alike, and the reading carries every
    member's maximum, its column and the headroom."""
    from woof.globe.da.analysis import (
        MEMBER_CEILING_BREAKAGE,
        RADIATION_SURFACE_PRESSURE_CEILING_PA,
        _apply_increments,
    )

    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = dual
    # The independent perturbation family by name: at T3 the balanced mass
    # of a 2.5 m/s wind is planetary and reads kilopascals, and the planted
    # rise below is what this test measures, not the draw.
    options = dataclasses.replace(options, perturbation_balance="none")
    for application in ("direct", "iau"):
        ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
        filter_options = FilterOptions(horizontal_cutoff_km=6000.0, thinning=False, increment_application=application,
                                       iau_window_s=4 * ecfg.dt_s)
        fields, _ln_p_full, _ln_ps = ensemble.grid_fields(tuple(filter_options.analysis_fields))
        increments = {name: np.zeros_like(np.asarray(fields[name])) for name in filter_options.analysis_fields}
        increments["theta"][:] = 0.01                      # every member moves a little
        half = increments["lnps"].shape[2] // 2
        # Member 0: ln ps up by 0.3 over half the globe (the mass rule takes
        # the mean back and the raised half sits near 117 kPa, above the
        # ceiling); member 1: up by 0.13 over half the globe (about 108.9 kPa,
        # above the report bound, under the ceiling).  A uniform rise would
        # be removed whole.
        increments["lnps"][0, :, :half] = 0.3
        increments["lnps"][1, :, :half] = 0.13
        backgrounds = [m.copy() for m in ensemble.members]
        _mean_increment, record, analysed = _apply_increments(ensemble, increments, filter_options)
        reading = record["members_surface_pressure_pa"]
        assert reading["max_member"] == 0 and len(reading["per_member_max"]) == ensemble.size
        assert reading["ceiling_pa"] == RADIATION_SURFACE_PRESSURE_CEILING_PA
        assert reading["max"] > RADIATION_SURFACE_PRESSURE_CEILING_PA and reading["headroom_pa"] < 0.0
        assert 108_000.0 < reading["per_member_max"][1] < RADIATION_SURFACE_PRESSURE_CEILING_PA
        named = record["members_above_radiation_ceiling"]
        assert [m["member"] for m in named] == [0]
        assert named[0]["columns_above_ceiling"] > 0 and named[0]["breakage"] == MEMBER_CEILING_BREAKAGE
        assert set(named[0]["column_of_max"]) == {"latitude_deg", "longitude_deg"}
        # Both members received their increments: nobody keeps the background.
        for k in (0, 1):
            assert not np.array_equal(np.asarray(analysed[k].atmosphere.log_surface_pressure),
                                      np.asarray(backgrounds[k].atmosphere.log_surface_pressure)), k
        if application == "iau":
            for k in (0, 1):
                pending = ensemble.pending_increments[k]
                assert float(np.max(np.abs(np.asarray(pending["log_surface_pressure"])))) > 0.0, k
        else:
            assert not np.array_equal(np.asarray(ensemble.members[0].atmosphere.log_surface_pressure),
                                      np.asarray(backgrounds[0].atmosphere.log_surface_pressure))
    # A bounded increment names nobody and reads the headroom (direct
    # insertion by name: the members are read back after the shift).
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    filter_options = FilterOptions(horizontal_cutoff_km=6000.0, thinning=False, increment_application="direct")
    fields, _a, _b = ensemble.grid_fields(tuple(filter_options.analysis_fields))
    increments = {name: np.zeros_like(np.asarray(fields[name])) for name in filter_options.analysis_fields}
    increments["lnps"][:] = 0.001
    _m, record, _a = _apply_increments(ensemble, increments, filter_options)
    assert record["members_above_radiation_ceiling"] == []
    assert record["members_surface_pressure_pa"]["headroom_pa"] > 0.0
    # Direct recentring reads the members again after the shift.
    control = ControlBackground(det_cold, det_model, det_transform, det_cfg)
    operators = MemberOperators.for_model(model, transform, ecfg)
    rows = _rows(operators, cold, n=20, seed=3, noisy=True)
    batches = batches_from_rows(rows, operators, ensemble.members)
    result = analyze_ensemble(ensemble, batches, FilterOptions(horizontal_cutoff_km=6000.0, thinning=False,
                                                                gate_minimum_count=1000,
                                                                increment_application="direct"),
                              analysis_time=EPOCH, control=control)
    rec = recenter(ensemble, result.control_analysis, det_transform, fraction=1.0, mode="increment",
                   control_increment=result.control_increment_spectral,
                   mean_increment=result.mean_increment_spectral)
    assert rec["members_surface_pressure_pa"]["headroom_pa"] > 0.0
    assert rec["members_above_radiation_ceiling"] == []


def test_the_twin_refuses_a_displaced_start_above_the_radiation_ceiling_by_name(tmp_path, monkeypatch):
    """A displaced start whose member already exceeds the radiation ceiling
    dies in the twin's first radiation call with no analysis at all (seed
    4242, scale 1.5 at T63 under T127, 2026-09-06: member 14 at 109,711 Pa,
    48 Pa above the ceiling, at the Andes' Pacific foot); the twin reads the
    headroom at the start and refuses such a start by name.  Here the
    ceiling is lowered below the smoke state's own pressure to plant it."""
    from woof.globe.da import analysis as analysis_module
    from woof.globe.da.osse import GlobalOsseSetup, run_global_osse

    setup = GlobalOsseSetup(config=CONFIG, members=3, cycles=1, truncation=3, control_truncation=7,
                            dt_s=10.0, control_dt_s=10.0, interval_s=40.0, spinup_s=20.0,
                            synthetic_stations=12, synthetic_soundings=3, amv_points=4, seed=5)
    monkeypatch.setattr(analysis_module, "RADIATION_SURFACE_PRESSURE_CEILING_PA", 90_000.0)
    with pytest.raises(ValueError, match="exceeds the radiation tables' ceiling") as caught:
        run_global_osse(setup, tmp_path, progress=lambda *_: None)
    text = str(caught.value)
    assert "member 0 at" in text and "seed 5" in text and "T3 members under a T7 control" in text
    assert "90,000 Pa" in text


def test_increment_recentring_keeps_the_ensembles_global_mean_surface_pressure(dual):
    """The mass rule owns every state's global-mean surface pressure through
    an increment; the recentring shift replaces the members' mean increment
    by the control's, so the control's increment must carry the rule's
    constant too, or the ensemble's mean drifts by the control's raw
    increment mean at every cycle (the first arm's members fell 3.1e-4 of
    ln ps below the control in three cycles before this was folded in)."""
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = dual
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    operators = MemberOperators.for_model(model, transform, ecfg)
    truth_inc, _ = draw_perturbation(model, transform, cold.atmosphere, options, member_rng(7, 0, "truth"), amplitude_scale=1.5)
    truth = perturbed_state(model, transform, cold, truth_inc)
    rows = _rows(operators, truth, n=40, seed=2, noisy=True)
    batches = batches_from_rows(rows, operators, ensemble.members)
    control = ControlBackground(det_cold, det_model, det_transform, det_cfg)
    filter_options = FilterOptions(horizontal_cutoff_km=6000.0, thinning=False, gate_minimum_count=1000)

    def member_mean_ps():
        return float(np.mean([_global_ps(transform, m) for m in ensemble.members]))

    before = member_mean_ps()
    result = analyze_ensemble(ensemble, batches, filter_options, analysis_time=EPOCH, control=control)
    assert result.status == "pass"
    offset = float(result.control_record["mass_preserving_log_offset"])
    assert abs(offset) > 1.0e-6, "the planted network must move the control's mean pressure for this test to bite"
    # The members' own increments keep each member's mean (the rule per state).
    assert member_mean_ps() == pytest.approx(before, abs=1.0e-6)
    # The control's tapered increment carries the rule's constant: its
    # degree-0 ln ps coefficient (the grid mean times sqrt(4 pi)) is the raw
    # increment's mean plus the offset, which cancel to the difference
    # between the grid mean and the pressure-weighted mean of the increment.
    handed_mean = float(np.real(np.asarray(result.control_increment_spectral["log_surface_pressure"])[0, 0])) / math.sqrt(4.0 * math.pi)
    # What is left is the second-order term of the log-mean-exp constant
    # (the difference between the grid mean and the pressure-weighted mean
    # of the increment), a quarter of the offset at most on the balanced
    # family's planetary T3 increments (0.14 measured), never the offset.
    assert abs(handed_mean) < 0.25 * abs(offset), (handed_mean, offset)
    recenter(ensemble, result.control_analysis, det_transform, fraction=1.0, mode="increment",
             control_increment=result.control_increment_spectral, mean_increment=result.mean_increment_spectral)
    after = member_mean_ps()
    # Without the constant the shift moved the ensemble mean by the control's
    # raw increment mean (-7.16 Pa on this twin against the control's own
    # +7.02 Pa offset); with it the residual is the second-order term of the
    # rule applied to another field (-0.14 Pa measured), bounded at 5 percent.
    assert abs(after - before) < 0.05 * abs(offset) * before + 1.0e-3, (after - before, offset * before)


def test_a_foreign_stream_without_control_h_is_refused_by_name_when_a_control_is_given(dual):
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = dual
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    sim = np.full((options.members, 1), 250.0) + np.arange(options.members)[:, None] * 0.1
    batch = PointObs("sat", "brightness_temperature_k", [10.0], [20.0], [np.log(50000.0)], [False], [251.0], [1.0],
                     simulated=sim)
    control = ControlBackground(det_cold, det_model, det_transform, det_cfg)
    with pytest.raises(ValueError, match="no control_simulated"):
        analyze_ensemble(ensemble, [batch], FilterOptions(), analysis_time=EPOCH, control=control)


# ---------------------------------------------------------------------------
# C. the transfer taper and the spectra
# ---------------------------------------------------------------------------

def test_the_taper_is_one_below_its_start_zero_at_its_end_and_smooth_between():
    w = taper_weights(127, 76, 127)
    assert w.shape == (128,)
    assert np.all(w[:77] == 1.0) and w[127] == 0.0
    assert np.all(np.diff(w[76:]) <= 0.0)
    assert w[101] == pytest.approx(math.cos(0.5 * math.pi * 25 / 51) ** 2)
    hard = taper_weights(7, 3, 3)
    assert hard.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert FilterOptions().taper_degrees(127) == (76, 127)
    assert FilterOptions(transfer_taper_start_degree=50, transfer_taper_end_degree=100).taper_degrees(127) == (50, 100)


def test_the_degree_power_obeys_parseval_and_the_bands_sum_to_the_total(dual):
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = dual
    theta = np.asarray(det_cold.atmosphere.theta)
    power = spectral_power_by_degree(theta)
    grid_field = np.asarray(det_transform.inverse(det_cold.atmosphere.theta))
    mean_square = float(np.mean([det_transform.grid.global_mean(level ** 2) for level in grid_field]))
    assert float(power.sum()) == pytest.approx(mean_square, rel=1e-9)
    bands = band_summary(power, 3)
    assert sum(v["power"] for k, v in bands.items() if k != "total") == pytest.approx(bands["total"]["power"], rel=1e-12)
    assert bands["above_ensemble_T"]["degrees"] == [4, 7]
    ke = wind_power_by_degree(det_cold.atmosphere.vorticity, det_cold.atmosphere.divergence, float(det_transform.grid.radius_m))
    u, v = det_model.vector.wind_from_vordiv(det_cold.atmosphere.vorticity, det_cold.atmosphere.divergence)
    ke_grid = 0.5 * float(np.mean([det_transform.grid.global_mean(np.asarray(uu) ** 2 + np.asarray(vv) ** 2)
                                   for uu, vv in zip(np.asarray(u), np.asarray(v))]))
    assert float(ke.sum()) == pytest.approx(ke_grid, rel=1e-6)


# ---------------------------------------------------------------------------
# B. observations at their own times
# ---------------------------------------------------------------------------

def test_a_window_of_one_bin_reproduces_the_instantaneous_operators_bitwise(dual):
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = dual
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    operators = MemberOperators.for_model(model, transform, ecfg)
    end = EPOCH + dt.timedelta(seconds=3 * ecfg.dt_s)
    rows = _rows(operators, ensemble.members[0], n=12, seed=3, when=end - dt.timedelta(seconds=8.0))
    batches = batches_unevaluated(rows, operators)
    assert all(b.simulated is None for b in batches)
    window = ObservationWindow(batches, start_s=0.0, end_s=3 * ecfg.dt_s, epoch=EPOCH, bin_s=3 * ecfg.dt_s, dt_s=ecfg.dt_s)
    assert window.nbins == 1 and window.bin_time_s(0) == 3 * ecfg.dt_s
    ensemble.advance_to(3 * ecfg.dt_s, ecfg.dt_s, observer=lambda ens, t: window.observe(ens.members, t, operators))
    assert window.complete()
    reference = batches_from_rows(rows, operators, ensemble.members)
    by_key = {(b.stream, b.variable): b for b in reference}
    for b in window.batches:
        ref = by_key[(b.stream, b.variable)]
        assert np.array_equal(b.simulated, ref.simulated), b.variable
        assert np.array_equal(b.ln_pressure, ref.ln_pressure)
    record = window.record()
    assert record["rows_observed_at_own_time"]["members"] == sum(b.count for b in batches)
    assert record["rows_observed_at_end"]["members"] == 0


def test_a_short_bin_observes_a_report_at_the_step_nearest_its_bin(dual):
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = dual
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    operators = MemberOperators.for_model(model, transform, ecfg)
    dt_s = ecfg.dt_s
    # Four steps; two bins of two steps each.  A report 3 s after the
    # first step belongs to bin 0, observed at the step its bin closes on
    # (t = 2 dt_s).
    early = _rows(operators, ensemble.members[0], n=6, seed=5, when=EPOCH + dt.timedelta(seconds=dt_s + 3.0))
    late = _rows(operators, ensemble.members[0], n=6, seed=6, when=EPOCH + dt.timedelta(seconds=4 * dt_s))
    late = [dataclasses.replace(r, station_id="L" + r.station_id) for r in late]
    rows = early + late
    batches = batches_unevaluated(rows, operators)
    window = ObservationWindow(batches, start_s=0.0, end_s=4 * dt_s, epoch=EPOCH, bin_s=2 * dt_s, dt_s=dt_s)
    assert window.nbins == 2 and window.bin_time_s(0) == 2 * dt_s and window.bin_time_s(1) == 4 * dt_s
    snapshots = {}

    def observer(ens, t):
        window.observe(ens.members, t, operators)
        snapshots[round(t)] = [m.copy() for m in ens.members]

    ensemble.advance_to(4 * dt_s, dt_s, observer=observer)
    assert window.complete()
    temp = next(b for b in window.batches if b.variable == "temperature_k")
    early_rows = np.array([i for i, r in enumerate(rows) if r.variable == "temperature_k" and not r.station_id.startswith("L")])
    # The early rows equal H(members at t = 2 dt_s), the late ones H(members at t = 4 dt_s).
    at_dt = batches_from_rows([rows[i] for i in early_rows], operators, snapshots[round(2 * dt_s)])[0]
    assert np.array_equal(temp.simulated[:, :6], at_dt.simulated)
    at_3dt = batches_from_rows([r for r in late if r.variable == "temperature_k"], operators, snapshots[round(4 * dt_s)])[0]
    assert np.array_equal(temp.simulated[:, 6:], at_3dt.simulated)
    at_end = batches_from_rows([rows[i] for i in early_rows], operators, ensemble.members)[0]
    assert not np.array_equal(temp.simulated[:, :6], at_end.simulated), "the early rows are not the end state's"
    assert np.array_equal(temp.simulated[:, 6:], batches_from_rows(
        [r for r in late if r.variable == "temperature_k"], operators, ensemble.members)[0].simulated)
    # The batches then analyse as usual (simulated prefilled, control too).
    control = ControlBackground(det_cold, det_model, det_transform, det_cfg)
    window.observe([det_cold], 2 * dt_s, control.operators, control=True)
    window.finish([det_cold], control.operators, control=True)
    assert window.complete(control=True)
    result = analyze_ensemble(ensemble, window.batches, FilterOptions(horizontal_cutoff_km=6000.0, thinning=False),
                              analysis_time=EPOCH + dt.timedelta(seconds=4 * dt_s), control=control)
    assert result.status == "pass"
    assert window.record()["rows_observed_at_end"]["control"] > 0


# ---------------------------------------------------------------------------
# D. incremental analysis update and the assessments
# ---------------------------------------------------------------------------

def test_the_iau_applies_the_whole_increment_over_the_window_and_folds_the_recentering_shift(dual):
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = dual
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    operators = MemberOperators.for_model(model, transform, ecfg)
    rows = _rows(operators, det_cold if False else ensemble.members[2], n=30, seed=9)
    batches = batches_from_rows(rows, operators, ensemble.members)
    filter_options = FilterOptions(horizontal_cutoff_km=6000.0, thinning=False, increment_application="iau",
                                   iau_window_s=4 * ecfg.dt_s)
    background_mean = ensemble.mean_spectral()
    result = analyze_ensemble(ensemble, batches, filter_options, analysis_time=EPOCH)
    assert result.status == "pass"
    assert result.report["increment"]["application"] == "iau"
    # The members are still the background; the increment is pending.
    assert ensemble.pending_increments is not None and len(ensemble.pending_increments) == options.members
    for name in SPECTRAL_FIELDS:
        assert np.array_equal(np.asarray(ensemble.mean_spectral()[name]), np.asarray(background_mean[name]))
        assert np.allclose(np.asarray(ensemble.mean_spectral(include_pending=True)[name]),
                           np.asarray(background_mean[name]) + np.asarray(result.mean_increment_spectral[name]), atol=1e-12)
    # A second schedule while one is pending is refused.
    with pytest.raises(ValueError, match="still pending"):
        ensemble.schedule_increments(ensemble.pending_increments, 4 * ecfg.dt_s)
    # Recentering folds the shift into the pending increments.
    control = ControlBackground(det_cold, det_model, det_transform, det_cfg)
    pending_before = [dict(p) for p in ensemble.pending_increments]
    shift = recenter(ensemble, det_cold, det_transform, fraction=1.0, mode="state")
    assert "pending" in shift["route"]
    target = truncate_spectral(np.asarray(det_cold.atmosphere.theta), 3)
    for k in range(options.members):
        delta = np.asarray(ensemble.pending_increments[k]["theta"]) - np.asarray(pending_before[k]["theta"])
        expected = target - np.asarray(ensemble.mean_spectral(include_pending=True)["theta"]) + delta
        # After the fold the pending-inclusive mean equals the target.
    mean_with_pending = ensemble.mean_spectral(include_pending=True)["theta"]
    assert np.allclose(np.asarray(mean_with_pending), target, atol=1e-9)
    # Four steps apply the four portions; the pending set is cleared and the epochs reopen.
    timings = ensemble.advance_to(4 * ecfg.dt_s, ecfg.dt_s)
    assert len(timings) == 4 and ensemble.pending_increments is None and ensemble.pending_steps_left == 0
    assert ensemble.step == cold.step + 4


def test_desroziers_ratios_read_one_on_a_sample_built_from_the_optimal_scalar_gain():
    rng = np.random.default_rng(0)
    n = 400_000
    sigma_b, sigma_o = 2.0, 1.0
    eps_b = rng.normal(0.0, sigma_b, n)
    eps_o = rng.normal(0.0, sigma_o, n)
    d_ob = eps_o - eps_b                              # y - H(x_b) = (truth + eps_o) - (truth + eps_b)
    gain = sigma_b ** 2 / (sigma_b ** 2 + sigma_o ** 2)
    d_oa = d_ob * (1.0 - gain)
    out = desroziers(d_ob, d_oa, np.full(n, sigma_o), np.full(n, sigma_b))
    assert out["error_variance_ratio"] == pytest.approx(1.0, abs=0.02)
    assert out["background_variance_ratio"] == pytest.approx(1.0, abs=0.02)
    assert out["innovation_ratio"] == pytest.approx(1.0, abs=0.02)
    # A spread twice too large reads half.
    out2 = desroziers(d_ob, d_oa, np.full(n, sigma_o), np.full(n, 2 * sigma_b))
    assert out2["background_variance_ratio"] == pytest.approx(0.25, abs=0.02)
    assert "Desroziers" in out["assumptions"]


def test_the_receipt_separates_the_four_assessments_and_engineering_alone_gates(dual):
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = dual
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    operators = MemberOperators.for_model(model, transform, ecfg)
    # Reports built from a state far from the ensemble with a tiny error:
    # the analysis fits them, the withheld rows need not improve, and the
    # status is still pass because everything ran.
    rows = _rows(operators, ensemble.members[3], n=60, seed=12)
    batches = batches_from_rows(rows, operators, ensemble.members)
    result = analyze_ensemble(ensemble, batches, FilterOptions(horizontal_cutoff_km=6000.0, thinning=False,
                                                                gate_minimum_count=20, desroziers_minimum_count=10),
                              analysis_time=EPOCH)
    assessments = result.report["assessments"]
    assert set(assessments) == {"engineering_validity", "statistical_consistency", "physical_consistency", "predictive_value"}
    assert assessments["engineering_validity"]["verdict"] == "pass" and result.status == "pass"
    assert assessments["statistical_consistency"]["verdict"] in ("consistent", "flagged")
    assert assessments["statistical_consistency"]["judged_streams"] >= 1
    assert assessments["physical_consistency"]["verdict"] in ("within_stated_bounds", "flagged")
    assert assessments["predictive_value"]["verdict"] == "not_assessed_in_receipt"
    entry = result.report["streams"]["probe"]["temperature_k"]
    assert entry["desroziers"]["count"] == entry["count"]
    assert "quantiles" in entry["regions"]["global"]["assimilated"]["o_minus_b"]
    assert entry["gated"] is True and "gate_passed" in entry
    assert result.report["gate_of_record"]["passed"] is True
    assert "engineering validity is the only hard gate" in result.report["gate_of_record"]["rule"]
    assert result.report["increment"]["mean_increment_spectrum"]["temperature_k2"]["bands"]["total"]["power"] > 0.0


def test_the_amended_options_refuse_what_they_cannot_run():
    with pytest.raises(ValueError, match="recentering_fraction"):
        FilterOptions(recentering_fraction=1.5)
    with pytest.raises(ValueError, match="increment_application"):
        FilterOptions(increment_application="gradual")
    with pytest.raises(ValueError, match="at or above"):
        FilterOptions(transfer_taper_start_degree=100, transfer_taper_end_degree=50)
    assert EnsembleOptions().additive_inflation_fraction == 0.0
    ident = FilterOptions().identity()
    assert ident["increment_application"] == "iau" and ident["recentering_fraction"] == 1.0
    # The hybrid knob of amendment A: the shipped beta is the package's
    # graded default, a beta below one needs a static covariance table
    # (tests/test_arwen_global_da_hybrid.py holds the hybrid's own gates).
    from woof.globe.da.options import DEFAULT_HYBRID_BETA

    assert ident["hybrid_beta"] == DEFAULT_HYBRID_BETA and ident["static_covariance"] == "packaged"
    with pytest.raises(ValueError, match="names no table"):
        FilterOptions(hybrid_beta=0.5, static_covariance=None)
    with pytest.raises(ValueError, match="hybrid_beta must lie"):
        FilterOptions(hybrid_beta=0.0)
    with pytest.raises(ValueError, match="hybrid_beta must lie"):
        FilterOptions(hybrid_beta=1.5)


def test_the_spectra_read_device_arrays_through_get_without_a_host_conversion():
    # A cupy array refuses an implicit numpy conversion and offers .get();
    # the degree power and the wind power take that route (the T63 twin on
    # the RTX 5090 died in spectral_power_by_degree before this).
    class DeviceArray:
        def __init__(self, host):
            self._host = host
            self.shape = host.shape

        def get(self):
            return self._host

        def __array__(self, *args, **kwargs):
            raise TypeError("Implicit conversion to a NumPy array is not allowed")

    rng = np.random.default_rng(1)
    host = rng.standard_normal((2, 6, 6)) + 1j * rng.standard_normal((2, 6, 6))
    assert np.array_equal(spectral_power_by_degree(DeviceArray(host)), spectral_power_by_degree(host))
    assert np.array_equal(wind_power_by_degree(DeviceArray(host), DeviceArray(host), 6.4e6),
                          wind_power_by_degree(host, host, 6.4e6))


def test_one_evaluation_over_every_batch_equals_the_per_batch_operators(dual):
    # The aloft wind synthesis is the operators' cost; the analysis pays it
    # once per state set, not once per batch and split, and gets the same
    # values to 1e-9 relative (the T63 twin spent its cycle in fifty
    # per-batch evaluations).  Not bitwise: the point set's size changes
    # the GEMM tiling of the basis contraction, and the dewpoint's log
    # amplifies the last bit of the vapor.
    from woof.globe.da.operators import evaluate_batches
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = dual
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    operators = MemberOperators.for_model(model, transform, ecfg)
    rows = _rows(operators, ensemble.members[0], n=15, seed=21)
    rows += [ObsRow("sonde", f"U{k}", 10.0 * k - 40.0, 30.0 * k, 0.0, 50000.0, EPOCH, v, 250.0 + k, 1.0)
             for k in range(5) for v in ("temperature_k", "wind_u_m_s", "dewpoint_k")]
    reference = batches_from_rows(rows, operators, ensemble.members)
    batches = batches_unevaluated(rows, operators)
    evaluate_batches(operators, ensemble.members, batches, target="simulated")
    by_key = {(b.stream, b.variable): b for b in reference}
    for b in batches:
        ref = by_key[(b.stream, b.variable)]
        assert np.allclose(b.simulated, ref.simulated, rtol=1e-9, atol=0.0), b.variable
        assert np.allclose(b.ln_pressure, ref.ln_pressure, rtol=1e-12, atol=0.0), b.variable
    # A subset of rows per batch, values returned rather than written.
    picks = [np.arange(0, b.count, 2) for b in batches]
    values = evaluate_batches(operators, ensemble.members, batches, rows=picks, target=None)
    for b, idx, got in zip(batches, picks, values):
        assert np.allclose(got, b.simulated[:, idx], rtol=1e-9, atol=0.0), b.variable
    # Only the asked variables are computed; the wind synthesis is skipped
    # when no wind row is present.
    temp_only = [b for b in batches if b.variable == "temperature_k"]
    out, _ = operators.evaluate(ensemble.members[:2], temp_only[0].latitude_deg, temp_only[0].longitude_deg,
                                temp_only[0].elevation_m, temp_only[0].level_pa(), variables=("temperature_k",))
    assert np.all(np.isfinite(out["temperature_k"])) and np.all(np.isnan(out["wind_u_m_s"]))
    assert np.allclose(out["temperature_k"], temp_only[0].simulated[:2], rtol=1e-9, atol=0.0)


# ---------------------------------------------------------------------------
# quality control: the sounding-humidity pressure floor
# ---------------------------------------------------------------------------

def test_sounding_humidity_above_the_pressure_floor_is_rejected_by_name():
    """Dewpoint rows aloft above ``humidity_pressure_floor_pa`` are counted
    as ``humidity_above_floor`` and dropped; rows at or below the floor,
    surface dewpoint rows and every other variable pass; ``None`` keeps
    every level.  On the case's first real window 362 of 417 sounding
    dewpoint rows lay above 100 hPa and read 13.5 K wetter than the
    background at a 2.5 K assigned error (Desroziers 7.1)."""
    from woof.globe.da.analysis import _quality_control

    moment = EPOCH
    aloft = PointObs("igra2", "dewpoint_k", [10.0, 20.0, 30.0], [5.0, 5.0, 5.0],
                     [np.log(5000.0), np.log(30000.0), np.log(70000.0)], [False, False, False],
                     [220.0, 250.0, 280.0], [2.5, 2.5, 2.5], valid_time=[moment] * 3)
    kept, counts = _quality_control(aloft, None, FilterOptions(thinning=False), moment, {})
    assert counts["humidity_above_floor"] == 1 and kept.count == 2
    assert np.allclose(np.exp(kept.ln_pressure), [30000.0, 70000.0])
    surface = PointObs("iem-metar", "dewpoint_k", [10.0], [5.0], [np.nan], [True], [280.0], [1.5],
                       valid_time=[moment])
    kept, counts = _quality_control(surface, None, FilterOptions(thinning=False), moment, {})
    assert counts.get("humidity_above_floor", 0) == 0 and kept.count == 1
    temperature = PointObs("igra2", "temperature_k", [10.0], [5.0], [np.log(5000.0)], [False], [220.0], [1.0],
                           valid_time=[moment])
    kept, counts = _quality_control(temperature, None, FilterOptions(thinning=False), moment, {})
    assert counts.get("humidity_above_floor", 0) == 0 and kept.count == 1
    kept, counts = _quality_control(aloft, None, FilterOptions(thinning=False, humidity_pressure_floor_pa=None), moment, {})
    assert counts.get("humidity_above_floor", 0) == 0 and kept.count == 3
    assert FilterOptions().identity()["humidity_pressure_floor_pa"] == 30000.0


def test_a_wind_report_at_the_pole_is_refused_by_name_and_the_operators_do_not_raise(dual):
    """The Amundsen-Scott sounding sits at exactly 90 S: the wind operator
    hands back NaN there instead of the gradient sampler's refusal, the
    quality control counts the row as ``polar_wind`` (and any other
    non-finite equivalent as ``no_finite_equivalent``), and a scalar row at
    the pole still evaluates."""
    from woof.globe.da.analysis import _quality_control
    from woof.globe.da.operators import MemberOperators, evaluate_batches

    from woof.globe.da.ensemble import GlobalEnsemble

    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = dual
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    ops = MemberOperators.for_model(model, transform, ecfg)
    moment = EPOCH
    wind = PointObs("igra2", "wind_u_m_s", [-90.0, -60.0], [0.0, 10.0], [np.log(50000.0)] * 2, [False, False],
                    [5.0, 5.0], [2.0, 2.0], valid_time=[moment] * 2)
    temperature = PointObs("igra2", "temperature_k", [-90.0], [0.0], [np.log(50000.0)], [False], [240.0], [1.0],
                           valid_time=[moment])
    evaluate_batches(ops, ensemble.members, [wind, temperature], target="simulated")
    assert np.isnan(wind.simulated[:, 0]).all() and np.isfinite(wind.simulated[:, 1]).all()
    assert np.isfinite(temperature.simulated).all()
    kept, counts = _quality_control(wind, ensemble, FilterOptions(thinning=False), moment, {})
    assert counts["polar_wind"] == 1 and kept.count == 1 and kept.latitude_deg[0] == -60.0
    poisoned = PointObs("s", "temperature_k", [10.0, 20.0], [0.0, 0.0], [np.log(50000.0)] * 2, [False, False],
                        [240.0, 240.0], [1.0, 1.0], valid_time=[moment] * 2,
                        simulated=np.array([[np.nan, 239.0], [240.5, 239.5]]))
    kept, counts = _quality_control(poisoned, ensemble, FilterOptions(thinning=False), moment, {})
    assert counts["no_finite_equivalent"] == 1 and kept.count == 1
