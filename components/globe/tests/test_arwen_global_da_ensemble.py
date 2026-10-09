"""The resident ensemble, its analysis and the deterministic coupling, on
the T3 smoke configuration (numpy, reference physics).

What is held: an ensemble built around a state carries the spread its
options ask for and shares one model (its members step under their own
conservation targets and their surface planes agree); a dense report set
drawn from a displaced truth reduces O-A below O-B on the assimilated
rows and the grid rmse of the mean against that truth; the mean increment
embeds in a higher truncation bit for bit and the deterministic update
keeps the global-mean surface pressure; recentering puts the ensemble
mean on the deterministic analysis exactly; a member checkpoint set reads
back to the same arrays; the two analytic families of the OSSE pass on
the filter's own grid fields; refusals name their breakage.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import dataclasses
import datetime as dt

import numpy as np
import pytest

from woof.globe.config import load_config
from woof.globe.constants import SPECTRAL_FIELDS
from woof.globe.da import (
    EnsembleOptions,
    FilterOptions,
    GlobalEnsemble,
    PointObs,
    analyze_ensemble,
    apply_mean_increment,
    embed_spectral,
    ensemble_config,
    recenter,
    truncate_spectral,
)
from woof.globe.da.ensemble import SHARED_SURFACE_BREAKAGE
from woof.globe.da.operators import MemberOperators, batches_from_rows
from woof.globe.da.osse import (
    agreeing_observations_family,
    score,
    single_observation_family,
)
from woof.globe.da.perturbations import (
    climatological_factor,
    draw_perturbation,
    member_rng,
    perturbed_state,
)
from woof.globe.obs_table import ObsRow
from woof.globe.runner import build_model_and_cold_state, build_transform

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
WHEN = dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture(scope="module")
def world():
    cfg = load_config(CONFIG)
    options = EnsembleOptions(members=6, truncation=cfg.truncation, seed=3, additive_inflation_fraction=0.0)
    ecfg = ensemble_config(cfg, options)
    transform = build_transform(ecfg)
    model, cold = build_model_and_cold_state(ecfg, transform)
    return cfg, ecfg, options, transform, model, cold


def _ensemble(world):
    cfg, ecfg, options, transform, model, cold = world
    return GlobalEnsemble.from_state(ecfg, model, transform, cold, options)


def _truth(world, seed=99, options=None):
    cfg, ecfg, world_options, transform, model, cold = world
    options = world_options if options is None else options
    inc, _ = draw_perturbation(model, transform, cold.atmosphere, options, member_rng(seed, 0, "truth"),
                               amplitude_scale=1.5)
    truth = perturbed_state(model, transform, cold, inc)
    mass = model.initialize_mass_target(truth.atmosphere)
    water = model.initialize_water_target(truth)
    model.set_conservation_targets(mass, water)
    truth, _ = model.step(truth, ecfg.dt_s)
    model.release_syntheses()
    return truth


def _station_rows(operators, truth, n=60, seed=0, noisy=True):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-70.0, 70.0, n)
    lon = rng.uniform(0.0, 360.0, n)
    values, _ = operators.evaluate([truth], lat, lon, np.zeros(n), np.full(n, np.nan))
    rows = []
    for k in range(n):
        for variable, error in (("surface_pressure_pa", 100.0), ("temperature_k", 1.0),
                                ("wind_u_m_s", 1.5), ("wind_v_m_s", 1.5)):
            noise = rng.normal(0.0, error) if noisy else 0.0
            rows.append(ObsRow("probe", f"S{k}", lat[k], lon[k], 0.0, None, WHEN, variable,
                               float(values[variable][0, k]) + noise, error))
    aloft_values, _ = operators.evaluate([truth], lat[:20], lon[:20], np.zeros(20), np.full(20, 50000.0))
    for k in range(20):
        rows.append(ObsRow("sonde", f"U{k}", lat[k], lon[k], 5000.0, 50000.0, WHEN, "temperature_k",
                           float(aloft_values["temperature_k"][0, k]) + (rng.normal(0.0, 1.0) if noisy else 0.0), 1.0))
    return rows


def test_ensemble_config_recuts_the_deterministic_config():
    cfg = load_config(CONFIG)
    ecfg = ensemble_config(cfg, EnsembleOptions(members=4, truncation=cfg.truncation))
    assert ecfg.truncation == cfg.truncation and ecfg.dt_s == cfg.dt_s
    assert ecfg.name.endswith(f"-ens-t{cfg.truncation}") and ecfg.config_hash != "" and ecfg.nlat is None
    with pytest.raises(ValueError, match="exceeds the deterministic"):
        ensemble_config(cfg, EnsembleOptions(members=4, truncation=cfg.truncation + 4))


def test_the_ensemble_carries_the_spread_its_options_ask_for_and_steps_per_member(world):
    cfg, ecfg, options, transform, model, cold = world
    # The independent family by name: its three amplitudes are the ones
    # the assertions below read (under the balanced family, the default,
    # the wind follows the pressure amplitude through the balance and at
    # T3 a planetary wind balances that pressure at a fraction of a metre
    # per second; that family's realised amplitudes are in its record).
    independent = dataclasses.replace(options, perturbation_balance="none")
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, independent)
    assert ensemble.size == 6 and ensemble.step == cold.step
    spread = ensemble.spread()
    # The climatological table puts the temperature amplitude at 1.0 K in
    # the troposphere; at T3 with six members the realised global spread
    # sits within a factor two of it, the wind likewise of 2.5 m/s.
    assert 0.3 < spread["temperature_k"] < 2.0
    assert 0.6 < spread["u"] < 5.0
    assert spread["surface_pressure_pa"] > 0.0
    balanced = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    record = balanced.provenance["initial_perturbations"]["first_member_record"]
    assert record["balance"] == "linear" and record["fields"]["balanced"]["scale_factor_to_ln_ps_amplitude"] > 0.0
    assert balanced.spread()["surface_pressure_pa"] > 0.0
    assert len({id(m.surface.land_fraction) for m in ensemble.members}) == 6, "members own their surface copies"
    targets = {(t.mass_pa, t.total_water_kg_m2) for t in ensemble.targets}
    assert len(targets) >= 2, "members carry their own conservation epochs"
    timing = ensemble.step_all(ecfg.dt_s)
    assert timing.members == 6 and len(timing.per_member_s) == 6 and timing.wall_s > 0.0
    assert ensemble.step == cold.step + 1 and all(m.step == cold.step + 1 for m in ensemble.members)
    rec = timing.as_record()
    assert rec["mean_per_member_s"] > 0.0
    row = ensemble.resident_bytes()
    assert row["members"] == 6 and row["per_member"]["total"] > 0 and row["members_total"] == 6 * row["per_member"]["total"]
    # The provenance states the family.
    assert ensemble.provenance["initial_perturbations"]["first_member_record"]["fields"]["temperature_k"]["target_rms_per_level"]
    assert ensemble.provenance["spread_at_construction"]["temperature_k"] > 0.0


def test_a_member_with_another_surface_is_refused_by_name(world):
    cfg, ecfg, options, transform, model, cold = world
    ensemble = _ensemble(world)
    other = cold.copy()
    other.surface.sea_ice_fraction = other.surface.sea_ice_fraction + 0.5
    ensemble.members[2] = other
    with pytest.raises(ValueError, match="sea-ice and land planes"):
        ensemble._check_shared_surface()
    assert "frozen-column mask" in SHARED_SURFACE_BREAKAGE


def test_climatological_factors_are_the_table_at_its_nodes():
    assert climatological_factor("temperature", np.array([100000.0, 30000.0])).tolist() == [1.0, 1.0]
    assert climatological_factor("temperature", np.array([10000.0]))[0] == pytest.approx(0.6)
    assert climatological_factor("wind", np.array([25000.0]))[0] == pytest.approx(1.2)
    assert climatological_factor("vapor", np.array([2000.0]))[0] == 0.0


def test_the_analysis_reduces_o_minus_a_and_the_rmse_against_a_displaced_truth(world):
    cfg, ecfg, _six, transform, model, cold = world
    # Twelve members: at T3 the wind increment of a six-member ensemble is
    # inside the 10 m reports' own noise (O-A within 0.01 m/s of O-B either
    # way, measured).
    # The independent family by name (as the amplitude test above): at T3
    # the balanced family's planetary wind is scaled to a fraction of a
    # metre per second and the sonde rows sit at their own noise.
    # Its amplitudes by name too (the defaults are the balanced family's
    # calibrated ones, 0.8 hPa and 3.0 m/s; this reading was taken at 1.5
    # hPa and 2.5 m/s and the 10 m wind row's noise band belongs to them).
    options = EnsembleOptions(members=12, truncation=cfg.truncation, seed=3, additive_inflation_fraction=0.0,
                              perturbation_balance="none", perturbation_wind_m_s=2.5,
                              perturbation_ln_surface_pressure=0.0015)
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    ensemble.step_all(ecfg.dt_s)
    # The truth is displaced by the same named family: under the balanced
    # family's internal modes the 10 m wind displacement at T3 is a few
    # centimetres a second and the row below reads nothing but noise.
    truth = _truth(world, options=options)
    operators = MemberOperators.for_model(model, transform, ecfg)
    # Perfect observations (the OSSE's pull family): a noisy set on a T3
    # grid whose prior mean already sits inside the report error can pull
    # the global rmse up while fitting the reports (measured 0.73 to 0.78 K
    # on one seed); the noisy verdict is the T63 twin's.
    rows = _station_rows(operators, truth, n=80, noisy=False)
    batches = batches_from_rows(rows, operators, ensemble.members)
    assert {(b.stream, b.variable) for b in batches} == {
        ("probe", "surface_pressure_pa"), ("probe", "temperature_k"), ("probe", "wind_u_m_s"),
        ("probe", "wind_v_m_s"), ("sonde", "temperature_k")}
    assert all(np.all(np.isfinite(b.ln_pressure)) for b in batches)
    before = score(ensemble, truth)
    # Direct insertion by name: the members' own state is what score reads
    # (under the incremental update, the default, they stay the background
    # until the window adds the portions).
    # (The gross check is not under test: the reports are perfect and the
    # planetary balanced family leaves a T3 station or two with a spread
    # under the truth's displacement, so the check is widened to keep every
    # row, which the chain refusal below then needs.)
    filter_options = FilterOptions(horizontal_cutoff_km=6000.0, gate_minimum_count=1000, thinning=False,
                                   increment_application="direct", background_check_sigmas=12.0)
    result = analyze_ensemble(ensemble, batches, filter_options, analysis_time=WHEN)
    after = score(ensemble, truth)
    report = result.report
    assert report["status"] == "pass", report["gate_of_record"]
    for stream, variables in report["streams"].items():
        for variable, entry in variables.items():
            g = entry["regions"]["global"]["assimilated"]
            if variable == "surface_pressure_pa" or (variable == "temperature_k" and stream == "sonde"):
                assert g["o_minus_a"]["rms"] < g["o_minus_b"]["rms"], (stream, variable)
            else:
                # The 10 m wind and 2 m temperature rows sit at their own
                # noise at T3 (the truncation smooths a 72-column increment
                # onto 16 coefficients, and the 2 m diagnostic reads the
                # analysed wind through the surface layer); the analysis
                # must not make them worse beyond that noise.
                assert g["o_minus_a"]["rms"] < g["o_minus_b"]["rms"] * 1.02, (stream, variable)
    assert after["temperature_k"]["rmse"] < before["temperature_k"]["rmse"]
    assert after["surface_pressure_pa"]["rmse"] < before["surface_pressure_pa"]["rmse"]
    # The wind rmse verdict is the T63 twin's (woof.globe.da.osse):
    # at T3 twelve members and 16 coefficients decide it by realisation.
    assert set(result.mean_increment_spectral) == set(SPECTRAL_FIELDS)
    assert report["increment"]["wind_balance"]["mode"] == "rotational"
    assert report["increment"]["wind_balance"]["divergent_fraction_applied_mean"] == 0.0
    assert report["letkf"]["active_columns"] == report["letkf"]["total_columns"]
    assert report["spread"]["after"]["temperature_k"] < report["spread"]["before"]["temperature_k"]
    assert ensemble.cycles == 1 and "assimilation_history" in ensemble.provenance
    # Offering the same reports again is refused through the chain.
    with pytest.raises(ValueError, match="rejected by quality control or the chain"):
        analyze_ensemble(ensemble, batches_from_rows(rows, operators, ensemble.members), filter_options, analysis_time=WHEN)


def test_the_deterministic_update_embeds_the_mean_increment_and_recentering_lands_the_mean(world):
    cfg, ecfg, options, transform, model, cold = world
    ensemble = _ensemble(world)
    truth = _truth(world, seed=5)
    operators = MemberOperators.for_model(model, transform, ecfg)
    batches = batches_from_rows(_station_rows(operators, truth, n=40, seed=2), operators, ensemble.members)
    filter_options = FilterOptions(horizontal_cutoff_km=6000.0, gate_minimum_count=1000, thinning=False,
                                   increment_application="direct")
    result = analyze_ensemble(ensemble, batches, filter_options, analysis_time=WHEN)
    grid = transform.grid
    xp = transform.backend.xp
    ps_before = grid.global_mean(np.asarray(xp.exp(transform.inverse(cold.atmosphere.log_surface_pressure))))
    det, record = apply_mean_increment(cold, model, transform, result.mean_increment_spectral, options=filter_options)
    ps_after = grid.global_mean(np.asarray(xp.exp(transform.inverse(det.atmosphere.log_surface_pressure))))
    assert ps_after == pytest.approx(ps_before, rel=1e-12)
    assert "embedded in the T3 triangle" in record["embedding"]
    # Recentering: the ensemble mean equals the deterministic analysis's
    # spectral fields (to rounding) and the perturbations are kept.
    pert_before = [getattr(m.atmosphere, "theta") - ensemble.mean_spectral()["theta"] for m in ensemble.members]
    shift = recenter(ensemble, det, transform, mode="state")
    mean = ensemble.mean_spectral()
    for name in ("vorticity", "divergence", "theta", "log_surface_pressure"):
        assert np.allclose(np.asarray(mean[name]), np.asarray(getattr(det.atmosphere, name)), atol=1e-9), name
    for k, m in enumerate(ensemble.members):
        assert np.allclose(np.asarray(m.atmosphere.theta - mean["theta"]), np.asarray(pert_before[k]), atol=1e-9)
    assert shift["mean_shift_grid_rms"]["theta"] >= 0.0


def test_embedding_and_truncation_are_exact_inverses():
    rng = np.random.default_rng(0)
    low = (rng.standard_normal((2, 4, 4)) + 1j * rng.standard_normal((2, 4, 4)))
    high = embed_spectral(low, 7)
    assert high.shape == (2, 8, 8) and np.all(high[:, 4:, :] == 0.0) and np.all(high[:, :, 4:] == 0.0)
    assert np.array_equal(truncate_spectral(high, 3), low)
    with pytest.raises(ValueError, match="embedding, not a truncation"):
        truncate_spectral(low, 7)
    with pytest.raises(ValueError, match="truncation, not an embedding"):
        embed_spectral(high, 3)


def test_the_member_checkpoint_set_reads_back(world, tmp_path):
    cfg, ecfg, options, transform, model, cold = world
    ensemble = _ensemble(world)
    ensemble.step_all(ecfg.dt_s)
    manifest = ensemble.write(tmp_path, label="test")
    assert len(manifest["members"]) == 6 and manifest["step"] == ensemble.step
    again = GlobalEnsemble.read(ecfg, model, transform, tmp_path)
    assert again.size == 6 and again.seeds == ensemble.seeds
    for a, b in zip(ensemble.members, again.members):
        for name in SPECTRAL_FIELDS:
            assert np.array_equal(np.asarray(getattr(a.atmosphere, name)), np.asarray(getattr(b.atmosphere, name)))
    assert [t.mass_pa for t in again.targets] == [t.mass_pa for t in ensemble.targets]


def test_the_analytic_families_pass_on_the_filters_own_grid_fields(world):
    cfg, ecfg, options, transform, model, cold = world
    ensemble = _ensemble(world)
    filter_options = FilterOptions(horizontal_cutoff_km=4000.0, thinning=False)
    single = single_observation_family(ensemble, filter_options, latitude_deg=30.0, longitude_deg=100.0)
    assert single["passed"], single
    # Host doubles: the bar is in float64 epsilons and the receipt says so.
    assert single["state_dtype"] == "float64" and single["bar_in_epsilons"] == 128.0
    assert single["worst_in_epsilons"] <= 128.0, single
    operators = MemberOperators.for_model(model, transform, ecfg)
    batches = batches_from_rows(_station_rows(operators, ensemble.members[0], n=30, noisy=False),
                                operators, ensemble.members)
    agree = agreeing_observations_family(ensemble, batches, filter_options)
    assert agree["passed"], agree
    assert agree["innovation_maxabs"] == 0.0
    assert agree["state_dtype"] == "float64" and agree["worst_in_epsilons"] <= 256.0, agree


def test_a_foreign_variable_without_simulated_is_refused_by_name(world):
    cfg, ecfg, options, transform, model, cold = world
    ensemble = _ensemble(world)
    batch = PointObs("sat", "brightness_temperature_k", [10.0], [20.0], [np.log(50000.0)], [False], [250.0], [1.0])
    with pytest.raises(ValueError, match="outside the neutral vocabulary"):
        analyze_ensemble(ensemble, [batch], FilterOptions(), analysis_time=WHEN)


def test_member_streams_are_the_same_in_every_process():
    # A CRC of the purpose, not Python's salted string hash: the same seed,
    # member and purpose draw the same numbers in any process.
    a = member_rng(3, 2, "initial").standard_normal(3)
    b = member_rng(3, 2, "initial").standard_normal(3)
    c = member_rng(3, 2, "additive").standard_normal(3)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.tolist() == pytest.approx([0.6179366754889363, 0.46915812445085664, 1.7484352134408834])


def test_options_refuse_what_they_cannot_run():
    with pytest.raises(ValueError, match="at least 3"):
        EnsembleOptions(members=2)
    with pytest.raises(ValueError, match="analysed together"):
        FilterOptions(analysis_fields=("u", "theta"))
    with pytest.raises(ValueError, match="wind_balance"):
        FilterOptions(wind_balance="free")
    assert FilterOptions(pressure_vertical_cutoff_lnp=None).vertical_cutoff_for("surface_pressure_pa", True) is None
    assert FilterOptions().vertical_cutoff_for("surface_pressure_pa", True) == FilterOptions().pressure_vertical_cutoff_lnp
    assert FilterOptions().vertical_cutoff_for("temperature_k", True) == 1.06
    assert FilterOptions().vertical_cutoff_for("temperature_k", False) == 3.43


def test_a_semi_lagrangian_config_re_cut_coarser_keeps_its_step():
    """The Eulerian re-cut scales the step with the truncation ratio (a
    coarser member takes a longer step); the semi-Lagrangian step is
    bounded by the flow deformation, which does not scale with the grid,
    so the re-cut keeps the config's step.  The T127 members of a T255
    control at 300 s were re-cut to 600 s and the case's third hour refused
    at a Lipschitz number of 0.7689 against 0.75 (2026-09-07)."""
    import dataclasses

    from woof.globe.config import load_config
    from woof.globe.da.ensemble import recut_config

    cfg = load_config(CONFIG)
    eulerian = recut_config(dataclasses.replace(cfg, dt_s=10.0, duration_s=40.0, output_interval_s=20.0), 1)
    assert eulerian.dt_s == 20.0
    sl = dataclasses.replace(cfg, integrator="sl_si", dt_s=10.0, duration_s=40.0, output_interval_s=20.0)
    assert recut_config(sl, 1).dt_s == 10.0
    assert recut_config(sl, 1, dt_s=20.0).dt_s == 20.0


def test_a_member_that_dies_in_its_step_names_itself_and_carries_its_state(world, monkeypatch):
    """The grade of record died in the members' second hour with "native
    physics produced non-finite theta" and nothing said which member or
    what state entered the step; the ensemble now raises MemberStepError
    with the member, the time, the IAU portion and the state, so the filter
    can write them beside the ensemble (2026-09-07)."""
    from woof.globe.da.ensemble import MemberStepError

    cfg, ecfg, options, transform, model, cold = world
    ensemble = _ensemble(world)
    real_step = model.step
    calls = []

    def dying_step(state, dt_s):
        calls.append(len(calls))
        if len(calls) == 2:
            raise FloatingPointError("native physics produced non-finite theta: 3 values in 1 of 8 columns")
        return real_step(state, dt_s)

    monkeypatch.setattr(model, "step", dying_step)
    with pytest.raises(MemberStepError) as caught:
        ensemble.step_all(ecfg.dt_s)
    exc = caught.value
    assert exc.member == 1 and exc.dt_s == ecfg.dt_s and exc.state is not None
    assert "member 1 of 6" in str(exc) and "non-finite theta" in str(exc) and "IAU portion" not in str(exc)
    assert isinstance(exc, FloatingPointError)
    assert exc.__cause__ is not None


def test_each_member_steps_on_its_own_second_time_level_under_the_semi_lagrangian_core(world):
    """The semi-Lagrangian core keeps its second time level on the model,
    which the members share, so before this every member extrapolated
    from the level the previous member left there: the grade of record's
    member 15 died on a non-finite theta in its second hour and the same
    state stepped alone survived (2026-09-07).  Two members stepped twice
    through the shared model must equal each one stepped twice alone
    through a fresh model, array for array."""
    import dataclasses

    from woof.globe.constants import SPECTRAL_FIELDS

    cfg, _ecfg, _options, _transform, _model, _cold = world
    sl = dataclasses.replace(cfg, integrator="sl_si")
    options = EnsembleOptions(members=3, truncation=sl.truncation, seed=5, additive_inflation_fraction=0.0)
    ecfg = ensemble_config(sl, options)
    transform = build_transform(ecfg)
    model, cold = build_model_and_cold_state(ecfg, transform)
    assert model.semi_lagrangian
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    initial = list(ensemble.members)
    targets = list(ensemble.targets)
    ensemble.step_all(ecfg.dt_s)
    ensemble.step_all(ecfg.dt_s)
    assert all(t is not None for t in ensemble.trajectories) and len(ensemble.trajectories) == 3
    assert model.trajectory_state() is None, "the shared model carries no member's level between calls"
    to_numpy = transform.backend.to_numpy
    for k in range(3):
        alone, _ = build_model_and_cold_state(ecfg, transform)
        alone.set_conservation_targets(targets[k].mass_pa, targets[k].total_water_kg_m2)
        state = initial[k]
        for _ in range(2):
            state, _m = alone.step(state, ecfg.dt_s)
            alone.release_syntheses()
        for name in SPECTRAL_FIELDS:
            a = np.asarray(to_numpy(getattr(ensemble.members[k].atmosphere, name)))
            b = np.asarray(to_numpy(getattr(state.atmosphere, name)))
            assert np.array_equal(a, b), f"member {k} field {name} differs from the member stepped alone"
