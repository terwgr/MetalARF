from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from pathlib import Path

from dataclasses import replace

import numpy as np
import pytest

from woof.globe.config import load_config
from woof.globe.initial_conditions import analytic_initial_state
from woof.globe.runner import build_model_and_cold_state, build_transform
from woof.globe.vertical import HybridCoordinate
from woof.globe.spectral.backend import get_backend


CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def test_config_is_strict_and_identity_is_stable(tmp_path):
    cfg = load_config(CONFIG)
    assert cfg.physics_mode == "reference"
    assert cfg.vertical.nlev == 4
    assert len(cfg.config_hash) == 64
    assert cfg.config_hash == load_config(CONFIG).config_hash

    bad = tmp_path / "bad.toml"
    text = Path(CONFIG).read_text(encoding="utf-8").replace(
        'acknowledgement = "research-only-arwen-global-v1"',
        'acknowledgement = "trust-me"',
    )
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="acknowledgement"):
        load_config(bad)


def test_hybrid_pressure_and_continuity_close():
    coordinate = HybridCoordinate.pressure_blend(6, 100.0)
    backend = get_backend("numpy", "float64")
    ps = np.full((5, 8), 100_000.0)
    pressure = coordinate.pressure(ps, backend)
    assert pressure["p_half"].shape == (7, 5, 8)
    assert np.all(np.diff(pressure["p_half"], axis=0) > 0.0)
    assert np.all(pressure["dp"] > 0.0)

    rng = np.random.default_rng(4)
    divm = rng.normal(0.0, 0.01, (6, 5, 8))
    ps_t = -np.sum(divm, axis=0)
    omega, residual = coordinate.continuity(divm, ps_t, backend)
    assert np.max(np.abs(residual)) < 1.0e-14
    assert np.array_equal(omega[0], np.zeros_like(ps))
    assert np.array_equal(omega[-1], np.zeros_like(ps))


def test_hydrostatic_geopotential_is_monotone_upward():
    coordinate = HybridCoordinate.pressure_blend(5, 100.0)
    backend = get_backend("numpy", "float64")
    ps = np.full((3, 4), 100_000.0)
    p = coordinate.pressure(ps, backend)
    tv = np.full((5, 3, 4), 270.0)
    phi = coordinate.hydrostatic_geopotential(tv, np.zeros((3, 4)), p["p_half"])
    # Arrays are top-to-bottom, so geopotential decreases toward the surface.
    assert np.all(np.diff(phi, axis=0) < 0.0)
    assert np.all(phi > 0.0)


def test_cold_state_and_one_step_are_finite_and_conservative():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    before = model.diagnostics(state)
    state, metrics = model.step(state, cfg.dt_s)
    after = model.diagnostics(state)
    assert state.step == 1
    assert state.time_s == cfg.dt_s
    assert metrics["spectral_cfl"] < cfg.maximum_cfl
    assert abs(after["global_mean_surface_pressure_pa"] - before["global_mean_surface_pressure_pa"]) < 1.0e-8
    assert abs(after["global_mean_total_water_kg_m2"] - before["global_mean_total_water_kg_m2"]) < 1.0e-10
    # The two means above are reset to their targets by the fixers inside
    # step(), so alone they cannot see a leak. Bound what the fixers ABSORBED
    # this step: healthy values measured 1.3e-13 (mass log offset) and
    # 4.7e-9 kg/m2 (water) on this config; the leaks these bounds exist to
    # catch measure 5e-3 and 1.2e-1.
    assert abs(metrics["mass_fixer_log_offset"]) < 1.0e-9
    assert abs(metrics["global_water_fixer_kg_m2"]) < 1.0e-6
    model.enforce(state)


def test_enforce_tolerates_kinked_ringing_and_refuses_negative_mass():
    # Projecting the kinked sparse morphology saturation produces rings up
    # to 0.28 negative/positive mass and 18% of the field maximum (measured
    # T31..T85); enforce must accept that and still refuse a field whose
    # negative mass rivals its positive mass, which no projection makes.
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    transform = model.transform
    rng = np.random.default_rng(7)
    coeff = transform.zeros()
    t = transform.truncation
    for n in range(1, t + 1):
        for m in range(0, n + 1):
            coeff[n, m] = (n + 1.0) ** -0.5 * (
                rng.normal() + (0 if m == 0 else 1j * rng.normal())
            )
    smooth = transform.backend.to_numpy(transform.inverse(coeff))
    smooth = (smooth - smooth.mean()) / max(smooth.std(), 1.0e-12)
    kinked = np.maximum(smooth - 0.8, 0.0)
    kinked *= 2.0e-4 / max(kinked.max(), 1.0e-12)
    nlev = cfg.vertical.nlev
    stack = np.broadcast_to(kinked, (nlev, *kinked.shape)).copy()
    # Vapor is the one water field in the spectral basis: its kinked
    # ringing is tolerated by the negative-mass fraction and a net
    # negative field is refused.
    state.atmosphere.qv = transform.project(transform.forward(stack))
    model.enforce(state)

    state.atmosphere.qv = transform.project(
        transform.forward(stack - 0.9 * float(stack.max()))
    )
    with pytest.raises(FloatingPointError, match="negative mass"):
        model.enforce(state)

    # A grid tracer has no ringing to tolerate: one negative cell refuses.
    state.atmosphere.qv = transform.project(transform.forward(stack))
    model.enforce(state)
    state.atmosphere.qc = np.array(state.atmosphere.qc, copy=True)
    state.atmosphere.qc[0, 0, 0] = -1.0e-12
    with pytest.raises(FloatingPointError, match="grid tracer qc is negative"):
        model.enforce(state)


def test_water_target_anchors_at_the_pre_step_state_like_mass():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    model._target_mass_pa = None
    model._target_total_water_kg_m2 = None
    before = model.diagnostics(state)
    model.step(state, cfg.dt_s)
    # Both targets are computed by the same diagnostics arithmetic as
    # `before`, so a pre-step anchor is bitwise equal; a post-step anchor
    # differs by the first step's drift (measured 1.2e-9 kg/m2 here).
    assert model._target_mass_pa == before["global_mean_surface_pressure_pa"]
    assert (
        model._target_total_water_kg_m2
        == before["global_mean_total_water_kg_m2"]
    )


def test_moment_positivity_repair_leaves_the_grid_moments_untouched():
    cfg = load_config(CONFIG)
    transform = build_transform(cfg)
    model, _ = build_model_and_cold_state(cfg, transform)
    raw, _geo, _provenance = analytic_initial_state(cfg, transform)
    host = transform.backend.to_numpy
    nc_raw = np.array(host(raw.atmosphere.nc), copy=True)
    # The seeded step-function droplet number is a grid tracer: exact on
    # the grid, no ringing (under the spectral representation it rang to
    # -1e6 per kg at this truncation and needed a mean-preserving repair).
    assert nc_raw.min() >= 0.0
    assert nc_raw.max() == 1.0e8

    repaired, negative_water, negative_moment, _paid = model._repair_positivity(raw)

    assert negative_water < 1.0
    assert negative_moment == 0.0
    np.testing.assert_array_equal(host(repaired.atmosphere.nc), nc_raw)
    model.enforce(repaired)


def test_scalar_tendency_divides_by_true_thickness_in_sub_pascal_layers():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    g = model.grid_state(state.atmosphere)
    nlev = model.nlev
    horizontal = g["ps"].shape
    thicknesses = (0.5, 50.0, 500.0, 5000.0)
    assert nlev == len(thicknesses)

    dp = np.empty((nlev, *horizontal))
    for k, thickness in enumerate(thicknesses):
        dp[k] = thickness
    p_half = np.empty((nlev + 1, *horizontal))
    p_half[0] = 100.0
    p_half[1:] = 100.0 + np.cumsum(dp, axis=0)
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    dp_t = 0.02 * g["v"]
    omega_half = np.zeros((nlev + 1, *horizontal))
    omega_half[1:-1] = 0.01 * g["u"][: nlev - 1]

    # The flux-form tendency of a mixing ratio is invariant under scaling the
    # mass field: dp, dp_t, omega and the pressures all scale together and
    # divide back out (the limited vertical reconstruction is a gradient per
    # Pa times a distance in Pa).  A thickness floor breaks exactly this
    # invariance in layers thinner than the floor (here the 0.5 Pa top layer).
    scale = 1000.0
    thin = model._scalar_tendency(
        g["qv"], dp, dp_t, g["u"], g["v"], omega_half, p_full, p_half
    )
    scaled = model._scalar_tendency(
        g["qv"], dp * scale, dp_t * scale, g["u"], g["v"], omega_half * scale,
        p_full * scale, p_half * scale,
    )
    np.testing.assert_allclose(
        thin, scaled, rtol=1.0e-12, atol=1.0e-12 * float(np.max(np.abs(scaled)))
    )

    p_full = np.empty((nlev, *horizontal))
    for k, pressure in enumerate((100.0, 100.5, 600.0, 5600.0)):
        p_full[k] = pressure
    thin_w = model._vertical_momentum_advection(g["u"], omega_half, p_full)
    scaled_w = model._vertical_momentum_advection(
        g["u"], omega_half * scale, p_full * scale
    )
    np.testing.assert_allclose(
        thin_w,
        scaled_w,
        rtol=1.0e-12,
        atol=1.0e-12 * float(np.max(np.abs(scaled_w))),
    )


def test_free_dynamics_drift_without_fixers_is_bounded():
    cfg = replace(load_config(CONFIG), mass_fixer=False, water_fixer=False)
    model, state = build_model_and_cold_state(cfg)
    before = model.diagnostics(state)
    state, metrics = model.step(state, cfg.dt_s)
    after = model.diagnostics(state)
    assert metrics["mass_fixer_log_offset"] == 0.0
    assert metrics["global_water_fixer_kg_m2"] == 0.0
    # With the fixers off these means measure the model itself. Free one-step
    # drift measured 2.8e-14 relative (mass) and 1.3e-12 relative (water) on
    # this config; the bounds sit 2-3 orders above that and far below any
    # dycore or physics actually destroying mass or water.
    ps0 = before["global_mean_surface_pressure_pa"]
    w0 = before["global_mean_total_water_kg_m2"]
    assert abs(after["global_mean_surface_pressure_pa"] - ps0) / ps0 < 1.0e-11
    assert abs(after["global_mean_total_water_kg_m2"] - w0) / w0 < 1.0e-9


def test_free_drift_arm_registers_water_destruction():
    cfg = replace(load_config(CONFIG), mass_fixer=False, water_fixer=False)
    model, state = build_model_and_cold_state(cfg)
    inner = model.physics

    class DeletesWater:
        def __init__(self, physics):
            self._physics = physics

        @property
        def identity(self):
            return self._physics.identity

        def step(self, exchange):
            result = self._physics.step(exchange)
            result.qv = result.qv * 0.99
            return result

    model.physics = DeletesWater(inner)
    before = model.diagnostics(state)
    state, _metrics = model.step(state, cfg.dt_s)
    after = model.diagnostics(state)
    w0 = before["global_mean_total_water_kg_m2"]
    # Instrument check for the arm above: destroying 1% of vapour per physics
    # call moves the free-drift measurement 3+ orders past its bound.
    assert abs(after["global_mean_total_water_kg_m2"] - w0) / w0 > 1.0e-6


def test_cfl_refusal_is_pre_update():
    cfg = replace(load_config(CONFIG), maximum_cfl=1.0e-12)
    model, state = build_model_and_cold_state(cfg)
    original = [np.array(field, copy=True) for field in state.atmosphere.fields()]
    with pytest.raises(ValueError, match="spectral CFL"):
        model.step(state, cfg.dt_s)
    for expected, actual in zip(original, state.atmosphere.fields()):
        assert np.array_equal(expected, actual)
    assert state.step == 0


# ---------------------------------------------------------------------------
# The dycore top-of-model absorber ([sponge]).  Ported from the suite-v7
# reference-physics tests when the absorber moved to the dycore: it
# compensates the dycore's rigid p_top lid, so it acts in every physics
# mode (its reference-suite home left the arwen-native T255 run
# unprotected; that run died at hour 6.55 on the 140 K research bound).
# ---------------------------------------------------------------------------


import math

from woof.globe.state import ArwenGlobalState


def _sponge_seam(dt_s=600.0):
    """Model, pressures, and crafted winds for direct absorber-seam tests.

    The smoke grid's ring-mean p_full is ~978 Pa at the top level and
    ~17,200 Pa at the next, so the 5000 Pa default base puts exactly the
    top level in the absorber on this coarse 4-level grid.
    """
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    g = model.grid_state(state.atmosphere, only=("p_half", "p_full"))
    return cfg, model, state, g


def _predicted_ring_decay(model, g, dt_s):
    """The graded decay factor per (level, latitude) ring."""
    ring_p = np.mean(np.asarray(g["p_full"]), axis=-1, keepdims=True)
    ring_lid = np.mean(np.asarray(g["p_half"][0]), axis=-1, keepdims=True)
    ramp = np.clip(
        (ring_p - ring_lid) / (model.sponge_base_pa - ring_lid), 0.0, 1.0
    )
    rate = (
        np.cos(0.5 * math.pi * ramp) ** 2 / model.sponge_lid_relaxation_time_s
    )
    in_sponge = ring_p < model.sponge_base_pa
    return in_sponge, np.exp(-dt_s * rate)


def _lat_lon(model, g):
    lat = np.deg2rad(model.transform.grid.latitude_deg)[:, None]
    lon = np.deg2rad(model.transform.grid.longitude_deg)[None, :]
    shape = np.asarray(g["p_full"]).shape
    return lat, lon, shape


def test_dycore_sponge_leaves_purely_zonal_flow_untouched():
    _, model, _, g = _sponge_seam()
    lat, _, shape = _lat_lon(model, g)
    level = np.arange(shape[0], dtype=np.float64)[:, None, None]
    # Lon-constant jet, strongest at the top level (inside the absorber):
    # a purely zonal flow IS its own zonal mean, so the anomaly damping
    # has nothing to act on and the jet must come through to roundoff.
    u = np.broadcast_to((50.0 - 10.0 * level) * np.cos(lat)[None], shape).copy()
    v = np.broadcast_to(3.0 * np.cos(lat)[None], shape).copy()
    u_out, v_out = model._top_sponge(u, v, g["p_full"], g["p_half"], 600.0)
    assert float(np.max(np.abs(u_out - u))) < 1.0e-12
    assert float(np.max(np.abs(v_out - v))) < 1.0e-12


def test_dycore_sponge_damps_wave_anomaly_by_the_graded_rayleigh_law():
    _, model, _, g = _sponge_seam()
    _, lon, shape = _lat_lon(model, g)
    u = np.full(shape, 12.0)
    v = np.zeros(shape)
    u[0] += 5.0 * np.sin(2.0 * lon)
    v[0] += 3.0 * np.cos(3.0 * lon)
    u_out, v_out = model._top_sponge(u, v, g["p_full"], g["p_half"], 600.0)
    # One application multiplies each ring's anomaly by exp(-dt *
    # rate(ring)), the exact integral of the graded Rayleigh rate
    # cos^2((pi/2)(p - lid)/(base - lid)) / tau_lid: at the smoke grid's
    # ~978 Pa top ring that is ~46% removed per 600 s step (rate
    # ~1.03e-3 1/s, surviving factor 0.54), not a uniform fraction.
    in_sponge, decay = _predicted_ring_decay(model, g, 600.0)
    assert bool(in_sponge[0].all()) and not bool(in_sponge[1:].any())
    assert float(decay[0].max()) < 0.60  # near full rate at ~978 Pa
    for before, after in ((u[0], u_out[0]), (v[0], v_out[0])):
        mean_before = np.mean(before, axis=-1, keepdims=True)
        anomaly_before = before - mean_before
        anomaly_after = after - np.mean(after, axis=-1, keepdims=True)
        assert float(np.max(np.abs(anomaly_before))) > 1.0
        np.testing.assert_allclose(
            anomaly_after, anomaly_before * decay[0], rtol=0.0, atol=1.0e-12
        )
        # The zonal-mean wind of each latitude ring - the jet - is
        # conserved by construction; only the anomalies are damped.
        np.testing.assert_allclose(
            np.mean(after, axis=-1), mean_before[..., 0], rtol=0.0,
            atol=1.0e-12,
        )


def test_dycore_sponge_ramp_damps_deeper_levels_less():
    _, model, _, g = _sponge_seam()
    _, lon, shape = _lat_lon(model, g)
    u = np.broadcast_to(12.0 + 5.0 * np.sin(2.0 * lon)[None], shape).copy()
    v = np.zeros(shape)
    ring_p = np.mean(np.asarray(g["p_full"]), axis=-1)
    # A base between the second and third levels' ring-mean pressures
    # puts exactly two levels inside the absorber, exercising the ramp.
    model.sponge_base_pa = 0.5 * (
        float(ring_p[1].max()) + float(ring_p[2].min())
    )
    u_out, _ = model._top_sponge(u, v, g["p_full"], g["p_half"], 600.0)
    removed = []
    for k in (0, 1):
        anomaly_before = u[k] - np.mean(u[k], axis=-1, keepdims=True)
        anomaly_after = u_out[k] - np.mean(u_out[k], axis=-1, keepdims=True)
        ratio = np.sqrt(
            np.sum(anomaly_after**2, axis=-1)
            / np.sum(anomaly_before**2, axis=-1)
        )
        removed.append(ratio)
        # Both sponged levels damp: the surviving fraction is below one.
        assert float(ratio.max()) < 1.0 - 1.0e-6
    # The graded ramp: the deeper (higher-pressure) level keeps strictly
    # more of its anomaly on every ring than the level above it.
    assert bool(np.all(removed[1] > removed[0]))
    # Levels below the base pass through bit-exactly.
    np.testing.assert_array_equal(u_out[2:], u[2:])


def test_dycore_sponge_leaves_levels_below_the_base_untouched():
    _, model, _, g = _sponge_seam()
    _, lon, shape = _lat_lon(model, g)
    # The same wave anomaly at every level: only the top level's ring-mean
    # p_full (~978 Pa) sits below the 5000 Pa default base, so the others
    # must pass through bit-exactly.
    u = np.broadcast_to(12.0 + 5.0 * np.sin(2.0 * lon)[None], shape).copy()
    v = np.broadcast_to(3.0 * np.cos(3.0 * lon)[None], shape).copy()
    u_out, v_out = model._top_sponge(u, v, g["p_full"], g["p_half"], 600.0)
    assert float(np.max(np.abs(u_out[0] - u[0]))) > 1.0e-3
    np.testing.assert_array_equal(u_out[1:], u[1:])
    np.testing.assert_array_equal(v_out[1:], v[1:])


def test_dycore_sponge_zero_base_disables_and_refusals_name_their_breakage():
    _, model, _, g = _sponge_seam()
    _, lon, shape = _lat_lon(model, g)
    u = np.broadcast_to(12.0 + 5.0 * np.sin(2.0 * lon)[None], shape).copy()
    v = np.broadcast_to(3.0 * np.cos(3.0 * lon)[None], shape).copy()
    model.sponge_base_pa = 0.0
    u_out, v_out = model._top_sponge(u, v, g["p_full"], g["p_half"], 600.0)
    assert u_out is u and v_out is v
    # Zero is the off switch; a negative base selects no ring while
    # reading as a configured absorber, so construction refuses it.
    with pytest.raises(ValueError, match="selects no ring"):
        replace(model, sponge_base_pa=-1.0)
    # A nonpositive lid time would turn exp(-dt/tau) into amplification
    # of the wave anomalies the absorber exists to remove.
    with pytest.raises(ValueError, match="amplify"):
        replace(model, sponge_lid_relaxation_time_s=0.0)
    with pytest.raises(ValueError, match="amplify"):
        replace(model, sponge_lid_relaxation_time_s=-900.0)


def _anomaly_bundle(model, state):
    """The cold state with a top-level wave anomaly analyzed into it."""
    g = model.grid_state(state.atmosphere, only=("u", "v"))
    lon = np.deg2rad(model.transform.grid.longitude_deg)[None, :]
    u = np.array(g["u"], copy=True)
    v = np.array(g["v"], copy=True)
    u[0] += 5.0 * np.sin(2.0 * lon)
    v[0] += 3.0 * np.cos(3.0 * lon)
    zeta, div = model.vector.vordiv_from_wind(u, v)
    fields = list(state.atmosphere.fields())
    fields[0] = zeta
    fields[1] = div
    return ArwenGlobalState(
        state.atmosphere.with_fields(fields), state.surface,
        state.physics_state,
    )


def test_absorber_acts_with_no_physics_suite_loaded():
    # physics.mode='none' has no exchange and no wind round trip of its
    # own, which is exactly the shape of the gap that killed the native
    # arm: the absorber must not depend on a suite being loaded.
    _, model, state, g = _sponge_seam()
    model.physics = None
    bundle = _anomaly_bundle(model, state)
    out, info = model.apply_physics(bundle, 600.0)
    assert info == {"physics_mode": "none"}
    assert out is not bundle
    # The pass is exactly: synthesize winds, absorber, analyze back.
    u_syn, v_syn = model.vector.wind_from_vordiv(
        bundle.atmosphere.vorticity, bundle.atmosphere.divergence
    )
    u_exp, v_exp = model._top_sponge(
        u_syn, v_syn, g["p_full"], g["p_half"], 600.0
    )
    zeta_exp, div_exp = model.vector.vordiv_from_wind(u_exp, v_exp)
    np.testing.assert_array_equal(
        np.asarray(out.atmosphere.vorticity), np.asarray(zeta_exp)
    )
    np.testing.assert_array_equal(
        np.asarray(out.atmosphere.divergence), np.asarray(div_exp)
    )
    # And the damping is real: the top-level anomaly shrank by the ring
    # decay law (the ring-uniform factor keeps the anomaly band-limited,
    # so the analysis/synthesis round trip adds only roundoff).
    in_sponge, decay = _predicted_ring_decay(model, g, 600.0)
    assert bool(in_sponge[0].all())
    u_after, _ = model.vector.wind_from_vordiv(
        out.atmosphere.vorticity, out.atmosphere.divergence
    )
    anomaly_before = np.asarray(u_syn[0]) - np.mean(
        np.asarray(u_syn[0]), axis=-1, keepdims=True
    )
    anomaly_after = np.asarray(u_after[0]) - np.mean(
        np.asarray(u_after[0]), axis=-1, keepdims=True
    )
    assert float(np.max(np.abs(anomaly_before))) > 1.0
    np.testing.assert_allclose(
        anomaly_after, anomaly_before * decay[0], rtol=0.0, atol=1.0e-9
    )
    # Every other spectral field passes through untouched.
    for index in range(2, len(bundle.atmosphere.fields())):
        np.testing.assert_array_equal(
            np.asarray(out.atmosphere.fields()[index]),
            np.asarray(bundle.atmosphere.fields()[index]),
        )


def test_mode_none_absorber_off_switch_and_deep_lid_leave_the_state_alone():
    _, model, state, g = _sponge_seam()
    model.physics = None
    bundle = _anomaly_bundle(model, state)
    # Off switch: no wind round trip at all, the bundle passes through.
    model.sponge_base_pa = 0.0
    out, info = model.apply_physics(bundle, 600.0)
    assert out is bundle and info == {"physics_mode": "none"}
    # Deep lid: a base below every ring-mean p_full (~978 Pa at the top)
    # selects no ring, so the pass must not spend the round trip either -
    # trajectories the absorber cannot affect pick up no analysis
    # roundoff.
    ring_top = float(np.mean(np.asarray(g["p_full"][0])))
    model.sponge_base_pa = 0.5 * ring_top
    out, info = model.apply_physics(bundle, 600.0)
    assert out is bundle and info == {"physics_mode": "none"}


def test_sponge_toml_door_defaults_identity_and_refusals(tmp_path):
    # Default-on: the smoke config carries no [sponge] table and the
    # defaults flow (fixed means default - a bare run is absorbed).
    cfg = load_config(CONFIG)
    assert cfg.sponge_base_pa == 5000.0
    assert cfg.sponge_lid_relaxation_time_s == 900.0
    assert cfg.config_identity["sponge_base_pa"] == 5000.0
    assert cfg.config_identity["sponge_lid_relaxation_time_s"] == 900.0

    base_text = Path(CONFIG).read_text(encoding="utf-8")

    def write(name, sponge_table):
        path = tmp_path / name
        path.write_text(base_text + "\n" + sponge_table, encoding="utf-8")
        return path

    # An explicit [sponge] parses and moves the config hash: the absorber
    # is part of the run's arithmetic identity, so stale checkpoints
    # refuse by design.
    tuned = load_config(
        write("tuned.toml", "[sponge]\nbase_pa = 4000.0\n")
    )
    assert tuned.sponge_base_pa == 4000.0
    assert tuned.config_hash != cfg.config_hash

    with pytest.raises(ValueError, match="selects no ring"):
        load_config(write("negative.toml", "[sponge]\nbase_pa = -1.0\n"))
    with pytest.raises(ValueError, match="amplify"):
        load_config(
            write("zero_tau.toml", "[sponge]\nlid_relaxation_time_s = 0.0\n")
        )
    with pytest.raises(ValueError, match="unknown keys in \\[sponge\\]"):
        load_config(write("unknown.toml", "[sponge]\nsponge_base_pa = 1.0\n"))

    # Retired keys refuse by name through the [reference_physics] door:
    # silently dropping them would run the very unsponged configuration
    # that killed both T533 runs at the 140 K research gate.
    retired = tmp_path / "retired.toml"
    retired.write_text(
        base_text.replace(
            "[reference_physics]",
            "[reference_physics]\nsponge_base_pa = 5000.0",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="retired.*\\[sponge\\]"):
        load_config(retired)
    retired_tau = tmp_path / "retired_tau.toml"
    retired_tau.write_text(
        base_text.replace(
            "[reference_physics]",
            "[reference_physics]\nsponge_lid_relaxation_time_s = 900.0",
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="sponge_lid_relaxation_time_s.*retired"
    ):
        load_config(retired_tau)
