"""The refutation families on the WOOF global ensemble filter: what the
filter's own families did not hold, both directions where a direction
exists.

Held here: two identical reports equal one report at ``sigma / sqrt(2)``
(the R^-1 accumulation and the padded slots); a report at the dateline
updates both sides the way the same report at Greenwich does, and a report
beside the pole is finite and active (the periodic gather and the geodesic
metric); RTPS on a planted spread deficit relaxes the posterior spread to
``(1 - alpha) sigma_a + alpha sigma_b`` pointwise, and the Desroziers
background ratio reads the deficit in the deficit direction (a spread half
as large reads four); the neutral operators reproduce the closed forms of
the door's reductions at one Gaussian column (station pressure, 2 m
temperature, ln p interpolation); a foreign stream the analysis cannot
re-evaluate reads INCOMPLETE and fails the engineering gate; the recentring
shift keeps every member's global-mean surface pressure (the raw shift
moved it); the spread is area-weighted like the rmse it is set beside.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import dataclasses
import datetime as dt
import math

import numpy as np
import pytest

from woof.globe.config import load_config
from woof.globe.constants import (
    DRY_AIR_GAS_CONSTANT,
    EARTH_RADIUS_M,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
)
from woof.globe.da import (
    ControlBackground,
    EnsembleOptions,
    FilterOptions,
    GlobalEnsemble,
    PointObs,
    analyze_ensemble,
    ensemble_config,
    recenter,
)
from woof.globe.da.analysis import desroziers
from woof.globe.da.letkf_point import (
    ColumnGeometry,
    PointLetkfConfig,
    PointLetkfDiagnostics,
    analyze_points,
    flatten_batches,
)
from woof.globe.da.operators import MemberOperators, batches_from_rows
from woof.globe.assimilate import SURFACE_LAPSE_K_M, _interp_ln_pressure
from woof.globe.obs_table import ObsRow
from woof.globe.runner import build_model_and_cold_state, build_transform

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
EPOCH = dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc)

NLAT, NLON, NLEV, R = 16, 32, 5, 8
HCUT_M = 3000.0e3
VCUT = 0.8


def _vertical(batch, surface):
    return np.full(batch.count, VCUT)


def _smooth(rng, shape):
    a = rng.standard_normal(shape)
    for axis in range(a.ndim - 2, a.ndim):
        a = np.roll(a, 1, axis) + a + np.roll(a, -1, axis)
    return a


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(17)
    lat = np.linspace(-80.0, 80.0, NLAT)
    lon = np.arange(NLON) * 360.0 / NLON
    lnp = np.log(np.linspace(20000.0, 100000.0, NLEV))[:, None, None] * np.ones((1, NLAT, NLON))
    lnps = np.log(1.0e5) * np.ones((NLAT, NLON))
    prior = {
        "theta": 300.0 + 2.0 * _smooth(rng, (R, NLEV, NLAT, NLON)),
        "u": 10.0 + 3.0 * _smooth(rng, (R, NLEV, NLAT, NLON)),
        "lnps": np.log(1.0e5) + 0.002 * _smooth(rng, (R, NLAT, NLON)),
    }
    return lat, lon, lnp, lnps, ColumnGeometry(lat, lon, lnp, lnps), prior


def _flat(batches, hcut=HCUT_M):
    return flatten_batches(batches, np, horizontal_cutoff_m=hcut,
                           vertical_cutoff_for=_vertical, solve_dtype="float64")


# ---------------------------------------------------------------------------
# Duplicated reports
# ---------------------------------------------------------------------------

def test_two_identical_reports_equal_one_report_at_sigma_over_root_two(world):
    lat, lon, lnp, lnps, geometry, prior = world
    j, i, k = 6, 20, 3
    sim = prior["theta"][:, k, j, i]
    y = sim.mean() + 1.2
    one = PointObs("probe", "temperature_k", [lat[j]], [lon[i]], [lnp[k, j, i]], [False],
                   [y], [1.0 / math.sqrt(2.0)], simulated=sim[:, None])
    two = PointObs("probe", "temperature_k", [lat[j]] * 2, [lon[i]] * 2, [lnp[k, j, i]] * 2, [False] * 2,
                   [y, y], [1.0, 1.0], simulated=np.stack([sim, sim], axis=1))
    config = PointLetkfConfig(rtps_alpha=0.9, chunk_rings=5)
    a = analyze_points(prior, _flat([one]), geometry, config)
    b = analyze_points(prior, _flat([two]), geometry, config)
    for name in prior:
        scale = float(np.max(np.abs(a[name])))
        assert scale > 0.0
        assert np.max(np.abs(a[name] - b[name])) <= 1.0e-12 * scale, name


# ---------------------------------------------------------------------------
# The dateline and the pole
# ---------------------------------------------------------------------------

def test_a_report_at_the_dateline_updates_both_sides_as_at_greenwich(world):
    lat, lon, lnp, lnps, geometry, prior = world
    rng = np.random.default_rng(3)
    half = NLON // 2
    # Reports straddling the prime meridian, one of them ON a column at
    # longitude 0 and two off-column; the rotated set straddles the
    # dateline, one of them on the 180 column.
    obs_lon = np.array([0.0, 4.0, 355.0, 20.0])
    obs_lat = np.array([lat[9], lat[8], lat[7], lat[6]])
    obs_k = np.array([1, 2, 3, 0])
    jj = np.argmin(np.abs(lat[:, None] - obs_lat[None, :]), axis=0)
    ii = (np.round(obs_lon / (360.0 / NLON)).astype(int)) % NLON
    sims = prior["theta"][:, obs_k, jj, ii]
    ys = sims.mean(axis=0) + rng.normal(0.0, 1.0, obs_lon.size)
    batch = PointObs("probe", "temperature_k", obs_lat, obs_lon, lnp[obs_k, jj, ii], np.zeros(4, bool),
                     ys, np.ones(4), simulated=sims)
    rolled = {name: np.roll(field, half, axis=-1) for name, field in prior.items()}
    rolled_batch = PointObs("probe", "temperature_k", obs_lat, np.mod(obs_lon + 180.0, 360.0),
                            lnp[obs_k, jj, ii], np.zeros(4, bool), ys, np.ones(4),
                            simulated=np.roll(prior["theta"], half, axis=-1)[:, obs_k, jj, (ii + half) % NLON])
    assert np.array_equal(rolled_batch.simulated, sims)
    config = PointLetkfConfig(rtps_alpha=0.9, chunk_rings=4)
    a = analyze_points(prior, _flat([batch]), geometry, config)
    b = analyze_points(rolled, _flat([rolled_batch]), geometry, config)
    for name in prior:
        scale = float(np.max(np.abs(a[name])))
        assert scale > 0.0
        assert np.max(np.abs(np.roll(a[name], half, axis=-1) - b[name])) <= 1.0e-11 * scale, name
    # The dateline report reaches columns on both sides of it.
    got = b["theta"].mean(axis=0)[obs_k[2], jj[2]]
    east_of_dateline = got[(lon > 180.0) & (lon < 200.0)]
    west_of_dateline = got[(lon > 160.0) & (lon < 180.0)]
    assert np.any(east_of_dateline != 0.0) and np.any(west_of_dateline != 0.0)


def test_a_report_beside_the_pole_is_finite_and_reaches_the_polar_ring(world):
    lat, lon, lnp, lnps, geometry, prior = world
    j, i, k = NLAT - 1, 5, 2
    sim = prior["theta"][:, k, j, i]
    batch = PointObs("probe", "temperature_k", [89.0], [lon[i] + 1.0], [lnp[k, j, i]], [False],
                     [sim.mean() + 1.0], [1.0], simulated=sim[:, None])
    diag = PointLetkfDiagnostics()
    inc = analyze_points(prior, _flat([batch], hcut=2000.0e3), geometry, PointLetkfConfig(rtps_alpha=0.9), diag)
    assert diag.active_points > 0
    mean = inc["theta"].mean(axis=0)
    assert np.all(np.isfinite(mean))
    # Every column of the pole-most ring lies within 2,000 km of a report
    # at 89 N, so every column of that ring is active; the ring below at
    # 69.3 N lies 2,190 km away and is not.
    assert np.all(mean[k, NLAT - 1, :] != 0.0)
    assert np.all(inc["theta"][:, :, NLAT - 2, :] == 0.0)


# ---------------------------------------------------------------------------
# Inflation on a planted spread deficit
# ---------------------------------------------------------------------------

def test_rtps_on_a_planted_spread_deficit_relaxes_to_the_stated_blend_pointwise(world):
    lat, lon, lnp, lnps, geometry, prior = world
    rng = np.random.default_rng(5)
    # The deficit: the perturbations shrunk four times.
    deficit = {name: field.mean(axis=0, keepdims=True) + 0.25 * (field - field.mean(axis=0, keepdims=True))
               for name, field in prior.items()}
    n = 300
    ol = rng.uniform(-60.0, 60.0, n)
    oo = rng.uniform(0.0, 360.0, n)
    ok = rng.integers(0, NLEV, n)
    jj = np.argmin(np.abs(lat[:, None] - ol[None, :]), axis=0)
    ii = (np.round(oo / (360.0 / NLON)).astype(int)) % NLON
    # Reports drawn from the FULL-spread truth (a member of the un-shrunk
    # prior): the innovations are four times what the shrunk spread expects.
    truth = prior["theta"][0]
    sims = deficit["theta"][:, ok, jj, ii]
    ys = truth[ok, jj, ii] + rng.normal(0.0, 1.0, n)
    batch = PointObs("dense", "temperature_k", ol, oo, lnp[ok, jj, ii], np.zeros(n, bool), ys, np.ones(n), simulated=sims)
    flat = _flat([batch], hcut=2500.0e3)
    alpha = 0.9
    plain = analyze_points(deficit, flat, geometry, PointLetkfConfig(rtps_alpha=0.0, max_local_obs=60))
    relaxed = analyze_points(deficit, flat, geometry, PointLetkfConfig(rtps_alpha=alpha, max_local_obs=60))
    for name in deficit:
        xb = deficit[name]
        sb = np.sqrt(((xb - xb.mean(0)) ** 2).sum(0) / (R - 1))
        xa = xb + plain[name]
        sa = np.sqrt(((xa - xa.mean(0)) ** 2).sum(0) / (R - 1))
        xr = xb + relaxed[name]
        sr = np.sqrt(((xr - xr.mean(0)) ** 2).sum(0) / (R - 1))
        active = sa < sb * (1.0 - 1.0e-6)
        assert active.any(), name
        expected = (1.0 - alpha) * sa + alpha * sb
        assert np.allclose(sr[active], expected[active], rtol=1.0e-10, atol=0.0), name
        # The mean is untouched by the relaxation.
        assert np.allclose(xr.mean(0), xa.mean(0), rtol=1.0e-12, atol=1.0e-12), name
    # The instrument reads the deficit: the Desroziers background ratio on
    # the innovations against the shrunk spread is far above one (the full
    # truth's spread is four times the ensemble's, the ratio ~16 on the
    # spread term), and above one for a mild deficit too.
    d_ob = ys - sims.mean(axis=0)
    post = deficit["theta"] + plain["theta"]
    d_oa = ys - post[:, ok, jj, ii].mean(axis=0)
    spread_h = np.sqrt(((sims - sims.mean(0)) ** 2).sum(0) / (R - 1))
    out = desroziers(d_ob, d_oa, np.ones(n), spread_h)
    assert out["background_variance_ratio"] > 4.0
    # Both directions on the synthetic optimal-gain sample: half the spread
    # reads four, twice the spread reads a quarter.
    rng2 = np.random.default_rng(0)
    m = 200_000
    eps_b = rng2.normal(0.0, 2.0, m)
    eps_o = rng2.normal(0.0, 1.0, m)
    d = eps_o - eps_b
    d_a = d * (1.0 - 4.0 / 5.0)
    assert desroziers(d, d_a, np.ones(m), np.full(m, 1.0))["background_variance_ratio"] == pytest.approx(4.0, abs=0.08)
    assert desroziers(d, d_a, np.ones(m), np.full(m, 4.0))["background_variance_ratio"] == pytest.approx(0.25, abs=0.02)


# ---------------------------------------------------------------------------
# The operators against the closed forms at one column
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def smoke():
    base = load_config(CONFIG)
    det_cfg = dataclasses.replace(base, truncation=7, nlat=None, nlon=None, name="smoke-t7")
    det_transform = build_transform(det_cfg)
    det_model, det_cold = build_model_and_cold_state(det_cfg, det_transform)
    options = EnsembleOptions(members=6, truncation=3, seed=23)
    ecfg = ensemble_config(det_cfg, options)
    transform = build_transform(ecfg)
    model, cold = build_model_and_cold_state(ecfg, transform)
    return det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options


def test_the_operators_reproduce_the_closed_forms_at_a_gaussian_column(smoke):
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = smoke
    operators = MemberOperators.for_model(model, transform, ecfg)
    grid = transform.grid
    j, i = 3, 5
    lat, lon = float(grid.latitude_deg[j]), float(grid.longitude_deg[i])
    g = model.grid_state(cold.atmosphere, only=("ps", "theta", "qv", "p_full", "temperature"))
    ps = float(np.asarray(g["ps"])[j, i])
    theta_low = float(np.asarray(g["theta"])[-1, j, i])
    qv_low = max(float(np.asarray(g["qv"])[-1, j, i]), 0.0)
    p_full = np.asarray(g["p_full"])[:, j, i]
    t_profile = np.asarray(g["temperature"])[:, j, i]
    model.release_syntheses()
    # The operator's terrain reference is the SPECTRAL projection of the
    # model's grid terrain sampled at the station (the door's _ModelSpace
    # arithmetic, inherited).  On the smoke config the grid terrain is not
    # band-limited at the truncation, so the two differ by the
    # sub-truncation residual (measured: 4.3 m of 22 m at T3, 1.9 m at
    # this column, 22 Pa of station pressure); on the T63 and T127 native
    # configs the grid terrain is band-limited and the gap is 1.2 and 2.0
    # mm at the worst point (module notes, decision 24).  The closed forms
    # below take the operator's own reference so they test the reduction
    # arithmetic; the smoke-config gap is asserted to exist so a change to
    # either side flips this test knowingly.
    from woof.globe.spectral.sampling import sample_scalar
    z_grid = float(np.asarray(model.surface_geopotential)[j, i]) / GRAVITY_M_S2
    z_model = float(sample_scalar(transform, operators.terrain, np.array([lat]), np.array([lon]))[0]) / GRAVITY_M_S2
    assert abs(z_model - z_grid) > 0.1, "the smoke terrain's sub-truncation residual at this column"
    elevation = z_model + 37.0
    values, ln_pressure = operators.evaluate(
        [cold], np.array([lat, lat]), np.array([lon, lon]), np.array([elevation, 0.0]), np.array([np.nan, 50000.0]))
    # Closed forms (the door's reductions written out by hand).
    p_low = float(p_full[-1])
    t_low = theta_low * (p_low / REFERENCE_PRESSURE_PA) ** KAPPA
    tv_low = t_low * (1.0 + 0.61 * qv_low)
    z_low = z_model + DRY_AIR_GAS_CONSTANT * tv_low / GRAVITY_M_S2 * math.log(ps / p_low)
    p_station = ps * math.exp(-GRAVITY_M_S2 * (elevation - z_model) / (DRY_AIR_GAS_CONSTANT * tv_low))
    t_2m = t_low + SURFACE_LAPSE_K_M * (z_low - (elevation + 2.0))
    assert values["surface_pressure_pa"][0, 0] == pytest.approx(p_station, rel=1.0e-9)
    assert values["temperature_k"][0, 0] == pytest.approx(t_2m, rel=1.0e-9)
    assert ln_pressure[0] == pytest.approx(math.log(ps), rel=1.0e-9)
    # Aloft: linear in ln p between the two bracketing full levels.
    ln_p = np.log(p_full)
    target = math.log(50000.0)
    above = int(np.sum(ln_p <= target)) - 1
    below = above + 1
    w = (target - ln_p[above]) / (ln_p[below] - ln_p[above])
    t_500 = t_profile[above] * (1.0 - w) + t_profile[below] * w
    assert values["temperature_k"][0, 1] == pytest.approx(t_500, rel=1.0e-9)
    assert values["temperature_k"][0, 1] == pytest.approx(
        float(_interp_ln_pressure(t_profile[:, None], ln_p[:, None], np.array([target]))[0]), rel=1.0e-12)
    assert ln_pressure[1] == math.log(50000.0)
    assert np.isnan(values["surface_pressure_pa"][0, 1])


# ---------------------------------------------------------------------------
# INCOMPLETE, the recentring mass rule, the spread's weighting
# ---------------------------------------------------------------------------

def _rows(operators, truth, n=40, seed=0, when=EPOCH):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-70.0, 70.0, n)
    lon = rng.uniform(0.0, 360.0, n)
    values, _ = operators.evaluate([truth], lat, lon, np.zeros(n), np.full(n, np.nan))
    rows = []
    for k in range(n):
        for variable, error in (("surface_pressure_pa", 100.0), ("temperature_k", 1.0),
                                ("wind_u_m_s", 1.5), ("wind_v_m_s", 1.5)):
            rows.append(ObsRow("probe", f"S{k}", lat[k], lon[k], 0.0, None, when, variable,
                               float(values[variable][0, k]) + rng.normal(0.0, error), error))
    return rows


def _global_ps(transform, state) -> float:
    xp = transform.backend.xp
    return float(transform.grid.global_mean(np.asarray(xp.exp(transform.inverse(state.atmosphere.log_surface_pressure)))))


def test_a_foreign_stream_the_analysis_cannot_re_evaluate_reads_incomplete_and_fails_the_gate(smoke):
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = smoke
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    operators = MemberOperators.for_model(model, transform, ecfg)
    batches = batches_from_rows(_rows(operators, ensemble.members[1], n=30, seed=4), operators, ensemble.members)
    sim = 250.0 + np.arange(options.members)[:, None] * 0.3 + np.arange(3)[None, :]
    foreign = PointObs("sat", "brightness_temperature_k", [10.0, 12.0, 14.0], [20.0, 22.0, 24.0],
                       [math.log(50000.0)] * 3, [False] * 3, [251.0, 252.0, 250.5], [1.0] * 3, simulated=sim)
    result = analyze_ensemble(ensemble, batches + [foreign], FilterOptions(horizontal_cutoff_km=6000.0, thinning=False),
                              analysis_time=EPOCH)
    assert result.status == "fail"
    assert result.report["gate_of_record"]["incomplete"] == ["sat/brightness_temperature_k"]
    assert result.report["gate_of_record"]["passed"] is False
    assert result.report["assessments"]["engineering_validity"]["verdict"] == "fail"
    entry = result.report["streams"]["sat"]["brightness_temperature_k"]
    assert entry["verdict"].startswith("UNJUDGED")
    assert entry["regions"]["global"]["assimilated"]["o_minus_a"] is None
    # The neutral streams were judged as usual beside it.
    assert result.report["streams"]["probe"]["temperature_k"]["regions"]["global"]["assimilated"]["o_minus_a"] is not None


def test_recentring_keeps_every_members_global_mean_surface_pressure(smoke):
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = smoke
    operators = MemberOperators.for_model(model, transform, ecfg)
    filter_options = FilterOptions(horizontal_cutoff_km=6000.0, thinning=False, gate_minimum_count=1000,
                                   increment_application="direct")
    control = ControlBackground(det_cold, det_model, det_transform, det_cfg)

    def analysed():
        ens = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
        rows = _rows(operators, ens.members[2], n=40, seed=9)
        batches = batches_from_rows(rows, operators, ens.members)
        res = analyze_ensemble(ens, batches, filter_options, analysis_time=EPOCH, control=control)
        assert res.status == "pass"
        return ens, res

    ens, res = analysed()
    before = [_global_ps(transform, m) for m in ens.members]
    record = recenter(ens, res.control_analysis, det_transform, fraction=1.0, mode="increment",
                      control_increment=res.control_increment_spectral, mean_increment=res.mean_increment_spectral)
    after = [_global_ps(transform, m) for m in ens.members]
    assert record["preserve_global_mean_pressure"] is True
    for b, a in zip(before, after):
        assert a == pytest.approx(b, rel=1.0e-10)
    assert record["members_global_mean_surface_pressure_pa"]["after"] == pytest.approx(
        record["members_global_mean_surface_pressure_pa"]["before"], rel=1.0e-10)
    assert record["mass_preserving_log_offset"]["maxabs"] >= 0.0
    # The perturbations are untouched by the constant: each member's
    # departure from the mean is the same field before and after.
    # The raw shift (the first twins' form) moves the members' mean surface
    # pressure by the raw shift's global mean, which is not zero in general.
    ens2, res2 = analysed()
    before2 = [_global_ps(transform, m) for m in ens2.members]
    record2 = recenter(ens2, res2.control_analysis, det_transform, fraction=1.0, mode="increment",
                       control_increment=res2.control_increment_spectral, mean_increment=res2.mean_increment_spectral,
                       preserve_global_mean_pressure=False)
    after2 = [_global_ps(transform, m) for m in ens2.members]
    raw = record2["mass_preserving_log_offset"]["raw_shift_global_mean_ln_ps"]
    expected = np.mean(before2) * math.exp(raw)
    assert record2["preserve_global_mean_pressure"] is False
    assert np.mean(after2) == pytest.approx(expected, rel=1.0e-6)
    # And the kept form leaves the members' mean increment equal to the
    # control's increment in every degree but the constant.
    mean_after = ens.mean_spectral()["theta"]
    ens_ref, res_ref = analysed()
    ref = np.asarray(ens_ref.mean_spectral()["theta"]) - np.asarray(res_ref.mean_increment_spectral["theta"]) \
        + np.asarray(res_ref.control_increment_spectral["theta"])
    assert np.allclose(np.asarray(mean_after), ref, atol=1e-9)


def test_the_spread_is_area_weighted_like_the_rmse_beside_it(smoke):
    det_cfg, det_model, det_transform, det_cold, ecfg, model, transform, cold, options = smoke
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    weighted = ensemble.spread()
    equal = ensemble.spread(area_weighted=False)
    grid = transform.grid
    backend = transform.backend
    stack = np.stack([np.asarray(backend.to_numpy(model.grid_state(m.atmosphere, only=("temperature",))["temperature"]))
                      for m in ensemble.members])
    model.release_syntheses()
    var = ((stack - stack.mean(axis=0)) ** 2).sum(axis=0) / (options.members - 1)
    by_hand = math.sqrt(np.mean([grid.global_mean(level) for level in var]))
    assert weighted["temperature_k"] == pytest.approx(by_hand, rel=1.0e-12)
    assert equal["temperature_k"] == pytest.approx(math.sqrt(np.mean(var)), rel=1.0e-12)
    # On this grid the two instruments differ: the equal-weight form counts
    # the polar rings by 1/cos(latitude) more than their area.
    assert weighted["temperature_k"] != equal["temperature_k"]
