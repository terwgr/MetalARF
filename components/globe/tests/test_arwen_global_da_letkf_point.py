"""The point-observation LETKF on the sphere: its calibration families.

Every family is planted on a small Gaussian-like grid with a correlated
random ensemble; the truth of each is analytic.  Both directions are held:
a planted report reproduces the localised single-report gain everywhere
and moves nothing beyond the cutoff (bitwise); a report equal to the
ensemble mean of H(x) moves the mean by nothing but the rounding of a
zero-mean transform; no report moves nothing; a dense set reduces O-A
below O-B and the spread with it; the cap on local reports is applied and
counted; every degenerate input is refused by name.
"""
from __future__ import annotations

import numpy as np
import pytest

from woof.globe.constants import EARTH_RADIUS_M
from woof.globe.da.letkf_point import (
    ColumnGeometry,
    PointLetkfConfig,
    PointLetkfDiagnostics,
    _geodesic,
    analyze_points,
    analyze_points_with_control,
    flatten_batches,
    single_observation_increment,
)
from woof.globe.da.observations import PointObs
from woof.da.letkf import LetkfError, gaspari_cohn

from conftest import requires_gpu

NLAT, NLON, NLEV, R = 16, 32, 5, 8
HCUT_M = 3000.0e3
VCUT = 0.8


def _vertical(batch, surface):
    return np.full(batch.count, VCUT)


def _grid():
    lat = np.linspace(-80.0, 80.0, NLAT)
    lon = np.arange(NLON) * 360.0 / NLON
    lnp = np.log(np.linspace(20000.0, 100000.0, NLEV))[:, None, None] * np.ones((1, NLAT, NLON))
    lnps = np.log(1.0e5) * np.ones((NLAT, NLON))
    return lat, lon, lnp, lnps, ColumnGeometry(lat, lon, lnp, lnps)


def _smooth(rng, shape):
    a = rng.standard_normal(shape)
    for axis in range(a.ndim - 2, a.ndim):
        a = np.roll(a, 1, axis) + a + np.roll(a, -1, axis)
    return a


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(1)
    lat, lon, lnp, lnps, geometry = _grid()
    prior = {
        "theta": 300.0 + 2.0 * _smooth(rng, (R, NLEV, NLAT, NLON)),
        "u": 10.0 + 3.0 * _smooth(rng, (R, NLEV, NLAT, NLON)),
        "lnps": np.log(1.0e5) + 0.002 * _smooth(rng, (R, NLAT, NLON)),
    }
    return lat, lon, lnp, lnps, geometry, prior


def _flat(batches):
    return flatten_batches(batches, np, horizontal_cutoff_m=HCUT_M,
                           vertical_cutoff_for=_vertical, solve_dtype="float64")


def test_a_planted_report_reproduces_the_analytic_localised_gain_everywhere(world):
    lat, lon, lnp, lnps, geometry, prior = world
    j, i, k = 8, 10, 2
    sim = prior["theta"][:, k, j, i]
    y = sim.mean() + 1.5
    batch = PointObs("probe", "temperature_k", [lat[j]], [lon[i]], [lnp[k, j, i]], [False],
                     [y], [1.0], simulated=sim[:, None])
    inc = analyze_points(prior, _flat([batch]), geometry,
                         PointLetkfConfig(rtps_alpha=0.0, chunk_rings=4))
    lat_g, lon_g = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    wh = gaspari_cohn(_geodesic(np.deg2rad(lat[j]), np.deg2rad(lon[i]), lat_g, lon_g,
                                EARTH_RADIUS_M, np) / HCUT_M, 1.0)
    w3 = wh[None] * gaspari_cohn(np.abs(lnp - lnp[k, j, i]) / VCUT, 1.0)
    w2 = wh * gaspari_cohn(np.abs(lnps - lnp[k, j, i]) / VCUT, 1.0)
    for name, w in (("theta", w3), ("u", w3), ("lnps", w2)):
        analytic = single_observation_increment(prior[name], sim, y, 1.0, w)
        got = inc[name].mean(axis=0)
        assert np.max(np.abs(got)) > 0.0
        assert np.max(np.abs(analytic - got)) <= 1.0e-12 * np.max(np.abs(got)), name
        # Beyond the cutoff: bitwise zero for the mean and for every member.
        assert np.all(got[w == 0.0] == 0.0), name
        assert np.all(inc[name][:, w == 0.0] == 0.0), name
        # And the shape inside: the increment follows the weight's sign pattern
        # where the covariance is one sign (theta at the report's own column).
    assert np.any(w3 == 0.0), "the cutoff must leave some of the grid unreached"


def test_a_report_equal_to_the_ensemble_mean_moves_the_mean_by_rounding_only(world):
    lat, lon, lnp, lnps, geometry, prior = world
    j, i, k = 8, 10, 2
    sim = prior["theta"][:, k, j, i][:, None]
    batch = PointObs("probe", "temperature_k", [lat[j]], [lon[i]], [lnp[k, j, i]], [False],
                     [0.0], [1.0], simulated=sim)
    flat = _flat([batch])
    flat.value = flat.simbar.copy()
    diag = PointLetkfDiagnostics()
    inc = analyze_points(prior, flat, geometry, PointLetkfConfig(rtps_alpha=0.9, chunk_rings=16), diag)
    assert diag.active_points > 0
    for name in prior:
        scale = float(np.max(np.abs(prior[name])))
        assert float(np.max(np.abs(inc[name].mean(axis=0)))) <= 1.0e-13 * scale, name
    # RTPS at alpha 1 restores the prior spread exactly where the transform acted.
    inc1 = analyze_points(prior, flat, geometry, PointLetkfConfig(rtps_alpha=1.0, chunk_rings=16))
    post = prior["theta"] + inc1["theta"]
    sb = np.sqrt(((prior["theta"] - prior["theta"].mean(0)) ** 2).sum(0) / (R - 1))
    sa = np.sqrt(((post - post.mean(0)) ** 2).sum(0) / (R - 1))
    assert np.allclose(sa, sb, rtol=1e-9, atol=1e-12)


def test_rtps_relaxes_the_posterior_spread_to_alpha_prior_plus_the_rest_posterior(world):
    """RTPS both directions on a dense set: the spread after the transform
    at alpha is ``alpha s_b + (1 - alpha) s_a`` with ``s_a`` the alpha-0
    posterior spread, at every active gridpoint, and a gridpoint the
    transform never touched keeps its perturbations (the 2026-09-06
    refutation's reading, held to 1e-12 of the prior spread)."""
    lat, lon, lnp, lnps, geometry, prior = world
    rng = np.random.default_rng(11)
    n = 300
    ol = rng.uniform(-70.0, 70.0, n)
    oo = rng.uniform(0.0, 360.0, n)
    ok = rng.integers(0, NLEV, n)
    jj = np.abs(lat[:, None] - ol[None, :]).argmin(axis=0)
    ii = np.abs(((lon[:, None] - oo[None, :] + 180.0) % 360.0) - 180.0).argmin(axis=0)
    sims = np.stack([prior["theta"][:, ok[m], jj[m], ii[m]] for m in range(n)], axis=1)
    values = sims.mean(axis=0) + rng.normal(0.0, 1.0, n)
    batch = PointObs("dense", "temperature_k", ol, oo, lnp[ok, jj, ii], np.zeros(n, dtype=bool),
                     values, np.ones(n), simulated=sims)
    inc0 = analyze_points(prior, _flat([batch]), geometry, PointLetkfConfig(rtps_alpha=0.0, chunk_rings=5))
    inc9 = analyze_points(prior, _flat([batch]), geometry, PointLetkfConfig(rtps_alpha=0.9, chunk_rings=5))

    def spread(x):
        return np.sqrt(((x - x.mean(0)) ** 2).sum(0) / (R - 1))

    sb = spread(prior["theta"])
    sa = spread(prior["theta"] + inc0["theta"])
    s9 = spread(prior["theta"] + inc9["theta"])
    active = np.abs(inc0["theta"].mean(0)) > 0.0
    assert active.sum() > 100
    assert np.all(sa[active] < sb[active])
    assert np.max(np.abs(s9 - (0.9 * sb + 0.1 * sa))[active] / sb[active]) < 1e-12
    assert np.all(inc9["theta"][:, ~active] == 0.0)


def test_no_report_moves_nothing_and_rho_inflates_everywhere(world):
    lat, lon, lnp, lnps, geometry, prior = world
    inc = analyze_points(prior, _flat([]), geometry, PointLetkfConfig(rtps_alpha=0.9))
    assert all(np.all(v == 0.0) for v in inc.values())
    rho = 1.21
    inc = analyze_points(prior, _flat([]), geometry, PointLetkfConfig(rtps_alpha=0.0, prior_inflation=rho))
    pert = prior["theta"] - prior["theta"].mean(0)
    assert np.allclose(inc["theta"], (np.sqrt(rho) - 1.0) * pert)


def test_a_dense_report_set_reduces_o_minus_a_and_the_spread_and_the_cap_is_counted(world):
    lat, lon, lnp, lnps, geometry, prior = world
    rng = np.random.default_rng(3)
    n = 600
    ol = rng.uniform(-60.0, 60.0, n)
    oo = rng.uniform(0.0, 360.0, n)
    ok = rng.integers(0, NLEV, n)
    jj = np.argmin(np.abs(lat[:, None] - ol[None, :]), axis=0)
    ii = (np.round(oo / (360.0 / NLON)).astype(int)) % NLON
    sims = prior["theta"][:, ok, jj, ii]
    ys = sims.mean(axis=0) + rng.normal(0.0, 1.0, n)
    batch = PointObs("dense", "temperature_k", ol, oo, lnp[ok, jj, ii], np.zeros(n, bool),
                     ys, np.ones(n), simulated=sims)
    flat = flatten_batches([batch], np, horizontal_cutoff_m=2500.0e3,
                           vertical_cutoff_for=_vertical, solve_dtype="float64")
    diag = PointLetkfDiagnostics()
    inc = analyze_points(prior, flat, geometry, PointLetkfConfig(rtps_alpha=0.9, max_local_obs=20), diag)
    assert diag.max_local_obs > 20 and diag.max_padded_slots == 20 and diag.dropped_by_cap > 0
    post = prior["theta"] + inc["theta"]
    o_b = np.sqrt(np.mean((ys - sims.mean(0)) ** 2))
    o_a = np.sqrt(np.mean((ys - post[:, ok, jj, ii].mean(0)) ** 2))
    assert o_a < o_b
    assert diag.posterior_spread["theta"] < diag.prior_spread["theta"]
    # The cap changes the answer only where it acted: without it the fit is at least as good.
    inc_full = analyze_points(prior, flat, geometry, PointLetkfConfig(rtps_alpha=0.9, max_local_obs=10_000))
    post_full = prior["theta"] + inc_full["theta"]
    o_a_full = np.sqrt(np.mean((ys - post_full[:, ok, jj, ii].mean(0)) ** 2))
    assert o_a_full <= o_a * 1.05


def test_chunking_does_not_change_the_answer(world):
    lat, lon, lnp, lnps, geometry, prior = world
    rng = np.random.default_rng(5)
    n = 80
    ol = rng.uniform(-70.0, 70.0, n)
    oo = rng.uniform(0.0, 360.0, n)
    ok = rng.integers(0, NLEV, n)
    jj = np.argmin(np.abs(lat[:, None] - ol[None, :]), axis=0)
    ii = (np.round(oo / (360.0 / NLON)).astype(int)) % NLON
    sims = prior["u"][:, ok, jj, ii]
    batch = PointObs("wind", "wind_u_m_s", ol, oo, lnp[ok, jj, ii], np.zeros(n, bool),
                     sims.mean(0) + rng.normal(0.0, 1.5, n), np.full(n, 1.5), simulated=sims)
    flat = _flat([batch])
    a = analyze_points(prior, flat, geometry, PointLetkfConfig(rtps_alpha=0.9, chunk_rings=1))
    b = analyze_points(prior, flat, geometry, PointLetkfConfig(rtps_alpha=0.9, chunk_rings=16))
    for name in prior:
        assert np.allclose(a[name], b[name], rtol=1e-10, atol=1e-12), name


def test_a_ring_taken_in_longitude_segments_is_the_same_analysis(world):
    """The fallback a full card reaches: a chunk of one ring cut into
    longitude segments (down to single columns) solves every column on its
    own reports, so the analysis and the counted columns are the whole
    ring's; only the footprint changes."""
    lat, lon, lnp, lnps, geometry, prior = world
    rng = np.random.default_rng(6)
    n = 80
    ol = rng.uniform(-70.0, 70.0, n)
    oo = rng.uniform(0.0, 360.0, n)
    ok = rng.integers(0, NLEV, n)
    jj = np.argmin(np.abs(lat[:, None] - ol[None, :]), axis=0)
    ii = (np.round(oo / (360.0 / NLON)).astype(int)) % NLON
    sims = prior["u"][:, ok, jj, ii]
    batch = PointObs("wind", "wind_u_m_s", ol, oo, lnp[ok, jj, ii], np.zeros(n, bool),
                     sims.mean(0) + rng.normal(0.0, 1.5, n), np.full(n, 1.5), simulated=sims)
    flat = _flat([batch])
    whole = PointLetkfDiagnostics()
    a = analyze_points(prior, flat, geometry, PointLetkfConfig(rtps_alpha=0.9, chunk_rings=1), whole)
    for segments in (4, 7, NLON):
        diag = PointLetkfDiagnostics()
        b = analyze_points(prior, flat, geometry,
                           PointLetkfConfig(rtps_alpha=0.9, chunk_rings=1, chunk_segments=segments), diag)
        for name in prior:
            assert np.allclose(a[name], b[name], rtol=1e-10, atol=1e-12), (name, segments)
        assert diag.chunk_segments == segments
        assert diag.chunk_rings == 1 and whole.chunk_rings == 1 and whole.chunk_segments == 1
        assert diag.active_columns == whole.active_columns
        assert diag.active_points == whole.active_points
        assert diag.max_local_obs == whole.max_local_obs
        assert diag.chunks >= whole.chunks
    with pytest.raises(LetkfError, match="chunk_segments"):
        PointLetkfConfig(rtps_alpha=0.9, chunk_segments=0)


def test_refusals_name_their_breakage(world):
    lat, lon, lnp, lnps, geometry, prior = world
    j, i, k = 8, 10, 2
    sim = prior["theta"][:, k, j, i][:, None]
    unsimulated = PointObs("probe", "temperature_k", [lat[j]], [lon[i]], [lnp[k, j, i]], [False], [1.0], [1.0])
    with pytest.raises(LetkfError, match="no simulated"):
        _flat([unsimulated])
    unfilled = PointObs("probe", "temperature_k", [lat[j]], [lon[i]], [np.nan], [True], [1.0], [1.0], simulated=sim)
    with pytest.raises(LetkfError, match="ln_pressure"):
        _flat([unfilled])
    good = PointObs("probe", "temperature_k", [lat[j]], [lon[i]], [lnp[k, j, i]], [False], [1.0], [1.0], simulated=sim)
    flat = _flat([good])
    # One field the members agree on to rounding sits out the solve with a
    # zero increment and is named; an ensemble with no spread in any field
    # is refused.
    constant = dict(prior)
    constant["theta"] = np.broadcast_to(prior["theta"][:1], prior["theta"].shape).copy()
    diag = PointLetkfDiagnostics()
    inc = analyze_points(constant, flat, geometry, PointLetkfConfig(rtps_alpha=0.0), diag)
    assert diag.zero_spread_fields == ["theta"] and not np.asarray(inc["theta"]).any()
    all_constant = {n: np.broadcast_to(prior[n][:1], prior[n].shape).copy() for n in prior}
    with pytest.raises(LetkfError, match="no usable ensemble spread"):
        analyze_points(all_constant, flat, geometry, PointLetkfConfig(rtps_alpha=0.0))
    with pytest.raises(LetkfError, match="rtps_alpha"):
        PointLetkfConfig(rtps_alpha=1.5)
    with pytest.raises(LetkfError, match="ln_p_full"):
        ColumnGeometry(lat, lon, lnp[:, :4], lnps)


def test_column_geometry_reads_device_shaped_arrays_without_a_host_conversion():
    # A cupy array refuses an implicit numpy conversion; the geometry reads
    # shapes from the arrays themselves (the T127 analysis on the RTX 5090
    # died on this, twice: the second time on an eagerly evaluated getattr
    # default).
    class DeviceShaped:
        def __init__(self, shape):
            self.shape = shape

        def __array__(self, *args, **kwargs):
            raise TypeError("Implicit conversion to a NumPy array is not allowed")

    geometry = ColumnGeometry(
        latitude_deg=np.linspace(-80.0, 80.0, 4), longitude_deg=np.arange(0.0, 360.0, 45.0),
        ln_p_full=DeviceShaped((5, 4, 8)), ln_ps=DeviceShaped((4, 8)),
    )
    assert (geometry.nlev, geometry.nlat, geometry.nlon) == (5, 4, 8)
    with pytest.raises(LetkfError, match="ln_p_full must be"):
        ColumnGeometry(latitude_deg=np.linspace(-80.0, 80.0, 4), longitude_deg=np.arange(0.0, 360.0, 45.0),
                       ln_p_full=DeviceShaped((5, 3, 8)), ln_ps=DeviceShaped((4, 8)))


def test_the_level_batch_size_does_not_change_the_answer_and_the_receipt_names_the_path(world):
    """Every plane of a column sub-chunk enters one eigendecomposition
    batch; a budget that forces one column per batch and one ring per
    chunk has to give the default batching's increments to rounding (the
    batched products associate differently), and the diagnostics have to
    say where it ran."""
    lat, lon, lnp, lnps, geometry, prior = world
    rng = np.random.default_rng(21)
    n = 40
    rows = PointObs(
        stream="s", variable="temperature_k",
        latitude_deg=rng.uniform(-60.0, 60.0, n), longitude_deg=rng.uniform(0.0, 360.0, n),
        ln_pressure=np.log(rng.uniform(30000.0, 95000.0, n)), surface=np.zeros(n, dtype=bool),
        value=300.0 + rng.standard_normal(n), error=np.full(n, 1.0),
        simulated=300.0 + 0.5 * rng.standard_normal((R, n)),
        control_simulated=(300.0 + 0.3 * rng.standard_normal(n))[None],
    )
    flat = flatten_batches([rows], np, horizontal_cutoff_m=HCUT_M, vertical_cutoff_for=_vertical,
                           solve_dtype="float64", control=True)
    wide = PointLetkfDiagnostics()
    default = analyze_points_with_control(
        {k: v.copy() for k, v in prior.items()}, flat, geometry, PointLetkfConfig(rtps_alpha=0.9), wide)
    narrow = PointLetkfDiagnostics()
    tiny = analyze_points_with_control(
        {k: v.copy() for k, v in prior.items()}, flat, geometry,
        PointLetkfConfig(rtps_alpha=0.9, chunk_rings=1, memory_budget_mib=0.001), narrow)
    assert wide.path == "host" and narrow.path == "host"
    assert wide.wall_seconds > 0.0
    assert narrow.level_batches > wide.level_batches
    assert narrow.max_batch_points == NLEV + 1        # one column, every plane
    assert wide.active_points == narrow.active_points
    for name in prior:
        assert np.allclose(default.increments[name], tiny.increments[name], rtol=1e-10, atol=1e-12), name
        assert np.allclose(default.control_increment[name], tiny.control_increment[name], rtol=1e-10, atol=1e-12), name


def test_far_columns_take_the_closed_form_when_rho_is_not_one(world):
    """A column no report reaches, in a chunk that solves other columns
    or in a band no candidate touches, gets ``(s - 1) x'`` exactly, with
    ``s = (1 - alpha) sqrt(rho) + alpha``; at rho = 1 it gets exact
    zeros."""
    lat, lon, lnp, lnps, geometry, prior = world
    one = PointObs(
        stream="s", variable="temperature_k",
        latitude_deg=np.array([10.0]), longitude_deg=np.array([100.0]),
        ln_pressure=np.log(np.array([50000.0])), surface=np.array([False]),
        value=np.array([301.0]), error=np.array([1.0]),
        simulated=300.0 + 0.5 * np.random.default_rng(4).standard_normal((R, 1)),
    )
    flat = flatten_batches([one], np, horizontal_cutoff_m=1500.0e3, vertical_cutoff_for=_vertical,
                           solve_dtype="float64")
    rho, alpha = 1.21, 0.4
    inc = analyze_points({k: v.copy() for k, v in prior.items()}, flat, geometry,
                         PointLetkfConfig(rtps_alpha=alpha, prior_inflation=rho))
    s = (1.0 - alpha) * np.sqrt(rho) + alpha
    dist = _geodesic(np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :],
                     np.deg2rad(10.0), np.deg2rad(100.0), EARTH_RADIUS_M, np)
    far = dist >= 1500.0e3
    assert far.sum() > NLAT * NLON // 2
    for name, field in prior.items():
        pert = field - field.mean(axis=0, keepdims=True)
        expected = pert * np.float64(s - 1.0)
        got = inc[name]
        assert np.array_equal(got[..., far], expected[..., far]), name
        assert not np.array_equal(got[..., ~far], expected[..., ~far]), name
    zero = analyze_points({k: v.copy() for k, v in prior.items()}, flat, geometry,
                          PointLetkfConfig(rtps_alpha=alpha, prior_inflation=1.0))
    for name in prior:
        assert np.all(zero[name][..., far] == 0.0) and np.all(np.signbit(zero[name][..., far]) == False), name  # noqa: E712


def test_the_local_cap_keeps_the_lowest_indexed_reports_among_tied_weights(world):
    """Reports at one position tie in horizontal weight (every level of a
    sounding does); the cap keeps the p largest weights and, among ties,
    the LOWEST report indices, so the selection is the same on every array
    module and does not depend on the sort implementation (numpy and
    cupy broke the ties differently and the device and host analyses
    differed by up to 0.4 of a field's maximum on the case, 2026-09-07).
    Reversing the report order therefore changes which reports are kept,
    and the answer with the same reports kept is the same."""
    lat, lon, lnp, lnps, geometry, prior = world
    rng = np.random.default_rng(9)
    n = 12
    rows = PointObs(
        stream="s", variable="temperature_k",
        latitude_deg=np.full(n, 10.0), longitude_deg=np.full(n, 100.0),          # one sounding
        ln_pressure=np.log(np.linspace(30000.0, 95000.0, n)), surface=np.zeros(n, dtype=bool),
        value=300.0 + rng.standard_normal(n), error=np.full(n, 1.0),
        simulated=300.0 + 0.5 * rng.standard_normal((R, n)),
    )
    flat = flatten_batches([rows], np, horizontal_cutoff_m=HCUT_M, vertical_cutoff_for=_vertical,
                           solve_dtype="float64")
    first = analyze_points({k: v.copy() for k, v in prior.items()}, flat, geometry,
                           PointLetkfConfig(rtps_alpha=0.0, max_local_obs=5))
    # The same reports handed over in reverse order: the cap now keeps what
    # were the last five, so the analysis moves; handed over twice in the
    # same order it does not.
    flipped = rows.subset(np.arange(n)[::-1])
    flat_flipped = flatten_batches([flipped], np, horizontal_cutoff_m=HCUT_M, vertical_cutoff_for=_vertical,
                                   solve_dtype="float64")
    second = analyze_points({k: v.copy() for k, v in prior.items()}, flat_flipped, geometry,
                            PointLetkfConfig(rtps_alpha=0.0, max_local_obs=5))
    again = analyze_points({k: v.copy() for k, v in prior.items()}, flat, geometry,
                           PointLetkfConfig(rtps_alpha=0.0, max_local_obs=5))
    assert any(not np.array_equal(first[k], second[k]) for k in prior)
    assert all(np.array_equal(first[k], again[k]) for k in prior)
    # With the cap above the count every report is kept and the order of
    # the reports cannot matter beyond rounding.
    wide = analyze_points({k: v.copy() for k, v in prior.items()}, flat, geometry,
                          PointLetkfConfig(rtps_alpha=0.0, max_local_obs=n))
    wide_flipped = analyze_points({k: v.copy() for k, v in prior.items()}, flat_flipped, geometry,
                                  PointLetkfConfig(rtps_alpha=0.0, max_local_obs=n))
    for k in prior:
        assert np.allclose(wide[k], wide_flipped[k], rtol=1e-9, atol=1e-12), k


@requires_gpu
def test_the_device_solve_matches_the_host_solve_within_the_states_own_precision(world):
    """The contract the shipped default rests on: the localised solve on the
    card and the same code in numpy, from one float32 prior (the state
    dtype of record) and one report set that ties at the cap (soundings:
    every level at one position) with a control innovation, give the same
    receipts (active columns and points, the widest local count) and the
    same increments to the state's own precision.  The bound is 64 units
    of float32 roundoff on the field's own magnitude: the perturbations
    are formed in float32 on both paths and the two array modules reduce
    the member mean in different orders, so an increment differs by a few
    tens of ulps of the STATE (measured on the case's real hour, 75,385
    reports, 2026-09-07: 19 ulps on ln ps, 7 on theta, well under one on
    u, v and qv), never by a fraction of the increment.  The control
    increment is formed in the solve dtype and agrees more closely.
    Refuted on 2026-09-07 before the cap's tie rule: 0.4 of a field's
    maximum."""
    import cupy as cp

    lat, lon, lnp, lnps, geometry, prior = world
    rng = np.random.default_rng(11)
    batches = []
    for k in range(6):                                   # six soundings, two of them at one position
        n = 9
        pos_lat, pos_lon = (-30.0 + 12.0 * k, 20.0 + 50.0 * k) if k < 5 else (-30.0 + 12.0 * 2, 20.0 + 50.0 * 2)
        batches.append(PointObs(
            stream="s", variable="temperature_k",
            latitude_deg=np.full(n, pos_lat), longitude_deg=np.full(n, pos_lon),
            ln_pressure=np.log(np.linspace(30000.0, 95000.0, n)), surface=np.zeros(n, dtype=bool),
            value=300.0 + rng.standard_normal(n), error=np.full(n, 1.0),
            simulated=300.0 + 0.5 * rng.standard_normal((R, n)),
            control_simulated=(300.0 + 0.3 * rng.standard_normal((1, n))),
        ))
    prior32 = {k: np.asarray(v, dtype=np.float32) for k, v in prior.items()}
    cfg = PointLetkfConfig(rtps_alpha=0.5, prior_inflation=1.02, max_local_obs=12, chunk_rings=4)
    outs = {}
    diags = {}
    for xp in (np, cp):
        flat = flatten_batches(batches, xp, horizontal_cutoff_m=HCUT_M, vertical_cutoff_for=_vertical,
                               solve_dtype="float64", control=True)
        geom = ColumnGeometry(lat, lon, xp.asarray(lnp), xp.asarray(lnps))
        diag = PointLetkfDiagnostics()
        out = analyze_points_with_control({k: xp.asarray(v) for k, v in prior32.items()}, flat, geom, cfg, diag)
        to_np = (lambda a: cp.asnumpy(a)) if xp is cp else np.asarray
        outs[diag.path] = ({k: to_np(v) for k, v in out.increments.items()},
                           {k: to_np(v) for k, v in out.control_increment.items()})
        diags[diag.path] = diag
    assert set(outs) == {"host", "device"}
    host, device = diags["host"], diags["device"]
    assert (host.active_columns, host.active_points, host.max_local_obs) == (
        device.active_columns, device.active_points, device.max_local_obs)
    assert host.dropped_by_cap == device.dropped_by_cap
    assert host.active_points > 0 and host.dropped_by_cap > 0
    eps = np.finfo(np.float32).eps
    for k in prior32:
        scale = float(np.abs(prior32[k]).max())
        inc_h, inc_d = outs["host"][0][k], outs["device"][0][k]
        assert inc_d.dtype == np.float32 and inc_h.dtype == np.float32
        assert np.abs(inc_d.astype(np.float64) - inc_h).max() <= 64 * eps * scale, k
        ctl_h, ctl_d = outs["host"][1][k], outs["device"][1][k]
        assert np.abs(ctl_d.astype(np.float64) - ctl_h).max() <= 64 * eps * scale, k
        assert abs(host.mean_increment_rms[k] - device.mean_increment_rms[k]) <= 1e-5 * max(host.mean_increment_rms[k], 1e-30)


@requires_gpu
def test_the_door_routes_the_solve_to_the_path_named_and_hands_the_increments_back_on_the_members_namespace(world):
    """``solve_on_path`` is the seam ``--letkf-solve-path`` reaches: ``host``
    moves a device-resident prior, its reports and its geometry to numpy,
    solves there (the receipt then names ``host`` and LAPACK), frees the
    device copies it was handed and hands the increments and the control
    increment back as device arrays, bit for bit what the same numpy call
    gives directly; ``auto`` on a device prior solves on the card and names
    ``device``; ``device`` on a numpy prior is refused by name.  No test
    named the seam before 2026-09-07; the epsilon of record was measured on
    the solver called directly, this is what connects it to the door."""
    import cupy as cp

    from woof.globe.da.analysis import solve_on_path

    lat, lon, lnp, lnps, geometry, prior = world
    rng = np.random.default_rng(5)
    n = 9
    batches = [PointObs(
        stream="s", variable="temperature_k",
        latitude_deg=np.full(n, 10.0), longitude_deg=np.full(n, 40.0),
        ln_pressure=np.log(np.linspace(30000.0, 95000.0, n)), surface=np.zeros(n, dtype=bool),
        value=300.0 + rng.standard_normal(n), error=np.full(n, 1.0),
        simulated=300.0 + 0.5 * rng.standard_normal((R, n)),
        control_simulated=300.0 + 0.3 * rng.standard_normal((1, n)),
    )]
    prior32 = {k: np.asarray(v, dtype=np.float32) for k, v in prior.items()}
    cfg = PointLetkfConfig(rtps_alpha=0.5, max_local_obs=12, chunk_rings=4)
    flat_np = flatten_batches(batches, np, horizontal_cutoff_m=HCUT_M, vertical_cutoff_for=_vertical,
                              solve_dtype="float64", control=True)
    reference = analyze_points_with_control(dict(prior32), flat_np, geometry, cfg, PointLetkfDiagnostics())

    flat_cp = flatten_batches(batches, cp, horizontal_cutoff_m=HCUT_M, vertical_cutoff_for=_vertical,
                              solve_dtype="float64", control=True)
    geometry_cp = ColumnGeometry(lat, lon, cp.asarray(lnp), cp.asarray(lnps))
    prior_cp = {k: cp.asarray(v) for k, v in prior32.items()}
    diag = PointLetkfDiagnostics()
    routed = solve_on_path(prior_cp, flat_cp, geometry_cp, cfg, diag, solve_path="host", xp=cp, to_numpy=cp.asnumpy)
    assert (diag.path, diag.eigensolver) == ("host", "library")
    assert prior_cp == {}, "the host route frees the device prior it was handed"
    for k in prior32:
        assert isinstance(routed.increments[k], cp.ndarray) and isinstance(routed.control_increment[k], cp.ndarray)
        assert np.array_equal(cp.asnumpy(routed.increments[k]), reference.increments[k]), k
        assert np.array_equal(cp.asnumpy(routed.control_increment[k]), reference.control_increment[k]), k

    prior_cp = {k: cp.asarray(v) for k, v in prior32.items()}
    diag = PointLetkfDiagnostics()
    on_card = solve_on_path(prior_cp, flat_cp, geometry_cp, cfg, diag, solve_path="auto", xp=cp, to_numpy=cp.asnumpy)
    assert (diag.path, diag.eigensolver) == ("device", "jacobi")
    assert all(isinstance(on_card.increments[k], cp.ndarray) for k in prior32)

    with pytest.raises(ValueError, match="solve_path 'device'"):
        solve_on_path(dict(prior32), flat_np, geometry, cfg, PointLetkfDiagnostics(),
                      solve_path="device", xp=np, to_numpy=np.asarray)
