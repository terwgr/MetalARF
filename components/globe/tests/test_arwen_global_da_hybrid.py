"""The hybrid covariance of the WOOF global ensemble filter: the static
table, its estimation from lagged differences, its sampler, and the
augmented control solve.

Every claim is held both ways.  The linear balance through the
transform's operators reproduces the analytic n +- 1 recurrence of the
sphere's linear balance; a table planted with known variances, vertical
correlations and balance regressions is recovered by the estimator from
its own draws and reproduced by the sampler in the large-sample limit;
the augmented control solve reproduces the closed-form hybrid gain of one
report at every gridpoint, equals the pure solve at beta one bitwise, and
refuses a beta below one without draws by name; the options refuse what
they must and lay the door's settings over the package's.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from woof.globe.da.letkf_point import (
    ColumnGeometry,
    PointLetkfConfig,
    PointLetkfDiagnostics,
    _geodesic,
    analyze_points_with_control,
    flatten_batches,
    hybrid_single_observation_increment,
)
from woof.globe.da.observations import PointObs
from woof.globe.da.options import FilterOptions
from woof.globe.da.static_covariance import (
    CONTROL_VARIABLES,
    StaticCovariance,
    band_index_by_degree,
    default_bands,
    draw_static_perturbations,
    draw_static_spectral,
    estimate_static_covariance,
    linear_balance_geopotential,
    load_static_covariance,
    resolve_static_covariance,
    spectral_variance_by_degree,
)
from woof.globe.da_control import ControlOptions
from woof.da.letkf import LetkfError, gaspari_cohn
from woof.globe.spectral.constants import EARTH_ROTATION_RATE_S
from woof.globe.spectral.transform import SphericalHarmonicTransform
from woof.globe.spectral.vector import VorticityDivergenceOperator

from test_arwen_global_assimilate import (  # noqa: F401 - the fixture rides the import
    CONFIG, _synthetic_obs_files, spun_up,
)
from test_arwen_global_cycle import OPTIONS, START_TEXT

NLAT, NLON, NLEV, R, K = 16, 32, 5, 8, 12
HCUT_M = 3000.0e3
VCUT = 0.8


# ---------------------------------------------------------------------------
# The linear balance and the variance convention
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sphere():
    transform = SphericalHarmonicTransform.create(21, dealias_factor=1.5, backend="numpy", precision="float64")
    return transform, VorticityDivergenceOperator(transform)


def _epsilon(n: int, m: int) -> float:
    return math.sqrt((n * n - m * m) / (4.0 * n * n - 1.0)) if n > abs(m) else 0.0


def test_the_linear_balance_through_the_transform_is_the_analytic_n_plus_minus_one_recurrence(sphere):
    transform, vector = sphere
    t = transform.truncation
    a = transform.grid.radius_m
    omega = EARTH_ROTATION_RATE_S

    def lam(k):
        return -k * (k + 1) / a ** 2

    for n, m in ((2, 1), (5, 0), (7, 3), (12, 12), (20, 4)):
        psi = np.zeros((1, t + 1, t + 1), dtype=np.complex128)
        psi[0, n, m] = 1.0 + 0.5j if m else 1.0
        phi = linear_balance_geopotential(transform, vector, transform.laplacian(psi))[0]
        expected = np.zeros_like(phi)
        for k in (n - 1, n + 1):
            if k < m or k > t or k < 1:
                continue
            if k == n + 1:
                mu_z = _epsilon(k, m) * lam(n) * psi[0, n, m]
                mu_p = -(k - 1) * _epsilon(k, m) * psi[0, n, m]
            else:
                mu_z = _epsilon(k + 1, m) * lam(n) * psi[0, n, m]
                mu_p = (k + 2) * _epsilon(k + 1, m) * psi[0, n, m]
            expected[k, m] = 2.0 * omega * (mu_z + mu_p / a ** 2) / lam(k)
        assert np.abs(phi - expected).max() <= 1.0e-12 * np.abs(expected).max()
        # Nothing at the same degree: a same-degree regression of mass on
        # vorticity would read no balance at all.
        assert abs(phi[n, m]) <= 1.0e-12 * np.abs(expected).max()


def test_the_variance_by_degree_reads_the_draw_convention_back():
    from woof.globe.da.perturbations import _degree_weights, draw_coefficients

    rng = np.random.default_rng(0)
    sigma = _degree_weights(21, 21, -3.0)
    acc = np.zeros(22)
    for _ in range(400):
        acc += spectral_variance_by_degree(draw_coefficients(rng, 21, 1, sigma)[0])
    ratio = acc[1:] / 400.0 / sigma[1:] ** 2
    assert 0.9 < ratio.min() and ratio.max() < 1.12 and abs(ratio.mean() - 1.0) < 0.02


# ---------------------------------------------------------------------------
# A planted table: the sampler reproduces it, the estimator recovers it
# ---------------------------------------------------------------------------

def _planted_table(truncation: int, nlev: int) -> StaticCovariance:
    bands = default_bands(truncation)
    nb = len(bands)
    n = np.arange(truncation + 1, dtype=np.float64)
    spectrum = np.where(n >= 1, (n + 1.0) ** -3.0, 0.0)
    levels = np.linspace(1.0, 2.0, nlev)
    variance = {
        "vorticity": np.outer(spectrum, levels) * 1.0e-10,
        "divergence_unbalanced": np.outer(spectrum, levels) * 2.0e-11,
        "theta_unbalanced": np.outer(spectrum, levels[::-1]) * 4.0,
        "qv": np.outer(spectrum, levels) * 1.0e-7,
        "log_surface_pressure_unbalanced": spectrum * 1.0e-6,
    }
    variance["theta_unbalanced"][0] = 0.5
    variance["qv"][0] = 1.0e-8
    corr = np.zeros((nb, nlev, nlev))
    for b in range(nb):
        rho = 0.9 - 0.1 * b / max(nb - 1, 1)
        corr[b] = rho ** np.abs(np.subtract.outer(np.arange(nlev), np.arange(nlev)))
    vertical = {name: corr.copy() for name in CONTROL_VARIABLES}
    vertical["log_surface_pressure_unbalanced"] = np.ones((nb, 1, 1))
    theta_on_phi = np.zeros((nb, nlev, nlev))
    lnps_on_phi = np.zeros((nb, nlev))
    div_on_psi = np.zeros((nb, nlev, nlev))
    for b in range(nb):
        strength = 1.0 - 0.8 * b / max(nb - 1, 1)
        theta_on_phi[b] = strength * (0.03 * np.eye(nlev) + 0.01 * np.eye(nlev, k=1))
        lnps_on_phi[b] = strength * 1.0e-5 * np.linspace(0.2, 1.0, nlev)
        div_on_psi[b] = strength * 1.0e-12 * np.eye(nlev)
    return StaticCovariance(
        truncation=truncation, nlev=nlev, bands=np.asarray(bands), reference_p_full_hpa=np.linspace(100, 1000, nlev),
        variance=variance, vertical_correlation=vertical, theta_on_phi=theta_on_phi,
        lnps_on_phi=lnps_on_phi, divergence_on_psi=div_on_psi, receipt={"samples": 0, "version": "planted"},
    )


def test_the_table_round_trips_through_its_file_and_refuses_an_altered_one(tmp_path):
    table = _planted_table(21, 4)
    path = table.save(tmp_path / "static.npz")
    back = StaticCovariance.load(path)
    assert back.sha256() == table.sha256() and back.identity()["truncation"] == 21
    assert back.identity()["bands"] == [[int(a), int(b)] for a, b in default_bands(21)]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["variance__qv"] = arrays["variance__qv"] * 2.0
    np.savez_compressed(tmp_path / "altered.npz", **arrays)
    with pytest.raises(ValueError, match="altered"):
        StaticCovariance.load(tmp_path / "altered.npz")
    with pytest.raises(ValueError, match="above the table's truncation"):
        back.check_against(42, 4)
    with pytest.raises(ValueError, match="levels"):
        back.check_against(21, 5)


def test_the_sampler_reproduces_the_planted_variances_and_correlations(sphere):
    transform, vector = sphere
    table = _planted_table(transform.truncation, 4)
    rng = np.random.default_rng(3)
    draws = draw_static_spectral(table, transform.truncation, 600, rng)
    for name in ("vorticity", "theta_unbalanced"):
        estimate = np.mean([spectral_variance_by_degree(draws[name][k]).T for k in range(600)], axis=0)
        target = table.variance[name]
        keep = target > 0.0
        ratio = estimate[keep] / target[keep]
        assert 0.85 < ratio.min() and ratio.max() < 1.15, (name, ratio.min(), ratio.max())
        # Vertical correlation of the last band, pooled over its (n, m).
        index = band_index_by_degree(table.bands, transform.truncation)
        b = int(index[-1])
        block = draws[name][:, :, index == b, :]
        flat = block.reshape(600, 4, -1)
        s = np.real(np.einsum("kin,kjn->ij", flat, np.conj(flat)))
        d = np.sqrt(np.diag(s))
        corr = s / np.outer(d, d)
        assert np.abs(corr - table.vertical_correlation[name][b]).max() < 0.05


def test_the_estimator_recovers_a_planted_table_from_its_own_draws(sphere):
    transform, vector = sphere
    table = _planted_table(transform.truncation, 4)
    rng = np.random.default_rng(11)
    samples = draw_static_perturbations(table, transform, vector, 160, rng)["spectral"]
    differences = [
        ({"sample": k}, {name: samples[name][k] for name in ("vorticity", "divergence", "theta", "qv")}
                        | {"log_surface_pressure": samples["log_surface_pressure"][k]})
        for k in range(160)
    ]
    estimate = estimate_static_covariance(differences, transform, vector, bands=table.bands, ridge=1.0e-6)
    # The balance regressions come back.
    for b in range(len(table.bands)):
        scale = max(np.abs(table.theta_on_phi[b]).max(), 1.0e-300)
        assert np.abs(estimate.theta_on_phi[b] - table.theta_on_phi[b]).max() < 0.15 * scale, b
        scale_p = max(np.abs(table.lnps_on_phi[b]).max(), 1.0e-300)
        assert np.abs(estimate.lnps_on_phi[b] - table.lnps_on_phi[b]).max() < 0.15 * scale_p, b
    # The unbalanced variances come back (degrees with power).
    for name in ("theta_unbalanced", "vorticity", "qv", "log_surface_pressure_unbalanced"):
        target = table.variance[name]
        keep = target > 0.0
        ratio = estimate.variance[name][keep] / target[keep]
        assert 0.7 < ratio.min() and ratio.max() < 1.3, (name, ratio.min(), ratio.max())
    share = estimate.receipt["balance"]["balanced_share_of_variance"]["theta"]
    assert share[0] is not None and share[0] > 0.05
    assert estimate.receipt["samples"] == 160


def test_a_drift_shared_by_every_pair_stays_out_of_the_table(sphere):
    """A lagged difference carries the model's systematic drift between the
    two forecast ranges, the same sign in every pair (on the sample of
    record the global-mean theta at the lid read -114.6 K in all eleven
    pairs and 20 to 50 percent of the theta second moment at every level
    was the sample mean's).  The estimator removes the sample mean before
    it squares anything, so a constant offset laid on every difference
    changes no variance, no correlation and no regression, and the receipt
    records the share of the raw second moment the drift held."""
    transform, vector = sphere
    table = _planted_table(transform.truncation, 4)
    rng = np.random.default_rng(23)
    samples = draw_static_perturbations(table, transform, vector, 120, rng)["spectral"]
    fields = ("vorticity", "divergence", "theta", "qv")

    def differences(offset):
        out = []
        for k in range(120):
            diff = {name: samples[name][k].copy() for name in fields}
            diff["log_surface_pressure"] = samples["log_surface_pressure"][k].copy()
            if offset:
                diff["theta"][0, 0, 0] += 40.0 * math.sqrt(4.0 * math.pi)   # 40 K in the global mean at level 0
                diff["theta"][1, 3, 1] += 2.0 + 1.0j                        # a planetary-scale drift at level 1
                diff["log_surface_pressure"][2, 0] += 1.0e-1
            out.append(({"sample": k}, diff))
        return out

    clean = estimate_static_covariance(differences(False), transform, vector, bands=table.bands, ridge=1.0e-6)
    drifted = estimate_static_covariance(differences(True), transform, vector, bands=table.bands, ridge=1.0e-6)
    for name in CONTROL_VARIABLES:
        assert np.allclose(drifted.variance[name], clean.variance[name], rtol=1.0e-9, atol=0.0), name
        assert np.allclose(drifted.vertical_correlation[name], clean.vertical_correlation[name], atol=1.0e-9), name
    assert np.allclose(drifted.theta_on_phi, clean.theta_on_phi, rtol=1.0e-9, atol=1.0e-12)
    assert np.allclose(drifted.lnps_on_phi, clean.lnps_on_phi, rtol=1.0e-9, atol=1.0e-15)
    # The planted global-mean variance survives: 0.5 K^2 at degree 0, not 0.5 plus 40^2 times 4 pi.
    assert 0.6 < drifted.variance["theta_unbalanced"][0, 0] / 0.5 < 1.5
    drift = drifted.receipt["drift"]
    assert drift["removed"] is True and abs(drift["expected_share_if_no_drift"] - 1.0 / 120) < 1.0e-12
    assert abs(drift["theta_global_mean_drift_k_by_level"][0] - 40.0) < 0.5
    # The share the drift held of the raw second moment (the planted
    # balanced theta and ln ps carry most of the level's power, so the
    # share is a fraction, not one): well above the 1/120 a drift-free
    # sample leaves to its mean, and above the clean estimate's by an order.
    share = drift["share_of_raw_second_moment_by_level"]
    clean_share = clean.receipt["drift"]["share_of_raw_second_moment_by_level"]
    assert share["theta"][0] > 0.25 and share["theta"][0] > 10.0 * clean_share["theta"][0]
    assert share["log_surface_pressure"][0] > 0.25 and share["log_surface_pressure"][0] > 10.0 * clean_share["log_surface_pressure"][0]
    assert clean_share["theta"][0] < 0.05 and clean_share["log_surface_pressure"][0] < 0.05


def test_the_sampler_hands_back_grid_perturbations_of_the_analysis_fields(sphere):
    transform, vector = sphere
    table = _planted_table(transform.truncation, 4)
    out = draw_static_perturbations(table, transform, vector, 3, np.random.default_rng(5))
    assert set(out) == {"u", "v", "theta", "qv", "lnps", "spectral"}
    assert out["u"].shape == (3, 4, transform.grid.nlat, transform.grid.nlon)
    assert out["lnps"].shape == (3, transform.grid.nlat, transform.grid.nlon)
    assert all(np.all(np.isfinite(out[name])) for name in ("u", "v", "theta", "qv", "lnps"))
    # The balance is in the draw: theta carries a part the unbalanced draw
    # alone would not, so theta and the balanced geopotential correlate.
    phi = linear_balance_geopotential(transform, vector, np.asarray(out["spectral"]["vorticity"]))
    theta = out["spectral"]["theta"]
    corr = np.real(np.vdot(phi[:, :, 1:3, :], theta[:, :, 1:3, :])) / (
        np.linalg.norm(phi[:, :, 1:3, :]) * np.linalg.norm(theta[:, :, 1:3, :]))
    assert corr > 0.2


# ---------------------------------------------------------------------------
# The augmented control solve
# ---------------------------------------------------------------------------

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
    static = {
        "theta": 1.5 * _smooth(rng, (K, NLEV, NLAT, NLON)),
        "u": 2.0 * _smooth(rng, (K, NLEV, NLAT, NLON)),
        "lnps": 0.001 * _smooth(rng, (K, NLAT, NLON)),
    }
    return lat, lon, lnp, lnps, geometry, prior, static


def _flat(batches, *, static: bool):
    return flatten_batches(batches, np, horizontal_cutoff_m=HCUT_M, vertical_cutoff_for=_vertical,
                           solve_dtype="float64", control=True, static=static)


def _one_report(world, *, static: bool):
    lat, lon, lnp, lnps, geometry, prior, static_prior = world
    j, i, k = 8, 10, 2
    sim = prior["theta"][:, k, j, i]
    control_sim = sim.mean() + 0.4
    y = sim.mean() + 1.5
    batch = PointObs("probe", "temperature_k", [lat[j]], [lon[i]], [lnp[k, j, i]], [False],
                     [y], [1.0], simulated=sim[:, None], control_simulated=[[control_sim]],
                     static_simulated=static_prior["theta"][:, k, j, i][:, None] if static else None)
    return batch, (j, i, k), sim, control_sim, y


def test_the_augmented_solve_reproduces_the_closed_form_hybrid_gain_everywhere(world):
    lat, lon, lnp, lnps, geometry, prior, static_prior = world
    batch, (j, i, k), sim, control_sim, y = _one_report(world, static=True)
    beta = 0.6
    diagnostics = PointLetkfDiagnostics()
    out = analyze_points_with_control(
        prior, _flat([batch], static=True), geometry, PointLetkfConfig(rtps_alpha=0.0, chunk_rings=4),
        diagnostics, static_prior=static_prior, hybrid_beta=beta)
    assert diagnostics.hybrid_beta == beta and diagnostics.static_samples == K
    assert diagnostics.hybrid_solver == "batched-solve"
    lat_g, lon_g = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    dist = _geodesic(lat_g, lon_g, np.deg2rad(lat[j]), np.deg2rad(lon[i]), geometry.radius_m, np)
    wh = gaspari_cohn(dist / HCUT_M, 1.0)
    for name in ("theta", "u", "lnps"):
        if name == "lnps":
            wv = gaspari_cohn(np.abs(lnps - lnp[k, j, i]) / VCUT, 1.0)
            weight = wh * wv
        else:
            wv = gaspari_cohn(np.abs(lnp - lnp[k, j, i]) / VCUT, 1.0)
            weight = wh[None] * wv
        expected = hybrid_single_observation_increment(
            prior[name], static_prior[name], sim, static_prior["theta"][:, k, j, i], y, 1.0, weight, beta)
        # The control innovation is y - H(x_H^b), not y - mean H(x_k):
        # the closed form is the gain times that innovation.
        expected = expected / (y - sim.mean()) * (y - control_sim)
        got = out.control_increment[name]
        assert np.abs(got - expected).max() <= 1.0e-10 * max(np.abs(expected).max(), 1.0e-300), name
        assert np.all(got[weight == 0.0] == 0.0)


def test_beta_one_is_the_pure_solve_bitwise_and_a_beta_below_one_needs_draws(world):
    lat, lon, lnp, lnps, geometry, prior, static_prior = world
    batch, _, _, _, _ = _one_report(world, static=True)
    config = PointLetkfConfig(rtps_alpha=0.0, chunk_rings=4)
    pure = analyze_points_with_control(prior, _flat([batch], static=False), geometry, config)
    hybrid_at_one = analyze_points_with_control(
        prior, _flat([batch], static=True), geometry, config, static_prior=static_prior, hybrid_beta=1.0)
    for name in prior:
        assert np.array_equal(pure.control_increment[name], hybrid_at_one.control_increment[name])
        assert np.array_equal(pure.increments[name], hybrid_at_one.increments[name])
    with pytest.raises(LetkfError, match="no static draws"):
        analyze_points_with_control(prior, _flat([batch], static=False), geometry, config, hybrid_beta=0.5)
    with pytest.raises(LetkfError, match="hybrid_beta must lie"):
        analyze_points_with_control(prior, _flat([batch], static=True), geometry, config,
                                    static_prior=static_prior, hybrid_beta=0.0)
    with pytest.raises(LetkfError, match="static_simulated"):
        _flat([_one_report(world, static=False)[0]], static=True)


def test_the_members_keep_the_ensemble_transform_under_the_hybrid(world):
    lat, lon, lnp, lnps, geometry, prior, static_prior = world
    batch, _, _, _, _ = _one_report(world, static=True)
    config = PointLetkfConfig(rtps_alpha=0.0, chunk_rings=4)
    pure = analyze_points_with_control(prior, _flat([batch], static=False), geometry, config)
    hybrid = analyze_points_with_control(
        prior, _flat([batch], static=True), geometry, config, static_prior=static_prior, hybrid_beta=0.5)
    for name in prior:
        assert np.array_equal(pure.increments[name], hybrid.increments[name])
    # The control differs: the static share moved it.
    assert not np.array_equal(pure.control_increment["theta"], hybrid.control_increment["theta"])


# ---------------------------------------------------------------------------
# The options
# ---------------------------------------------------------------------------

def test_the_options_accept_a_beta_below_one_with_a_table_and_refuse_one_without():
    opts = FilterOptions(hybrid_beta=0.5)
    assert opts.static_covariance == "packaged" and opts.identity()["hybrid_beta"] == 0.5
    with pytest.raises(ValueError, match="names no table"):
        FilterOptions(hybrid_beta=0.5, static_covariance=None)
    with pytest.raises(ValueError, match="static_samples"):
        FilterOptions(static_samples=0)
    with pytest.raises(ValueError, match="hybrid_beta must lie"):
        FilterOptions(hybrid_beta=0.0)
    door = ControlOptions(hybrid_beta=0.75, static_covariance="/tmp/x.npz", static_samples=16)
    laid = door.filter_options(FilterOptions())
    assert laid.hybrid_beta == 0.75 and laid.static_covariance == "/tmp/x.npz" and laid.static_samples == 16
    with pytest.raises(ValueError, match="hybrid_beta"):
        ControlOptions(hybrid_beta=1.5)


# ---------------------------------------------------------------------------
# The door: a table from the smoke run's own checkpoints, a hybrid cycle
# ---------------------------------------------------------------------------

def test_the_static_covariance_door_and_a_hybrid_cycle_at_the_smoke_truncation(spun_up, tmp_path):
    import json

    from woof.globe import da_door
    from woof.globe.da_filter import ENSEMBLE_MANIFEST_NAME
    from woof.globe.da_static import estimate_table

    cfg, checkpoint = spun_up
    files = sorted(checkpoint.parent.glob("arwen_global_step*.npz"))
    assert len(files) >= 3, files
    pairs = [(files[-1], files[-2]), (files[-2], files[-3])]
    receipt = estimate_table(cfg, pairs, tmp_path / "static", version="smoke", config_path=CONFIG)
    table = StaticCovariance.load(receipt["table"])
    assert table.truncation == cfg.truncation and table.nlev == cfg.vertical.nlev
    assert receipt["samples"] == 2 and receipt["charts"] and receipt["sha256"] == table.sha256()
    assert all(p["later"]["sha256"] and p["earlier"]["sha256"] for p in receipt["pairs"])
    with pytest.raises(FileExistsError):
        estimate_table(cfg, pairs, tmp_path / "static", charts=False)

    obs = _synthetic_obs_files(cfg, checkpoint, tmp_path)
    out = tmp_path / "hybrid"
    control = ControlOptions(hybrid_beta=0.5, static_covariance=str(receipt["table"]), static_samples=6)
    da_door.init(cfg, out, filter_name="letkf", members=3, ensemble_truncation=3,
                 analysis_time_utc=START_TEXT, options=OPTIONS, control_options=control)
    door = da_door.cycle(
        cfg, out, stream_specs=["local-tables:paths=" + ",".join(map(str, obs))], cycles=1,
        start_utc=START_TEXT, interval_s=20.0, ensemble=out / ENSEMBLE_MANIFEST_NAME,
        options=OPTIONS, control_options=control,
    )
    assert door["status"] == "pass"
    report = json.loads((out / "assimilation-report-step00000002.json").read_text())
    package = report["ensemble_report"]
    hybrid = package["hybrid"]
    assert hybrid["applied"] is True and hybrid["beta"] == 0.5 and hybrid["static_samples"] == 6
    assert hybrid["table"]["sha256"] == table.sha256()
    assert package["letkf"]["hybrid_beta"] == 0.5 and package["letkf"]["static_samples"] == 6
    assert package["letkf"]["hybrid_solver"] == "batched-solve"
    assert package["options"]["filter"]["hybrid_beta"] == 0.5
    streams = hybrid["observation_space"]
    assert streams and all(v["static_spread_rms"] >= 0.0 for s in streams.values() for v in s.values())
    assert all(np.isfinite(v) for v in hybrid["draw_grid_rms"].values())


# ---------------------------------------------------------------------------
# The packaged table
# ---------------------------------------------------------------------------

def test_the_packaged_table_is_the_lagged_forecast_estimate_of_record_on_the_t255_ladder():
    """``packaged`` resolves to the table shipped under ``woof/data``: the
    T255, 40-level estimate from the 24 h minus 12 h pairs of the
    2026-08-30 to 2026-09-02 GDAS-started forecasts, hash-bound, its
    receipt beside it carrying the same hash, every band correlation
    positive-semidefinite (the sampler's Cholesky factors exist), and the
    checks a hybrid run makes against it (a T127 ensemble on 40 levels is
    accepted, a T383 one refused by name)."""
    path = resolve_static_covariance("packaged")
    assert path.name == "static-covariance-v1.npz" and path.is_file()
    table = load_static_covariance("packaged")
    assert table.truncation == 255 and table.nlev == 40
    assert int(table.receipt["samples"]) >= 11
    assert str(table.receipt["version"]).startswith("lagged-24h-minus-12h")
    for pair in table.receipt["pairs"]:
        assert pair["later"]["time_s"] == 86400.0 and pair["earlier"]["time_s"] == 43200.0
    receipt = json.loads(path.with_name("static-covariance-v1-receipt.json").read_text(encoding="utf-8"))
    assert receipt["sha256"] == table.sha256()
    assert receipt["truncation"] == 255 and receipt["nlev"] == 40 and receipt["samples"] == table.receipt["samples"]
    # The shipped table is the estimate ABOUT the sample mean of the lagged
    # differences: a table estimated about zero carries the model's drift
    # between the two forecast ranges as variance (13,100 K^2 of lid theta
    # on this sample, the global-mean theta at the 1.2 hPa lid 32 K colder
    # in every 24 h forecast than in the 12 h one), and the receipt's drift
    # record says so with the drift it removed.
    drift = receipt["drift"]
    assert drift["removed"] is True and abs(drift["expected_share_if_no_drift"] - 1.0 / receipt["samples"]) < 1.0e-12
    assert drift["theta_global_mean_drift_k_by_level"][0] < -20.0
    assert max(drift["share_of_raw_second_moment_by_level"]["theta"]) > 0.5
    assert table.variance["theta_unbalanced"][0, 0] < 100.0     # the degree-0 lid term, 5 K^2 shipped, 13,139 K^2 about zero
    for name in CONTROL_VARIABLES:
        var = table.variance[name]
        want = (table.truncation + 1,) if name == "log_surface_pressure_unbalanced" else (table.truncation + 1, table.nlev)
        assert var.shape == want
        assert np.all(np.isfinite(var)) and np.all(var >= 0.0) and var.sum() > 0.0
        factors = table.cholesky_factors(name)
        assert factors.shape == table.vertical_correlation[name].shape and np.all(np.isfinite(factors))
    table.check_against(127, 40)
    with pytest.raises(ValueError, match="covers degrees up to T255"):
        table.check_against(383, 40)
    with pytest.raises(ValueError, match="40 levels"):
        table.check_against(127, 49)


def test_the_packaged_tables_two_receipts_carry_the_same_redacted_paths():
    """The table's own metadata member and the JSON receipt beside it are one
    receipt in two encodings, and they are held to one rule.

    THE BREAKAGE THIS PREVENTS, measured 2026-09-10 inside the built wheel:
    the JSON receipt's twenty-three input paths were rewritten to keep their
    file names and lose the root they sat under, and the identical paths
    inside the npz `__metadata__` member were not, because that member is a
    UTF-32 unicode array inside a zip and no text rule in this repository
    could open it. The pairs are asserted EQUAL rather than each scrubbed on
    its own: neither copy can be redacted without the other, and a rewrite
    that touches one is red here until it touches both.
    """

    path = resolve_static_covariance("packaged")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"].item()))
    receipt = json.loads(
        path.with_name("static-covariance-v1-receipt.json").read_text(
            encoding="utf-8"))
    assert metadata["receipt"]["pairs"] == receipt["pairs"]
    assert metadata["sha256"] == receipt["sha256"]
    assert metadata["receipt"]["config"].startswith("<covariance-sample>/")
    text = json.dumps(metadata)
    for token in ("/home/", "Users" + chr(92)):
        assert token not in text, token
    for pair in metadata["receipt"]["pairs"]:
        for side in ("earlier", "later"):
            assert pair[side]["path"].startswith("<covariance-sample>/")
