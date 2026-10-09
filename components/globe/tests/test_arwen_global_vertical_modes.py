"""The vertical-mode semi-implicit scheme (audit 2026-09-01, DN-1/DN-2/DN-5).

The external-only proxy left every internal gravity-wave mode explicit: the
rest-state step ceiling of the coded step at T533 was 104.7 s regardless
of the reference speed, the CFL gate did not bound the step, and at
250 hPa more than half of the mesoscale kinetic energy beyond ~330 km was
divergent.  The vertical-mode scheme treats every mode through the linear
gravity-wave operator of the hybrid hydrostatic equations about an
isothermal reference, derived in the model's own discrete forms
(semi_implicit.py docstring), and these tests hold it to the artifact:

* the operator IS the central-difference linearization of the coded rhs
  at the reference state (B to 3.3e-10, C to 2.5e-15 relative on the
  20-level pressure_blend stack; 2.1e-9 / 4.0e-15 on the 40-level default);
* its modes are real, positive, and over-cover the audit's 286/220 K
  column (325.4 / 191.5 / 120.7 / 81.2 m/s): 356.2 / 242.6 / 149.9 / 97.3
  at T_ref = 320 K on 20 levels;
* apply() is the dense off-centred Crank-Nicolson of the operator, neutral
  at weight 0.5 and damping above, and half_apply() is its square root;
* the linearized coded step (the auditor's t2/t7 instrument, re-implemented
  and cross-checked against the audit's own 104.7 s) has its rest ceiling
  raised at T21 and T533;
* a T21 nonlinear run stays bounded at 4x and 8x the former ceiling;
* a uniform isothermal rest state is a fixed point, and the rest state
  over a mountain (the auditor's t6 form, now through the full step) keeps
  its winds two orders below what the external-only era's split made;
* restart is bit-exact.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import dataclasses
import math

import numpy as np
import pytest

# CPU only: the device switch is set for each test in this module by
# `conftest._cpu_only_marked_tests` and put back afterwards, because a
# module that set it at import time decided it for the whole session.
pytestmark = pytest.mark.cpu_only

from woof.globe.config import load_config  # noqa: E402
from woof.globe.constants import (  # noqa: E402
    KAPPA,
    REFERENCE_PRESSURE_PA,
    SPECTRAL_FIELDS,
)
from woof.globe.dynamics import MoistHybridModel  # noqa: E402
from woof.globe.runner import build_model_and_cold_state  # noqa: E402
from woof.globe.semi_implicit import (  # noqa: E402
    BarotropicSemiImplicit,
    VerticalModeSemiImplicit,
    vertical_structure_operator,
)
from woof.globe.state import (  # noqa: E402
    ArwenGlobalState,
    MoistHybridState,
    SurfaceState,
)
from woof.globe.vertical import HybridCoordinate  # noqa: E402
from woof.globe.spectral.diffusion import ExponentialHyperdiffusion  # noqa: E402
from woof.globe.spectral.transform import SphericalHarmonicTransform  # noqa: E402
from woof.globe.spectral.vector import VorticityDivergenceOperator  # noqa: E402

from test_arwen_global_vertical_numerics import (  # noqa: E402
    _rest_state,
    _zero_grid_tracers,
)

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
AUDIT_COLUMN_MODES_M_S = (325.4, 191.5, 120.7, 81.2)
AUDIT_T533_REST_CEILING_S = 104.7


# ------------------------------------------------------------------ helpers
def _quiet_model(transform, vertical, semi_implicit, surface_geopotential=None, diffusion=None):
    if surface_geopotential is None:
        surface_geopotential = np.zeros(transform.grid.shape)
    # The split-era stepper: these tests measure the symmetric split the
    # IMEX default replaced (tests/test_arwen_global_imex.py measures that).
    return MoistHybridModel(
        transform=transform, vertical=vertical,
        surface_geopotential=surface_geopotential, physics=None,
        rotation_rate_s=0.0, diffusion=diffusion, semi_implicit=semi_implicit,
        integrator="ssprk3",
        mass_fixer=False, water_fixer=False, positivity_repair=False,
        maximum_cfl=1.0e9, sponge_base_pa=0.0,
    )


def _column_state(transform, vertical, temperature_of_normalized, ps0=1.0e5):
    """A horizontally uniform rest state with T = f(normalized ln p)."""
    ps = np.full(transform.grid.shape, ps0)
    p_full = vertical.pressure(ps, transform.backend)["p_full"]
    normalized = np.log(p_full / p_full[0:1]) / np.log(p_full[-1:] / p_full[0:1])
    temperature = temperature_of_normalized(normalized)
    theta = temperature / (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
    zeros = np.zeros((vertical.nlev, *transform.spectral_shape), dtype=np.complex128)
    return MoistHybridState(
        vorticity=zeros.copy(), divergence=zeros.copy(), theta=transform.forward(theta),
        log_surface_pressure=transform.forward(np.log(ps)),
        qv=zeros.copy(),
        **_zero_grid_tracers(transform, vertical),
    )


def _rhs_jacobian(model, state, n, m=0):
    """Central-difference Jacobian of the coded rhs in (D, theta, lnps) at
    harmonic (n, m): the auditor's t2_modes instrument."""
    nlev = model.nlev
    size = 2 * nlev + 1
    jacobian = np.zeros((size, size))

    def pack(r):
        return np.concatenate([
            r.divergence[:, n, m].real, r.theta[:, n, m].real,
            [r.log_surface_pressure[n, m].real],
        ])

    for j in range(size):
        out = []
        for sign in (1.0, -1.0):
            perturbed = state.with_fields([f.copy() for f in state.fields()])
            if j < nlev:
                eps = 1.0e-9
                perturbed.divergence[j, n, m] += sign * eps
            elif j < 2 * nlev:
                eps = 1.0e-3
                perturbed.theta[j - nlev, n, m] += sign * eps
            else:
                eps = 1.0e-6
                perturbed.log_surface_pressure[n, m] += sign * eps
            out.append(pack(model.rhs(perturbed)))
        jacobian[:, j] = (out[0] - out[1]) / (2.0 * eps)
    return jacobian


def _split_blocks(jacobian, nlev, k2):
    """(B, C) with D_t = k^2 B x and x_t = C D from a Jacobian at k^2."""
    return jacobian[:nlev, nlev:] / k2, jacobian[nlev:, :nlev]


def _wavenumber(transform, n):
    return math.sqrt(n * (n + 1.0)) / transform.grid.radius_m


def _linear_operator_matrix(semi, vertical, nlev):
    """The (2 nlev + 1)^2 operator a scheme subtracts, per unit k^2 in the
    divergence rows."""
    size = 2 * nlev + 1
    matrix = np.zeros((size, size))
    w = semi.divergence_weight
    if isinstance(semi, VerticalModeSemiImplicit):
        operator = semi.operator(vertical)
        matrix[:nlev, nlev:] = w * operator.b_matrix
        matrix[nlev:, :nlev] = w * operator.c_matrix
    else:
        matrix[:nlev, -1] = w * semi.external_wave_speed_m_s ** 2
        matrix[-1, :nlev] = -w * vertical.delta_b / np.sum(vertical.delta_b)
    return matrix


def _ssprk3(z):
    identity = np.eye(z.shape[0])
    return identity + z + z @ z / 2.0 + z @ z @ z / 6.0


def _dense_half_map(operator, nlev, k, dt, alpha):
    """H = P + Q L of the scheme (semi_implicit.symmetric_half_maps) as one
    dense (2 nlev + 1)^2 matrix at wavenumber k."""
    size = 2 * nlev + 1
    scalars = operator._half_map_scalars
    p_d, q_d = scalars(k * np.sqrt(operator.eigenvalues), dt, alpha)
    p_x, q_x = scalars(k * np.sqrt(-operator.x_eigenvalues), dt, alpha)
    v, w = operator.eigenvectors, operator.x_eigenvectors
    p = np.zeros((size, size))
    q = np.zeros((size, size))
    p[:nlev, :nlev] = v @ np.diag(p_d) @ np.linalg.inv(v)
    q[:nlev, :nlev] = v @ np.diag(q_d) @ np.linalg.inv(v)
    p[nlev:, nlev:] = w @ np.diag(p_x) @ np.linalg.inv(w)
    q[nlev:, nlev:] = w @ np.diag(q_x) @ np.linalg.inv(w)
    lk = np.zeros((size, size))
    lk[:nlev, nlev:] = k ** 2 * operator.b_matrix
    lk[nlev:, :nlev] = operator.c_matrix
    return p + q @ lk


def _coded_step_amplification(
    jacobian_blocks, linear, nlev, k, dt, alpha, *, wind=0.0,
    diffusion=None, half_map=None,
):
    """The coded step at wavenumber k, linearized: the auditor's
    coded_step ``Diff * CN_alpha(dt L) * R3(dt (J - L))`` for the Lie
    split of the external proxy, or ``Diff * H * R3(dt (J - L)) * H`` with
    ``H`` the scheme's half map (``H^2 = CN_alpha``) for the symmetric
    vertical-mode split."""
    b, c = jacobian_blocks
    size = 2 * nlev + 1
    j = np.zeros((size, size), dtype=complex)
    j[:nlev, nlev:] = k ** 2 * b
    j[nlev:, :nlev] = c
    j += 1j * k * wind * np.eye(size)
    lk = linear.astype(complex).copy()
    lk[:nlev, nlev:] *= k ** 2
    identity = np.eye(size)
    explicit = _ssprk3(dt * (j - lk))
    if half_map is None:
        corrector = np.linalg.solve(identity - alpha * dt * lk, identity + (1.0 - alpha) * dt * lk)
        g = corrector @ explicit
    else:
        h = half_map(k, dt, alpha)
        g = h @ explicit @ h
    if diffusion is not None:
        truncation, tau, n = diffusion
        f = math.exp(-(dt / tau) * (n / truncation) ** 8)
        d = np.ones(size)
        d[:nlev] = f ** 1.5
        d[nlev:2 * nlev] = f
        d[-1] = f ** 0.25
        g = np.diag(d) @ g
    return g


def _spectral_radius(g):
    return float(np.max(np.abs(np.linalg.eigvals(g))))


def _rest_ceiling(
    jacobian_blocks, linear, nlev, transform, truncation, alpha, *,
    tau=None, hi=1.0e6, half_map=None,
):
    """Largest dt with spectral radius <= 1 + 1e-9 at rest over EVERY
    total degree (25 degrees from 1 to the truncation; the auditor's
    instrument read only n = T, where the hyperdiffusion is strongest)."""
    degrees = sorted(set(int(v) for v in np.linspace(1, truncation, 25)))

    def stable(dt):
        for n in degrees:
            k = _wavenumber(transform, n)
            diffusion = None if tau is None else (truncation, tau, n)
            g = _coded_step_amplification(
                jacobian_blocks, linear, nlev, k, dt, alpha, diffusion=diffusion,
                half_map=half_map,
            )
            if _spectral_radius(g) > 1.0 + 1.0e-9:
                return False
        return True

    # The hyperdiffusion factor exp(-dt/tau) makes rho fall again at
    # absurd dt, so the ceiling is the FIRST unstable dt scanning upward
    # (ratio 1.1), refined by bisection between the last stable rung and
    # it; never unstable up to ``hi`` reads as unbounded.
    lo = 1.0
    if not stable(lo):
        return 0.0
    probe = lo
    while True:
        probe *= 1.1
        if probe > hi:
            return float("inf")
        if not stable(probe):
            hi = probe
            break
        lo = probe
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _surface(transform):
    """An all-water planet carrying the synthetic static fields.

    The surface state carries the categories and climatologies Noah is
    driven with; these dynamics tests run a quiet model, so the values are
    the declared synthetic planet and the radiative constants the tests
    were measured with.
    """
    from woof.globe.statics import synthetic_surface_statics

    z = np.zeros(transform.grid.shape)
    soil = np.zeros((4, *z.shape)) + 286.0
    statics = synthetic_surface_statics(z, soil)
    for name in ("land_fraction", "albedo", "emissivity", "roughness_m"):
        statics.pop(name)
    return SurfaceState(
        temperature_k=z + 286.0, water_kg_m2=z + 500.0, land_fraction=z.copy(),
        albedo=z + 0.1, emissivity=z + 0.96, roughness_m=z + 1.0e-4,
        heat_capacity_j_m2_k=z + 4.0e7,
        soil_temperature_k=soil,
        soil_water_fraction=np.zeros((4, *z.shape)) + 0.3,
        accumulated_rain_kg_m2=z.copy(), accumulated_snow_kg_m2=z.copy(),
        accumulated_graupel_kg_m2=z.copy(),
        **statics,
    )


# ------------------------------------------------------------ the operator
@pytest.mark.parametrize(
    "nlev,coordinate,b_bound,c_bound",
    [
        (20, "pressure_blend", 1.0e-7, 1.0e-12),
        (40, "surface_stretched", 1.0e-7, 1.0e-12),
        (4, "pressure_blend", 1.0e-7, 1.0e-12),
    ],
)
def test_operator_is_the_linearization_of_the_coded_rhs(nlev, coordinate, b_bound, c_bound):
    """Measured: B 3.3e-10 / C 2.5e-15 (20-level pressure_blend), 2.1e-9 /
    4.0e-15 (40-level surface_stretched), 1.4e-10 / 1.4e-15 (4-level);
    the bounds sit 2-3 orders above the central-difference truncation."""
    transform = SphericalHarmonicTransform.create(5, backend="numpy", precision="float64")
    vertical = getattr(HybridCoordinate, coordinate)(nlev, 100.0)
    semi = VerticalModeSemiImplicit()
    operator = semi.operator(vertical)
    model = _quiet_model(transform, vertical, BarotropicSemiImplicit(enabled=False))
    state = _column_state(
        transform, vertical, lambda _n: np.full_like(_n, semi.reference_temperature_k),
        ps0=semi.reference_surface_pressure_pa,
    )
    rest = model.rhs(state)
    assert float(np.max(np.abs(rest.divergence))) < 1.0e-18
    k2 = _wavenumber(transform, 5) ** 2
    b_num, c_num = _split_blocks(_rhs_jacobian(model, state, 5), nlev, k2)
    b_err = np.max(np.abs(operator.b_matrix - b_num)) / np.max(np.abs(b_num))
    c_err = np.max(np.abs(operator.c_matrix - c_num)) / np.max(np.abs(c_num))
    assert b_err < b_bound, b_err
    assert c_err < c_bound, c_err
    # The operator is degree-independent per unit k^2 (the Jacobian's
    # k-scaling the audit verified to 1.7e-9).
    k2_3 = _wavenumber(transform, 3) ** 2
    b_3, c_3 = _split_blocks(_rhs_jacobian(model, state, 3), nlev, k2_3)
    assert np.max(np.abs(b_3 - b_num)) / np.max(np.abs(b_num)) < 1.0e-7
    assert np.max(np.abs(c_3 - c_num)) / np.max(np.abs(c_num)) < 1.0e-12


def test_modes_are_real_positive_and_over_cover_the_audited_column():
    """20-level pressure_blend at T_ref 320 K: 356.2 / 242.6 / 149.9 / 97.3 /
    68.5 m/s (equivalent depths 12935 / 6004 / 2292 / 965 m); 40-level
    surface_stretched: 354.9 / 227.4 / 151.5 / 109.7 / 84.8 m/s.  The
    audit's 286/220 K column linearizes to 325.4 / 191.5 / 120.7 / 81.2:
    every reference mode is faster, the stability requirement of the
    split (a reference slower than the atmosphere leaves a residual
    explicit wave)."""
    twenty = vertical_structure_operator(HybridCoordinate.pressure_blend(20, 100.0), 320.0, 1.0e5)
    forty = vertical_structure_operator(HybridCoordinate.surface_stretched(40), 320.0, 1.0e5)
    for operator in (twenty, forty):
        assert operator.eigenvalues.shape == (operator.nlev,)
        assert np.all(operator.eigenvalues > 0.0)
        assert np.all(np.diff(operator.phase_speeds_m_s) < 0.0)
        # The modal basis is well conditioned enough for the Helmholtz
        # inverse to be assembled in it (measured 4.9 and 40.8).
        assert np.linalg.cond(operator.eigenvectors) < 100.0
    np.testing.assert_allclose(
        twenty.phase_speeds_m_s[:5], [356.2, 242.6, 149.9, 97.3, 68.5], atol=0.06
    )
    np.testing.assert_allclose(
        forty.phase_speeds_m_s[:5], [354.9, 227.4, 151.5, 109.7, 84.8], atol=0.06
    )
    for speed, audited in zip(twenty.phase_speeds_m_s, AUDIT_COLUMN_MODES_M_S):
        assert speed > audited
    # The external mode of an isothermal hydrostatic atmosphere is the Lamb
    # wave sqrt(R T / (1 - kappa)) = 358.6 m/s at 320 K; the 20-level
    # discretization carries it at 356.2 (0.7% low), the 40-level at 354.9.
    lamb = math.sqrt(287.05 * 320.0 / (1.0 - KAPPA))
    assert abs(twenty.fastest_mode_m_s - lamb) / lamb < 0.02
    described = VerticalModeSemiImplicit().describe(twenty.vertical)
    assert described["scheme"] == "vertical_modes"
    assert described["phase_speeds_m_s"][0] == twenty.fastest_mode_m_s


def test_apply_is_the_dense_off_centred_crank_nicolson_of_the_operator():
    """apply() solves (I - a dt L) y+ = (I + (1-a) dt L) y* per harmonic
    through the modal Helmholtz reduction; the dense solve is the check.
    At a = 0.5 every eigenvalue of that map has modulus 1 (neutral); at
    0.55 every gravity-wave eigenvalue is damped."""
    transform = SphericalHarmonicTransform.create(7, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(6, 100.0)
    nlev = vertical.nlev
    rng = np.random.default_rng(5)
    zeros = np.zeros((nlev, *transform.spectral_shape), dtype=np.complex128)

    def random_field(shape):
        real = rng.standard_normal(shape)
        imag = rng.standard_normal(shape)
        return transform.project(real + 1j * imag)

    state = MoistHybridState(
        vorticity=zeros.copy(),
        divergence=1.0e-5 * random_field(zeros.shape),
        theta=random_field(zeros.shape),
        log_surface_pressure=1.0e-3 * random_field(transform.spectral_shape),
        qv=zeros.copy(),
        **_zero_grid_tracers(transform, vertical),
    )
    state.divergence[:, 0, 0] = 0.0
    dt = 1800.0
    for alpha in (0.5, 0.55, 1.0):
        semi = VerticalModeSemiImplicit(off_centring_weight=alpha)
        stepped, metrics = semi.apply(state, transform, vertical, dt)
        assert metrics["semi_implicit_max_divergence_increment_s1"] > 0.0
        linear = _linear_operator_matrix(semi, vertical, nlev)
        size = 2 * nlev + 1
        for n in range(transform.truncation + 1):
            lk = linear.copy()
            lk[:nlev, nlev:] *= _wavenumber(transform, n) ** 2
            identity = np.eye(size)
            dense = np.linalg.solve(identity - alpha * dt * lk, identity + (1.0 - alpha) * dt * lk)
            moduli = np.abs(np.linalg.eigvals(dense))
            if n == 0:
                assert np.allclose(moduli, 1.0)
            elif alpha == 0.5:
                assert np.allclose(moduli, 1.0, atol=1.0e-10)
            else:
                # 2 nlev gravity-wave eigenvalues (+-i c_j k) damped; the
                # one-dimensional null space of L - the (theta, ln ps)
                # combination with no pressure-gradient force, B x = 0 -
                # stays neutral.
                assert np.sum(moduli < 1.0 - 1.0e-9) == 2 * nlev
                assert np.sum(np.abs(moduli - 1.0) < 1.0e-9) == 1
            for m in range(n + 1):
                before = np.concatenate([
                    state.divergence[:, n, m], state.theta[:, n, m],
                    [state.log_surface_pressure[n, m]],
                ])
                after = np.concatenate([
                    stepped.divergence[:, n, m], stepped.theta[:, n, m],
                    [stepped.log_surface_pressure[n, m]],
                ])
                expected = before if n == 0 else dense @ before
                np.testing.assert_allclose(after, expected, rtol=1.0e-9, atol=1.0e-14)
    # The scheme never touches vorticity or the tracers.
    stepped, _ = VerticalModeSemiImplicit().apply(state, transform, vertical, dt)
    assert stepped.vorticity is state.vorticity
    assert stepped.qv is state.qv
    # The half map the model runs on either side of the explicit step is
    # the square root of that map: twice is apply() (measured 2e-15), and
    # at weight 0.5 it is itself neutral.
    for alpha in (0.5, 0.55, 1.0):
        semi = VerticalModeSemiImplicit(off_centring_weight=alpha)
        full, _ = semi.apply(state, transform, vertical, dt)
        half, _ = semi.half_apply(state, transform, vertical, dt)
        twice, _ = semi.half_apply(half, transform, vertical, dt)
        for a, b in zip(full.fields()[1:4], twice.fields()[1:4]):
            np.testing.assert_allclose(b, a, rtol=1.0e-12, atol=1.0e-13 * float(np.max(np.abs(a))))
        dense_half = _dense_half_map(semi.operator(vertical), nlev, _wavenumber(transform, 4), dt, alpha)
        moduli = np.abs(np.linalg.eigvals(dense_half))
        if alpha == 0.5:
            assert np.allclose(moduli, 1.0, atol=1.0e-10)
        else:
            assert np.sum(moduli < 1.0 - 1.0e-9) == 2 * nlev


def test_rhs_subtracts_exactly_the_operator_it_integrates():
    """The split is only real if the explicit right-hand side stops
    carrying the operator: rhs(on) - rhs(off) must be -w L y in all three
    variables (a corrector that integrates L while this subtraction is
    missing double-steps the wave, the 1256 hPa defect)."""
    cfg = dataclasses.replace(load_config(CONFIG), semi_implicit_weight=1.0)
    model, state = build_model_and_cold_state(cfg)
    assert isinstance(model.semi_implicit, VerticalModeSemiImplicit)
    transform = model.transform
    on = model.rhs(state.atmosphere)
    semi = model.semi_implicit
    model.semi_implicit = BarotropicSemiImplicit(enabled=False)
    off = model.rhs(state.atmosphere)
    model.semi_implicit = semi
    linear = semi.linear_tendencies(state.atmosphere, transform, model.vertical)
    np.testing.assert_allclose(
        off.divergence - on.divergence, linear.divergence, rtol=1.0e-12, atol=1.0e-20
    )
    np.testing.assert_allclose(off.theta - on.theta, linear.theta, rtol=1.0e-12, atol=1.0e-20)
    np.testing.assert_allclose(
        off.log_surface_pressure - on.log_surface_pressure,
        linear.log_surface_pressure, rtol=1.0e-12, atol=1.0e-20,
    )
    # The subtracted operator is the matrix pair, harmonic by harmonic.
    operator = semi.operator(model.vertical)
    n, m = 2, 1
    k2 = _wavenumber(transform, n) ** 2
    x = np.concatenate([state.atmosphere.theta[:, n, m], [state.atmosphere.log_surface_pressure[n, m]]])
    np.testing.assert_allclose(linear.divergence[:, n, m], k2 * operator.b_matrix @ x, rtol=1.0e-10)
    x_t = operator.c_matrix @ state.atmosphere.divergence[:, n, m]
    scale = float(np.max(np.abs(linear.theta)))
    np.testing.assert_allclose(linear.theta[:, n, m], x_t[:-1], rtol=1.0e-10, atol=1.0e-12 * scale)
    np.testing.assert_allclose(
        linear.log_surface_pressure[n, m], x_t[-1], rtol=1.0e-10,
        atol=1.0e-12 * float(np.max(np.abs(linear.log_surface_pressure))),
    )


# --------------------------------------------------------------- rest states
def test_uniform_isothermal_rest_state_is_a_fixed_point_of_the_step():
    """No harmonic above degree zero carries anything, so the subtracted
    operator, the explicit step and the corrector all return zero: the
    state is unchanged to roundoff over a 3600 s step."""
    transform = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    vertical = HybridCoordinate.surface_stretched(40)
    model = _quiet_model(transform, vertical, VerticalModeSemiImplicit())
    state = _column_state(transform, vertical, lambda n: 220.0 + 66.0 * n)
    bundle = ArwenGlobalState(state, _surface(transform))
    out, metrics = model.step(bundle, 3600.0)
    # Roundoff of the transforms at degree > 0 (measured 1.0e-18 1/s; a
    # perturbed state's increment is ~1e-6).
    assert metrics["semi_implicit_max_divergence_increment_s1"] < 1.0e-15
    for before, after in zip(bundle.atmosphere.fields(), out.atmosphere.fields()):
        assert float(np.max(np.abs(after - before))) < 1.0e-12 * max(1.0, float(np.max(np.abs(before))))


def _balanced_mountain_run(model, state, transform, dt, steps):
    """Max wind (m/s) and max |theta - theta_b| (K) after ``steps`` steps
    from the balanced rest state."""
    theta_b = transform.inverse(state.theta)
    bundle = ArwenGlobalState(dataclasses.replace(state), _surface(transform))
    for _ in range(steps):
        bundle, _ = model.step(bundle, dt)
    u, v = model.vector.wind_from_vordiv(bundle.atmosphere.vorticity, bundle.atmosphere.divergence)
    theta = transform.inverse(bundle.atmosphere.theta) - theta_b
    return float(np.sqrt(u * u + v * v).max()), float(np.max(np.abs(theta)))


def test_rest_over_a_mountain_through_the_full_step():
    """The auditor's t6 form (T42, 20-level pressure_blend, 2 km Gaussian
    mountain, T linear in ln p) run through model.step for 15 minutes at
    dt = 60 s.  The rhs alone leaves 4.27e-7 m/s^2 (DN-3 fix), and with
    the semi-implicit off the hour stays under 4.2e-4 m/s and 7.6e-4 K.

    A corrector applied once AFTER the explicit step is not a fixed point
    at a balanced state (module docstring): measured under the external
    proxy at 450 m/s, 0.73 m/s and 0.19 K at 15 min, 2.7 m/s and 3.2 K at
    60 min; under the vertical-mode operator in that order, 0.39 m/s /
    0.24 K and 1.6 m/s / 1.2 K.  The symmetric split the vertical-mode
    scheme runs (sqrt of Crank-Nicolson on both sides) measures 1.3e-2
    m/s / 6.2e-2 K at 15 min and 8e-3 m/s / 0.25 K at 60 min.  Gate: the
    default's 15-minute wind below a tenth of the proxy's and below
    0.05 m/s; the proxy's own number is asserted so the comparison keeps
    its instrument."""
    transform = SphericalHarmonicTransform.create(42, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    state, phi_s = _rest_state(transform, vertical, 2000.0)
    dt, steps = 60.0, 15
    external = _quiet_model(
        transform, vertical,
        BarotropicSemiImplicit(enabled=True, external_wave_speed_m_s=450.0),
        surface_geopotential=phi_s,
    )
    modes = _quiet_model(transform, vertical, VerticalModeSemiImplicit(), surface_geopotential=phi_s)
    wind_external, theta_external = _balanced_mountain_run(external, state, transform, dt, steps)
    wind_modes, theta_modes = _balanced_mountain_run(modes, state, transform, dt, steps)
    assert 0.5 < wind_external < 1.0, wind_external
    assert 0.1 < theta_external < 0.3, theta_external
    assert wind_modes < 0.1 * wind_external, (wind_modes, wind_external)
    assert wind_modes < 0.05, wind_modes
    assert theta_modes < theta_external, (theta_modes, theta_external)
    assert theta_modes < 0.1, theta_modes


# ------------------------------------------------- the stability instrument
def _audit_column_blocks(nlev=20):
    transform = SphericalHarmonicTransform.create(5, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(nlev, 100.0)
    model = _quiet_model(transform, vertical, BarotropicSemiImplicit(enabled=False))
    state = _column_state(transform, vertical, lambda n: 220.0 + n * (286.0 - 220.0))
    k2 = _wavenumber(transform, 5) ** 2
    return transform, vertical, _split_blocks(_rhs_jacobian(model, state, 5), nlev, k2)


def test_linearized_rest_ceiling_rises_at_t21_and_t533():
    """The auditor's t2_modes/t7_ladder instrument re-implemented: the
    Jacobian of the coded rhs about the 286/220 K column on the 20-level
    pressure_blend stack, the coded step G = Diff * CN(dt L) * R3(dt (J -
    L)) at the truncation wavenumber with the shipped hyperdiffusion
    (order 4, 14400 s at n = T), geometric bisection for the largest dt
    with spectral radius <= 1 + 1e-9.

    Instrument check: the external-only scheme at 450 m/s measures
    102.0 s at T533 against the audit's 104.7 (the audit read n = T only;
    the minimum over degrees sits 3% lower) and 3003 s at T21 (audit
    3007.6).  Vertical-mode scheme (symmetric split, T_ref 320 K), weight
    0.5: 445 s at T533 (4.4x) and 16771 s at T21 (5.6x); the audit's
    4-mode ladder (325/191/121/81 m/s) is entirely inside the implicit
    operator, so what bounds the step is the split's own residual against
    the reference (DN-2), not an explicit wave.  At off-centring 0.55:
    1822 s at T533 (17.9x) and 69361 s at T21 (23.1x) - the
    order-of-magnitude rise arrives with the off-centring Task 2 measures
    before it ships, not with the neutral corrector.  (The Lie order,
    corrector after the step only, measured 423 / 14738 / 1763 / 67913 s:
    the symmetric split costs nothing in stability.)"""
    transform, vertical, blocks = _audit_column_blocks()
    nlev = vertical.nlev
    external = _linear_operator_matrix(
        BarotropicSemiImplicit(enabled=True, external_wave_speed_m_s=450.0), vertical, nlev
    )
    semi = VerticalModeSemiImplicit()
    modes = _linear_operator_matrix(semi, vertical, nlev)
    operator = semi.operator(vertical)

    def half_map(k, dt, alpha):
        return _dense_half_map(operator, nlev, k, dt, alpha)

    tau = 14400.0
    results = {}
    for truncation in (21, 533):
        results[truncation] = {
            "external": _rest_ceiling(blocks, external, nlev, transform, truncation, 0.5, tau=tau),
            "modes_0.5": _rest_ceiling(
                blocks, modes, nlev, transform, truncation, 0.5, tau=tau, half_map=half_map
            ),
            "modes_0.55": _rest_ceiling(
                blocks, modes, nlev, transform, truncation, 0.55, tau=tau, half_map=half_map
            ),
        }
    assert abs(results[533]["external"] - AUDIT_T533_REST_CEILING_S) < 0.05 * AUDIT_T533_REST_CEILING_S, results
    for truncation in (21, 533):
        row = results[truncation]
        assert row["modes_0.5"] > 3.5 * row["external"], (truncation, row)
        assert row["modes_0.55"] > 10.0 * row["external"], (truncation, row)


def test_cfl_gate_carries_no_explicit_wave_at_weight_one():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    speed = model.semi_implicit.explicit_wave_speed_m_s(model.vertical)
    fastest = model.semi_implicit.operator(model.vertical).fastest_mode_m_s
    assert speed == (1.0 - cfg.semi_implicit_weight) * fastest
    model.semi_implicit = VerticalModeSemiImplicit(divergence_weight=1.0)
    assert model.semi_implicit.explicit_wave_speed_m_s(model.vertical) == 0.0
    model.semi_implicit = VerticalModeSemiImplicit(enabled=False)
    assert model.semi_implicit.explicit_wave_speed_m_s(model.vertical) == fastest


# ------------------------------------------------- nonlinear T21 stability
def _t21_bundle(transform, vertical, wind_m_s=5.0):
    """The auditor's v1 state: near rest, U = 5 m/s, a small n = 6
    baroclinic theta perturbation seeding every vertical mode."""
    lat = transform.grid.lat_rad[:, None]
    lon = transform.grid.lon_rad[None, :]
    ps = np.full(transform.grid.shape, 1.0e5)
    p_full = vertical.pressure(ps, transform.backend)["p_full"]
    normalized = np.log(p_full / p_full[0:1]) / np.log(p_full[-1:] / p_full[0:1])
    temperature = (
        220.0 + normalized * (286.0 - 220.0)
        + 0.05 * np.cos(6.0 * lon) * np.cos(lat) ** 2 * np.sin(math.pi * normalized)
    )
    theta = temperature / (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
    u = np.broadcast_to(wind_m_s * np.cos(lat) * np.ones_like(lon), p_full.shape).copy()
    zeta, div = VorticityDivergenceOperator(transform).vordiv_from_wind(u, np.zeros_like(u))
    zeros = np.zeros((vertical.nlev, *transform.spectral_shape), dtype=np.complex128)
    atmosphere = MoistHybridState(
        vorticity=zeta, divergence=div, theta=transform.forward(theta),
        log_surface_pressure=transform.forward(np.log(ps)),
        qv=zeros.copy(),
        **_zero_grid_tracers(transform, vertical),
    )
    return ArwenGlobalState(atmosphere, _surface(transform))


@pytest.mark.parametrize("dt_s,alpha", [(12000.0, 0.5), (24000.0, 0.55)])
def test_t21_nonlinear_run_is_bounded_at_4x_and_8x_the_former_ceiling(dt_s, alpha):
    """The auditor's v1 reproduction (T21, 20 levels, no physics, rotation
    off, shipped hyperdiffusion, U = 5 m/s): under the external-only
    scheme dt = 3000 s completed and dt = 4000 s died at step 17 on the
    explicit internal modes.  200 steps each, measured:

      4x (12000 s), weight 0.5 (default): peak max|D| 7.5e-7 1/s, final
        4.4e-7, max wind 5.58 m/s, temperature 212..286 K;
      8x (24000 s), weight 0.5: dies at step 78 (CFL 0.906, max|D|
        2.2e-5, 11.2 m/s) - past the linearized ceiling of 14738 s the
        instrument above measures at the neutral weight;
      8x (24000 s), weight 0.55: peak 1.0e-7, 5.0 m/s, 220..286 K;
      4x, weight 0.55: peak 2.0e-7, 5.0 m/s.

    So the 8x arm runs at the off-centring Task 2 measures, and the
    bounds below sit an order above the passing arms and below the dying
    one."""
    transform = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    diffusion = ExponentialHyperdiffusion(
        order=4, e_folding_time_s_at_truncation=14400.0, preserve_degree=1
    )
    model = _quiet_model(
        transform, vertical, VerticalModeSemiImplicit(off_centring_weight=alpha),
        diffusion=diffusion,
    )
    model.maximum_cfl = 0.75
    bundle = _t21_bundle(transform, vertical)
    assert model.cfl(bundle.atmosphere, dt_s) < 0.75
    peak = 0.0
    for _ in range(200):
        bundle, _metrics = model.step(bundle, dt_s)
        peak = max(peak, float(np.max(np.abs(bundle.atmosphere.divergence))))
    assert np.isfinite(peak)
    assert peak < 5.0e-6, peak
    diagnostics = model.diagnostics(bundle)
    assert diagnostics["maximum_wind_m_s"] < 8.0
    assert 205.0 < diagnostics["minimum_temperature_k"]
    assert diagnostics["maximum_temperature_k"] < 290.0


# ----------------------------------------------------------------- restart
def test_restart_is_bit_exact_under_the_default_scheme():
    cfg = load_config(CONFIG)
    assert cfg.semi_implicit_scheme == "vertical_modes"
    model, state = build_model_and_cold_state(cfg)
    straight = state
    for _ in range(3):
        straight, _ = model.step(straight, cfg.dt_s)
    model2, state2 = build_model_and_cold_state(cfg)
    resumed = state2
    for _ in range(2):
        resumed, _ = model2.step(resumed, cfg.dt_s)
    # A fresh model resumes from the intermediate state: the operator and
    # its Helmholtz inverses are rebuilt identically from the config.
    model3, _ = build_model_and_cold_state(cfg)
    model3._target_mass_pa = model2._target_mass_pa
    model3._target_total_water_kg_m2 = model2._target_total_water_kg_m2
    resumed, _ = model3.step(resumed, cfg.dt_s)
    for before, after in zip(straight.atmosphere.fields(), resumed.atmosphere.fields()):
        assert np.array_equal(np.asarray(before), np.asarray(after))
