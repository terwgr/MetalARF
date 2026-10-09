"""The IMEX Runge-Kutta integrator (woof.globe.imex): the gates.

Every gate carries its number.  The shipped symmetric split H R3 H is not
a fixed point at a balanced state (8.0e-3 m/s and 0.25 K after 60 min at
rest over a 2 km mountain at dt = 60 s; 0.18 m/s and 1.5 K at dt = 300 s);
the IMEX pair with shared abscissae is one exactly, and these tests hold it
to that AND to the stability, conservation and no-damping bars of the
split it replaces, on the same instruments that measured the split.
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

from woof.globe.checkpoint import read_checkpoint, write_checkpoint  # noqa: E402
from woof.globe.config import (  # noqa: E402
    DEFAULT_DIFFUSION_EFOLD_S, DEFAULT_DIFFUSION_ORDER, TIME_INTEGRATORS, load_config,
)
from woof.globe.constants import (  # noqa: E402
    DRY_AIR_CP,
    EARTH_ROTATION_RATE_S,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
    SPECTRAL_FIELDS,
)
from woof.globe.dynamics import MoistHybridModel  # noqa: E402
from woof.globe.imex import (  # noqa: E402
    IMEX_SSP3,
    IMEX_TABLEAUX,
    IMPLICIT_NEUTRAL_LIMIT,
    ImexTableau,
    imex_step,
    ssp3_tableau,
)
from woof.globe import pins  # noqa: E402
from woof.globe.runner import build_model_and_cold_state  # noqa: E402
from woof.globe.semi_implicit import (  # noqa: E402
    BarotropicSemiImplicit,
    VerticalModeSemiImplicit,
)
from woof.globe.state import ArwenGlobalState, MoistHybridState  # noqa: E402
from woof.globe.vertical import HybridCoordinate  # noqa: E402
from woof.globe.spectral.diffusion import ExponentialHyperdiffusion  # noqa: E402
from woof.globe.spectral.transform import SphericalHarmonicTransform  # noqa: E402
from woof.globe.spectral.vector import VorticityDivergenceOperator  # noqa: E402

from test_arwen_global_vertical_modes import (  # noqa: E402
    _column_state,
    _quiet_model,
    _rhs_jacobian,
    _split_blocks,
    _surface,
    _t21_bundle,
    _wavenumber,
)
from test_arwen_global_vertical_numerics import (  # noqa: E402
    _rest_state,
    _zero_grid_tracers,
)

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
IMEX = IMEX_SSP3.name

#: The shipped split's linearized rest ceilings on the 40-level default
#: stack (T_ref 320 K, four resting columns 286 / 300 / 320 / 340 K,
#: shipped hyperdiffusion, minimum over eight degrees), the instrument of
#: semi_implicit.py's reference-temperature ladder.
SHIPPED_CEILING_S = {533: 395.3, 255: 868.2}
#: The shipped split's largest Doppler growth per step at T533, dt = 60 s,
#: U = 150 m/s on the 20-level pressure_blend stack (dynamics.cfl).
SHIPPED_DOPPLER_RHO = 1.000745
#: The T21 20-level moist 24 h conservation instrument under the shipped
#: split at dt = 600 s (relative drifts; kinetic-energy ratio).
SHIPPED_CONSERVATION = {
    "mass": 1.295e-7, "water": 3.215e-6, "aam": 3.328e-6, "energy": 9.722e-5,
    "ke_ratio": 0.9475,
}
#: T42 20-level jet, dt = 600 s: rotational kinetic energy retained at 24 h.
SHIPPED_ROTATIONAL_RETENTION = 0.8462


def _imex_model(transform, vertical, semi, **kwargs):
    model = _quiet_model(transform, vertical, semi, **kwargs)
    model.integrator = IMEX
    return model


# ------------------------------------------------------------- the tableau
def test_tableau_shares_every_abscissa_and_refuses_the_alternative():
    tableau = IMEX_SSP3
    assert IMEX_TABLEAUX[IMEX] is tableau
    assert IMEX in TIME_INTEGRATORS and "ssprk3" in TIME_INTEGRATORS and "rk4" in TIME_INTEGRATORS
    np.testing.assert_allclose(tableau.abscissae, [0.0, 1.0, 0.5])
    a = np.asarray(tableau.implicit_a)
    at = np.asarray(tableau.explicit_a)
    np.testing.assert_allclose(a.sum(axis=1), at.sum(axis=1), atol=1e-15)
    np.testing.assert_allclose(tableau.explicit_b, tableau.implicit_b)
    # Second order for the implicit part and the coupling: b . c = 1/2.
    assert abs(float(np.dot(tableau.implicit_b, tableau.abscissae)) - 0.5) < 1e-15
    # Two identical stage solves: one cached Helmholtz inverse per step.
    assert a[1, 1] == a[2, 2] > 0.0
    assert float(IMPLICIT_NEUTRAL_LIMIT[IMEX]) > 17.0
    with pytest.raises(ValueError, match="share every abscissa"):
        ImexTableau(
            name="mismatch",
            explicit_a=tableau.explicit_a, explicit_b=tableau.explicit_b,
            implicit_a=((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.0, 0.0, 0.25)),
            implicit_b=tableau.implicit_b,
        )
    with pytest.raises(ValueError, match="sum to one"):
        ImexTableau(
            name="weights", explicit_a=tableau.explicit_a, explicit_b=(0.5, 0.5, 0.5),
            implicit_a=tableau.implicit_a, implicit_b=tableau.implicit_b,
        )


def test_implicit_stability_function_is_fourth_order_neutral_to_its_limit():
    """R(z) = (1 + 2z/5 - z^2/100 + z^3/150) / (1 - 3z/10)^2 for the
    shipped member (a22 = a33 = 3/10, a32 = 1/10): |R(iy)|^2 = 1 - y^4/75
    + O(y^6), so resolved modes of L are neutral to fourth order and
    unresolved ones damped, up to omega dt = 17.33 where the numerator's
    cubic term takes over; dynamics.step refuses a step past that limit
    by name.  (The members with a vanishing cubic term, bounded at
    infinity, measured unstable at every small dt on the 20-level stack:
    +2.4e-4 per step at dt = 60 s for a22 = 1/3, a32 = 1/12.)"""
    tableau = IMEX_SSP3
    np.testing.assert_allclose(tableau.implicit_a, [[0, 0, 0], [0.7, 0.3, 0], [0.1, 0.1, 0.3]])
    for y in (0.05, 0.2, 1.0, 2.0, 10.0, 17.0, 100.0):
        z = 1j * y
        r = tableau.implicit_stability_function(z)
        expected = (1.0 + 0.4 * z - z * z / 100.0 + z ** 3 / 150.0) / (1.0 - 0.3 * z) ** 2
        assert abs(r - expected) < 1e-12 * max(1.0, abs(expected))
        if y <= 17.0:
            assert abs(r) <= 1.0 + 1e-14, (y, abs(r))
    assert abs(abs(tableau.implicit_stability_function(18j)) - 1.0598) < 1e-3
    assert abs(tableau.implicit_neutral_limit() - 17.33) < 0.02
    assert abs(IMPLICIT_NEUTRAL_LIMIT[IMEX] - tableau.implicit_neutral_limit()) < 1e-12
    y = 0.2
    assert abs(abs(tableau.implicit_stability_function(1j * y)) ** 2 - (1.0 - y ** 4 / 75.0)) < 3e-7
    member = ssp3_tableau(0.3, 0.1, 0.3, name="member")
    np.testing.assert_allclose(member.implicit_a, tableau.implicit_a)
    assert tableau.describe()["implicit_neutral_limit"] == tableau.implicit_neutral_limit()


def test_step_refuses_beyond_the_implicit_neutral_limit():
    """c_max k_T dt past 17.33 is refused by name; under the split-era
    steppers the number is zero and nothing is refused (their ceiling is
    the linearized rest instrument's, not this one)."""
    transform = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    model = _imex_model(transform, vertical, VerticalModeSemiImplicit())
    fastest = model.semi_implicit.operator(vertical).fastest_mode_m_s
    k_top = math.sqrt(21 * 22.0) / transform.grid.radius_m
    limit_dt = IMPLICIT_NEUTRAL_LIMIT[IMEX] / (fastest * k_top)
    assert 14000.0 < limit_dt < 15000.0, limit_dt
    assert abs(model.implicit_wave_number(limit_dt) - IMPLICIT_NEUTRAL_LIMIT[IMEX]) < 1e-9
    bundle = _t21_bundle(transform, vertical)
    with pytest.raises(ValueError, match="amplifies the fastest gravity-wave mode"):
        model.step(bundle, 1.01 * limit_dt)
    model.integrator = "ssprk3"
    assert model.implicit_wave_number(1.01 * limit_dt) == 0.0
    model.step(bundle, 1.01 * limit_dt)


def test_stage_solve_is_the_shifted_inverse_of_the_subtracted_operator():
    """solve_shifted returns y with (I - tau w L) y = r, checked against the
    dense (2 nlev + 1)^2 solve per harmonic for both schemes."""
    transform = SphericalHarmonicTransform.create(7, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(6, 100.0)
    nlev = vertical.nlev
    rng = np.random.default_rng(11)
    zeros = np.zeros((nlev, *transform.spectral_shape), dtype=np.complex128)

    def random_field(shape):
        return transform.project(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    state = MoistHybridState(
        vorticity=zeros.copy(), divergence=1.0e-5 * random_field(zeros.shape),
        theta=random_field(zeros.shape),
        log_surface_pressure=1.0e-3 * random_field(transform.spectral_shape),
        qv=zeros.copy(),
        **_zero_grid_tracers(transform, vertical),
    )
    state.divergence[:, 0, 0] = 0.0
    tau = 400.0
    for semi in (
        VerticalModeSemiImplicit(), VerticalModeSemiImplicit(divergence_weight=0.7),
        BarotropicSemiImplicit(enabled=True, external_wave_speed_m_s=450.0),
    ):
        solved = semi.solve_shifted(state, transform, vertical, tau)
        linear = semi.linear_tendencies(solved, transform, vertical)
        # (I - tau w L) y == r on the three operator rows, harmonic by harmonic.
        np.testing.assert_allclose(
            solved.divergence - tau * linear.divergence, state.divergence,
            rtol=1e-10, atol=1e-12 * float(np.max(np.abs(state.divergence))),
        )
        if linear.theta is not None:
            np.testing.assert_allclose(
                solved.theta - tau * linear.theta, state.theta,
                rtol=1e-10, atol=1e-12 * float(np.max(np.abs(state.theta))),
            )
        else:
            assert solved.theta is state.theta
        np.testing.assert_allclose(
            solved.log_surface_pressure - tau * linear.log_surface_pressure,
            state.log_surface_pressure, rtol=1e-10,
            atol=1e-12 * float(np.max(np.abs(state.log_surface_pressure))),
        )
        assert solved.vorticity is state.vorticity and solved.qv is state.qv
    inactive = VerticalModeSemiImplicit(enabled=False).solve_shifted(state, transform, vertical, tau)
    assert inactive is state


# ------------------------------------------------------------- gate (a)
def test_gate_a_uniform_isothermal_rest_is_a_fixed_point():
    """(a) Uniform rest on the 40-level default at T21, one 3600 s step:
    increment below 1e-15 (measured 3.0e-18 1/s; fields to 6.4e-17)."""
    transform = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    vertical = HybridCoordinate.surface_stretched(40)
    for semi in (VerticalModeSemiImplicit(), BarotropicSemiImplicit(enabled=True, external_wave_speed_m_s=450.0)):
        model = _imex_model(transform, vertical, semi)
        state = _column_state(transform, vertical, lambda n: 220.0 + 66.0 * n)
        bundle = ArwenGlobalState(state, _surface(transform))
        out, metrics = model.step(bundle, 3600.0)
        assert metrics["semi_implicit_max_divergence_increment_s1"] < 1.0e-15
        for before, after in zip(bundle.atmosphere.fields(), out.atmosphere.fields()):
            assert float(np.max(np.abs(after - before))) < 1.0e-12 * max(1.0, float(np.max(np.abs(before))))


# ------------------------------------------------------------- gate (b)
def _mountain_drift(model, state, phi_s, transform, dt, steps):
    theta_b = transform.inverse(state.theta)
    bundle = ArwenGlobalState(dataclasses.replace(state), _surface(transform))
    for _ in range(steps):
        bundle, _ = model.step(bundle, dt)
    u, v = model.vector.wind_from_vordiv(bundle.atmosphere.vorticity, bundle.atmosphere.divergence)
    return float(np.sqrt(u * u + v * v).max()), float(np.max(np.abs(transform.inverse(bundle.atmosphere.theta) - theta_b)))


@pytest.mark.parametrize("dt,steps", [(60.0, 60), (300.0, 12)])
def test_gate_b_rest_over_a_mountain_for_an_hour(dt, steps):
    """(b) The auditor's t6 form (T42, 20-level pressure_blend, 2 km
    Gaussian mountain, T linear in ln p) through the full step for 60 min.

    The state is balanced to the rhs residual 4.27e-7 m/s^2 (DN-3), and
    that residual's own response is the floor any consistent scheme
    reproduces: the fully explicit step measures 4.194e-4 m/s / 5.51e-4 K
    at dt = 60 s and 4.197e-4 / 5.50e-4 at dt = 300 s.  The IMEX step
    measures 4.194e-4 / 5.51e-4 and 4.196e-4 / 5.54e-4: the scheme's own
    contribution is the difference, 1e-7 m/s and 4e-6 K, where the shipped
    split adds 8.0e-3 m/s / 0.25 K (dt 60) and 0.18 m/s / 1.5 K (dt 300).
    Gate: scheme-attributable drift below 1e-4 m/s and 1e-3 K at both
    steps, absolute theta below 1e-3 K, absolute wind within 1% of the
    explicit floor."""
    transform = SphericalHarmonicTransform.create(42, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    state, phi_s = _rest_state(transform, vertical, 2000.0)
    explicit = _quiet_model(transform, vertical, BarotropicSemiImplicit(enabled=False), surface_geopotential=phi_s)
    imex = _imex_model(transform, vertical, VerticalModeSemiImplicit(), surface_geopotential=phi_s)
    wind_explicit, theta_explicit = _mountain_drift(explicit, state, phi_s, transform, dt, steps)
    wind_imex, theta_imex = _mountain_drift(imex, state, phi_s, transform, dt, steps)
    assert abs(wind_imex - wind_explicit) < 1.0e-4, (wind_imex, wind_explicit)
    assert abs(theta_imex - theta_explicit) < 1.0e-3, (theta_imex, theta_explicit)
    assert theta_imex < 1.0e-3, theta_imex
    assert wind_imex < 1.01 * wind_explicit + 1.0e-6, (wind_imex, wind_explicit)
    if dt == 300.0:
        # The instrument still sees the split's first-order residual.
        shipped = _quiet_model(transform, vertical, VerticalModeSemiImplicit(), surface_geopotential=phi_s)
        wind_shipped, theta_shipped = _mountain_drift(shipped, state, phi_s, transform, dt, steps)
        assert wind_shipped > 0.1 and theta_shipped > 1.0, (wind_shipped, theta_shipped)


# ------------------------------------------------------------- gate (c)
def _imex_amplification(blocks, operator_matrix, nlev, k, dt, tableau, *, wind=0.0, diffusion=None):
    """The coded IMEX step at wavenumber k, linearized: stage recursion of
    the module docstring on E = J - L (Doppler i k U I) and L."""
    b, c = blocks
    size = 2 * nlev + 1
    j = np.zeros((size, size), dtype=complex)
    j[:nlev, nlev:] = k ** 2 * b
    j[nlev:, :nlev] = c
    j += 1j * k * wind * np.eye(size)
    lk = operator_matrix.astype(complex).copy()
    lk[:nlev, nlev:] *= k ** 2
    e = j - lk
    at = np.asarray(tableau.explicit_a)
    bt = np.asarray(tableau.explicit_b)
    a = np.asarray(tableau.implicit_a)
    bi = np.asarray(tableau.implicit_b)
    identity = np.eye(size, dtype=complex)
    stages = [identity]
    for i in range(1, tableau.stages):
        acc = identity.copy()
        for m in range(i):
            if at[i, m]:
                acc = acc + dt * at[i, m] * (e @ stages[m])
            if a[i, m]:
                acc = acc + dt * a[i, m] * (lk @ stages[m])
        if a[i, i]:
            acc = np.linalg.solve(identity - dt * a[i, i] * lk, acc)
        stages.append(acc)
    g = identity.copy()
    for i in range(tableau.stages):
        g = g + dt * bt[i] * (e @ stages[i]) + dt * bi[i] * (lk @ stages[i])
    if diffusion is not None:
        # (truncation, tau, n) reads order 4 (the ladder's ruler); a fourth
        # entry names the order (the shipped default is order 8, config.py).
        truncation, tau, n = diffusion[:3]
        order = diffusion[3] if len(diffusion) > 3 else 4
        f = math.exp(-(dt / tau) * (n * (n + 1.0) / (truncation * (truncation + 1.0))) ** order)
        d = np.ones(size)
        d[:nlev] = f ** 1.5
        d[nlev:2 * nlev] = f
        d[-1] = f ** 0.25
        g = np.diag(d) @ g
    return g


def _operator_matrix(semi, vertical):
    nlev = vertical.nlev
    matrix = np.zeros((2 * nlev + 1, 2 * nlev + 1))
    operator = semi.operator(vertical)
    matrix[:nlev, nlev:] = operator.b_matrix
    matrix[nlev:, :nlev] = operator.c_matrix
    return matrix


def _column_blocks(transform, vertical, surface_k):
    model = _quiet_model(transform, vertical, BarotropicSemiImplicit(enabled=False))
    state = _column_state(transform, vertical, lambda n: 220.0 + n * (surface_k - 220.0))
    k2 = _wavenumber(transform, 5) ** 2
    return _split_blocks(_rhs_jacobian(model, state, 5), vertical.nlev, k2)


def _ceiling(stable, lo=5.0, hi=2.0e4):
    if not stable(lo):
        return 0.0
    probe = lo
    while stable(probe):
        lo = probe
        probe *= 1.25
        if probe > hi:
            return float("inf")
    hi = probe
    for _ in range(30):
        mid = math.sqrt(lo * hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return lo


def test_gate_c_linearized_rest_ceiling_on_the_default_stack():
    """(c) The reference-temperature ladder's instrument (semi_implicit.py):
    whole-rhs Jacobian about resting 286 / 300 / 320 / 340 K columns of the
    40-level surface_stretched default, T_ref 320 K, shipped hyperdiffusion
    (order 4, 14400 s), spectral radius <= 1 + 1e-9 over eight degrees,
    minimum over the columns.  Shipped split: 395.3 s at T533, 868.2 s at
    T255.  IMEX: 582.9 s and 1247.8 s (1.47x and 1.44x; the 340 K column
    binds at T533, 286 K at T255); 452.9 s on the 20-level pressure_blend
    stack at T533 against the shipped 445.3.  Gate: within 10% of the
    shipped ceilings, i.e. at least 0.9x."""
    transform = SphericalHarmonicTransform.create(5, backend="numpy", precision="float64")
    vertical = HybridCoordinate.surface_stretched(40)
    nlev = vertical.nlev
    semi = VerticalModeSemiImplicit()
    operator_matrix = _operator_matrix(semi, vertical)
    blocks = {ts: _column_blocks(transform, vertical, ts) for ts in (286.0, 300.0, 320.0, 340.0)}
    tau = 14400.0
    measured = {}
    for truncation in (533, 255):
        degrees = sorted(set([truncation] + [int(v) for v in np.linspace(max(2, truncation // 8), truncation, 8)]))

        def ceiling_of(column):
            def stable(dt):
                for n in degrees:
                    g = _imex_amplification(
                        blocks[column], operator_matrix, nlev, _wavenumber(transform, n), dt,
                        IMEX_SSP3, diffusion=(truncation, tau, n),
                    )
                    if not np.all(np.isfinite(g)) or np.max(np.abs(np.linalg.eigvals(g))) > 1.0 + 1e-9:
                        return False
                return True
            return _ceiling(stable)

        measured[truncation] = min(ceiling_of(column) for column in blocks)
        assert measured[truncation] >= 0.9 * SHIPPED_CEILING_S[truncation], measured
    # The instrument reads the shipped split's own number back (the
    # reference-temperature ladder, 320 K row), so the comparison keeps
    # its ruler.
    from test_arwen_global_vertical_modes import _coded_step_amplification, _dense_half_map

    operator = semi.operator(vertical)
    degrees = sorted(set([533] + [int(v) for v in np.linspace(66, 533, 8)]))

    def shipped_stable(dt):
        for n in degrees:
            g = _coded_step_amplification(
                blocks[286.0], operator_matrix, nlev, _wavenumber(transform, n), dt, 0.5,
                diffusion=(533, tau, n), half_map=lambda k, dt, alpha: _dense_half_map(operator, nlev, k, dt, alpha),
            )
            if np.max(np.abs(np.linalg.eigvals(g))) > 1.0 + 1e-9:
                return False
        return True

    shipped = _ceiling(shipped_stable)
    assert abs(shipped - SHIPPED_CEILING_S[533]) < 0.02 * SHIPPED_CEILING_S[533], shipped
    assert measured[533] > shipped


def test_gate_c_rest_ceiling_holds_at_the_shipped_diffusion():
    """The reference-temperature ladder's ruler was cut with the order 4,
    14400 s hyperdiffusion; the shipped default is order 8 at 2160 s
    (config.DEFAULT_DIFFUSION_ORDER, 2026-09-04), which damps the last
    tenth of the degrees harder and everything below n = 0.91 T less.
    The same instrument (four resting columns, T_ref 320 K, eight degrees
    plus the truncation, minimum over the columns) reads the IMEX rest
    ceiling at the shipped diffusion within 10% of the order-4 reading
    at both truncations, so the stability record of gate (c) carries to
    the shipped shape."""
    transform = SphericalHarmonicTransform.create(5, backend="numpy", precision="float64")
    vertical = HybridCoordinate.surface_stretched(40)
    nlev = vertical.nlev
    semi = VerticalModeSemiImplicit()
    operator_matrix = _operator_matrix(semi, vertical)
    blocks = {ts: _column_blocks(transform, vertical, ts) for ts in (286.0, 300.0, 320.0, 340.0)}
    shapes = {"order4_14400": (14400.0, 4), "shipped": (DEFAULT_DIFFUSION_EFOLD_S, DEFAULT_DIFFUSION_ORDER)}
    measured = {}
    for truncation in (533, 255):
        degrees = sorted(set([truncation] + [int(v) for v in np.linspace(max(2, truncation // 8), truncation, 8)]))
        for label, (tau, order) in shapes.items():
            def ceiling_of(column):
                def stable(dt):
                    for n in degrees:
                        g = _imex_amplification(
                            blocks[column], operator_matrix, nlev, _wavenumber(transform, n), dt,
                            IMEX_SSP3, diffusion=(truncation, tau, n, order),
                        )
                        if not np.all(np.isfinite(g)) or np.max(np.abs(np.linalg.eigvals(g))) > 1.0 + 1e-9:
                            return False
                    return True
                return _ceiling(stable)
            measured[(truncation, label)] = min(ceiling_of(column) for column in blocks)
        assert measured[(truncation, "shipped")] >= 0.9 * measured[(truncation, "order4_14400")], measured
        assert measured[(truncation, "order4_14400")] >= 0.9 * SHIPPED_CEILING_S[truncation], measured


# ------------------------------------------------------------- gate (d)
def test_gate_d_doppler_growth_at_the_cfl_gate():
    """(d) T533, 20-level pressure_blend, 286/220 K column, dt = 60 s,
    U = 150 m/s (the advective CFL gate sits at 149.3): largest spectral
    radius over 28 degrees.  Shipped split 1.000745 (dynamics.cfl); IMEX
    1.000000.  Gate: rho <= 1 + 1e-3."""
    transform = SphericalHarmonicTransform.create(5, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    nlev = vertical.nlev
    operator_matrix = _operator_matrix(VerticalModeSemiImplicit(), vertical)
    blocks = _column_blocks(transform, vertical, 286.0)
    degrees = sorted(set([int(v) for v in np.linspace(1, 533, 25)] + [345, 400, 450, 500]))
    worst = 0.0
    for n in degrees:
        g = _imex_amplification(
            blocks, operator_matrix, nlev, _wavenumber(transform, n), 60.0, IMEX_SSP3,
            wind=150.0, diffusion=(533, 14400.0, n),
        )
        worst = max(worst, float(np.max(np.abs(np.linalg.eigvals(g)))))
    assert worst <= 1.0 + 1.0e-3, worst
    assert worst < SHIPPED_DOPPLER_RHO


# ------------------------------------------------------------- gate (e)
def test_gate_e_t21_nonlinear_run_is_bounded_at_4x_the_former_ceiling():
    """(e) The auditor's v1 reproduction at dt = 12000 s (4x the external
    era's 3000 s ceiling), 200 steps, shipped hyperdiffusion: bounded with
    the bounds of the shipped split's own test."""
    transform = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    diffusion = ExponentialHyperdiffusion(order=4, e_folding_time_s_at_truncation=14400.0, preserve_degree=1)
    model = _imex_model(transform, vertical, VerticalModeSemiImplicit(), diffusion=diffusion)
    model.maximum_cfl = 0.75
    bundle = _t21_bundle(transform, vertical)
    peak = 0.0
    for _ in range(200):
        bundle, _metrics = model.step(bundle, 12000.0)
        peak = max(peak, float(np.max(np.abs(bundle.atmosphere.divergence))))
    assert np.isfinite(peak)
    assert peak < 5.0e-6, peak
    diagnostics = model.diagnostics(bundle)
    assert diagnostics["maximum_wind_m_s"] < 8.0
    assert 205.0 < diagnostics["minimum_temperature_k"]
    assert diagnostics["maximum_temperature_k"] < 290.0


# ------------------------------------------------------------- gate (f)
def test_gate_f_restart_is_bit_exact(tmp_path):
    """(f) Three straight steps equal two steps, a checkpoint round trip
    under the IMEX pin, and one more step on a fresh model."""
    cfg = dataclasses.replace(load_config(CONFIG), integrator=IMEX)
    model, state = build_model_and_cold_state(cfg)
    assert model.integrator == IMEX
    straight = state
    for _ in range(3):
        straight, _ = model.step(straight, cfg.dt_s)
    model2, state2 = build_model_and_cold_state(cfg)
    resumed = state2
    for _ in range(2):
        resumed, _ = model2.step(resumed, cfg.dt_s)
    path = write_checkpoint(
        tmp_path / "mid.npz", resumed, config_hash=cfg.config_hash, to_numpy=np.asarray,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
    )
    metadata, arrays = read_checkpoint(path, expected_config_hash=cfg.config_hash,
                                       semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator)
    assert metadata["pins_hash"] == pins.pins_hash("vertical_modes", IMEX)
    with pytest.raises(ValueError, match="arithmetic pins mismatch"):
        read_checkpoint(path, semi_implicit_scheme=cfg.semi_implicit_scheme, integrator="ssprk3")
    from woof.globe.checkpoint import state_from_checkpoint

    reloaded = state_from_checkpoint(metadata, arrays, model.transform.backend)
    model3, _ = build_model_and_cold_state(cfg)
    model3._target_mass_pa = model2._target_mass_pa
    model3._target_total_water_kg_m2 = model2._target_total_water_kg_m2
    resumed, _ = model3.step(reloaded, cfg.dt_s)
    for before, after in zip(straight.atmosphere.fields(), resumed.atmosphere.fields()):
        assert np.array_equal(np.asarray(before), np.asarray(after))


# ------------------------------------------------------------- gate (g)
def _moist_t21_case(transform, vertical):
    """The conservation instrument's state: ps and theta waves, a moist
    column, a sheared zonal jet and a meridional seed, rotation on."""
    lat = transform.grid.lat_rad[:, None]
    lon = transform.grid.lon_rad[None, :]
    vec = VorticityDivergenceOperator(transform)
    ps = 1.0e5 * (1.0 + 0.01 * np.cos(lat) ** 2 * np.cos(3 * lon))
    pf = vertical.pressure(ps, transform.backend)["p_full"]
    nrm = np.log(pf / pf[0:1]) / np.log(pf[-1:] / pf[0:1])
    temp = 220.0 + nrm * 66.0 + 2.0 * np.cos(4 * lon) * np.cos(lat) ** 2 * np.sin(math.pi * nrm)
    theta = temp / (pf / REFERENCE_PRESSURE_PA) ** KAPPA
    qv = 0.006 * nrm ** 3 * np.cos(lat) ** 2 + 1e-6
    u = 15.0 * np.cos(lat) * np.ones_like(lon) * np.ones(pf.shape) * (0.3 + 0.7 * nrm)
    v = 2.0 * np.sin(4 * lon) * np.cos(lat) * np.ones(pf.shape)
    zeta, div = vec.vordiv_from_wind(u, v)
    zeros = np.zeros((vertical.nlev, *transform.spectral_shape), dtype=np.complex128)
    # Vapor is spectral; the condensate and moments are grid tracers.
    fields = {"qv": transform.forward(qv), **_zero_grid_tracers(transform, vertical)}
    atmosphere = MoistHybridState(
        vorticity=zeta, divergence=div, theta=transform.forward(theta),
        log_surface_pressure=transform.forward(np.log(ps)), **fields,
    )
    return ArwenGlobalState(atmosphere, _surface(transform))


def _budgets(model, transform, bundle):
    g = model.grid_state(bundle.atmosphere)
    lat = transform.grid.lat_rad[:, None]
    lon = transform.grid.lon_rad[None, :]
    a = transform.grid.radius_m
    gm = transform.grid.global_mean
    dp, u, v, t, qv = g["dp"], g["u"], g["v"], g["temperature"], g["qv"]
    cosl = np.cos(lat) * np.ones_like(lon)
    return np.array([
        gm(g["ps"]),
        gm(np.sum(qv * dp, axis=0) / GRAVITY_M_S2),
        gm(np.sum((u + EARTH_ROTATION_RATE_S * a * cosl) * a * cosl * dp, axis=0) / GRAVITY_M_S2),
        gm(np.sum((DRY_AIR_CP * t + 0.5 * (u ** 2 + v ** 2)) * dp, axis=0) / GRAVITY_M_S2),
        gm(np.sum(0.5 * (u ** 2 + v ** 2) * dp, axis=0) / GRAVITY_M_S2),
    ])


def test_gate_g_conservation_on_the_t21_moist_day():
    """(g) T21 20-level moist, rotation on, no physics, no fixers, shipped
    hyperdiffusion, dt = 600 s, 24 h: relative drifts of mass, water,
    axial angular momentum and total energy, and the kinetic-energy ratio,
    not worse than the shipped split's +1.3e-7 / -3.2e-6 / -3.3e-6 /
    +9.7e-5 and 0.9475 (semi_implicit.py's conservation instrument)."""
    transform = SphericalHarmonicTransform.create(21, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    diffusion = ExponentialHyperdiffusion(order=4, e_folding_time_s_at_truncation=14400.0, preserve_degree=1)
    model = MoistHybridModel(
        transform=transform, vertical=vertical, surface_geopotential=np.zeros(transform.grid.shape),
        physics=None, rotation_rate_s=EARTH_ROTATION_RATE_S, diffusion=diffusion,
        semi_implicit=VerticalModeSemiImplicit(), integrator=IMEX, mass_fixer=False,
        water_fixer=False, positivity_repair=False, sponge_base_pa=0.0, maximum_cfl=1e9,
    )
    bundle = _moist_t21_case(transform, vertical)
    before = _budgets(model, transform, bundle)
    for _ in range(144):
        bundle, _ = model.step(bundle, 600.0)
    after = _budgets(model, transform, bundle)
    relative = (after - before) / before
    ke_ratio = after[4] / before[4]
    assert abs(relative[0]) <= SHIPPED_CONSERVATION["mass"] * 1.05, relative
    assert abs(relative[1]) <= SHIPPED_CONSERVATION["water"] * 1.05, relative
    assert abs(relative[2]) <= SHIPPED_CONSERVATION["aam"] * 1.05, relative
    assert abs(relative[3]) <= SHIPPED_CONSERVATION["energy"] * 1.05, relative
    assert ke_ratio >= SHIPPED_CONSERVATION["ke_ratio"] - 5e-4, ke_ratio
    diagnostics = model.diagnostics(bundle)
    assert 195.0 < diagnostics["minimum_temperature_k"] and diagnostics["maximum_temperature_k"] < 290.0


# ------------------------------------------------------------- gate (h)
def _jet_t42_case(transform, vertical):
    lat = transform.grid.lat_rad[:, None]
    lon = transform.grid.lon_rad[None, :]
    a = transform.grid.radius_m
    ps = np.full(transform.grid.shape, 1.0e5)
    pf = vertical.pressure(ps, transform.backend)["p_full"]
    nrm = np.log(pf / pf[0:1]) / np.log(pf[-1:] / pf[0:1])
    temp = 220.0 + nrm * 66.0 - 12.0 * (np.sin(lat) ** 2 - 1 / 3) * nrm
    theta = temp / (pf / REFERENCE_PRESSURE_PA) ** KAPPA
    shear = 0.3 + 0.7 * nrm
    psi = (-a * 20.0 * np.sin(lat) + a * 6.0 * np.cos(lat) ** 4 * np.sin(lat) * np.cos(4 * lon)) * np.ones(pf.shape) * shear
    chi = a * 0.5 * np.cos(lat) ** 2 * np.cos(3 * lon) * np.ones(pf.shape) * shear
    zeta = transform.laplacian(transform.forward(psi))
    div = transform.laplacian(transform.forward(chi))
    zeros = np.zeros((vertical.nlev, *transform.spectral_shape), dtype=np.complex128)
    atmosphere = MoistHybridState(
        vorticity=transform.project(zeta), divergence=transform.project(div), theta=transform.forward(theta),
        log_surface_pressure=transform.forward(np.log(ps)), qv=zeros.copy(),
        **_zero_grid_tracers(transform, vertical),
    )
    return ArwenGlobalState(atmosphere, _surface(transform))


def _rotational_ke(transform, vertical, atmosphere):
    truncation = transform.truncation
    a = transform.grid.radius_m
    n = np.arange(truncation + 1, dtype=float)
    inv_k2 = np.zeros(truncation + 1)
    inv_k2[1:] = a ** 2 / (n[1:] * (n[1:] + 1))
    wm = np.ones(truncation + 1)
    wm[1:] = 2.0
    dp_ref = np.diff(vertical.a_half_pa + vertical.b_half * 1.0e5)
    wlev = dp_ref / dp_ref.sum()
    rot = 0.5 * np.sum(np.abs(atmosphere.vorticity) ** 2 * wm[None, None, :] * inv_k2[None, :, None], axis=(1, 2)) / (4 * math.pi)
    return float(np.sum(rot * wlev))


def test_gate_h_rotational_kinetic_energy_is_not_damped():
    """(h) T42 20-level jet with a Rossby-Haurwitz-like wave and a small
    divergent seed, rotation on, shipped hyperdiffusion, dt = 600 s, 24 h:
    rotational kinetic energy retained.  Fully explicit 0.8455, shipped
    split 0.8462, off-centred split (alpha 0.55) 0.8319: a scheme that
    stabilizes by damping reads low here.  Gate: not below 0.846 - 5e-4."""
    transform = SphericalHarmonicTransform.create(42, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    diffusion = ExponentialHyperdiffusion(order=4, e_folding_time_s_at_truncation=14400.0, preserve_degree=1)
    model = MoistHybridModel(
        transform=transform, vertical=vertical, surface_geopotential=np.zeros(transform.grid.shape),
        physics=None, rotation_rate_s=EARTH_ROTATION_RATE_S, diffusion=diffusion,
        semi_implicit=VerticalModeSemiImplicit(), integrator=IMEX, mass_fixer=False,
        water_fixer=False, positivity_repair=False, sponge_base_pa=0.0, maximum_cfl=1e9,
    )
    bundle = _jet_t42_case(transform, vertical)
    initial = _rotational_ke(transform, vertical, bundle.atmosphere)
    for _ in range(144):
        bundle, _ = model.step(bundle, 600.0)
    retained = _rotational_ke(transform, vertical, bundle.atmosphere) / initial
    assert retained >= SHIPPED_ROTATIONAL_RETENTION - 5.0e-4, retained
    assert retained < 1.0


# --------------------------------------------------- doors, pins, identity
def test_config_door_and_pins(tmp_path):
    cfg = load_config(CONFIG)
    assert cfg.integrator == "ssprk3"  # the smoke fixture names the split stepper
    # the quickstart and the bare config of record run the shipped default
    # core, the semi-Lagrangian one (2026-09-06); the Eulerian config of
    # record names IMEX by name at its rule step
    assert load_config(str(_shipped_configs() / "arwen_global_t255_quickstart.toml")).integrator == "sl_si"
    assert load_config(str(_shipped_configs() / "arwen_global_gdas_t255_native_24h.toml")).integrator == "sl_si"
    assert load_config(str(_shipped_configs() / "arwen_global_gdas_t255_native_imex_24h.toml")).integrator == IMEX
    imex_cfg = dataclasses.replace(cfg, integrator=IMEX)
    assert imex_cfg.config_hash != cfg.config_hash
    model, _ = build_model_and_cold_state(imex_cfg)
    assert model.integrator == IMEX
    # Every arithmetic pins apart; the split-era pins are unchanged and
    # the default pin is the IMEX one.
    assert pins.pins_hash("vertical_modes", "ssprk3") == pins.pins_hash("vertical_modes", "rk4")
    assert pins.pins_hash("vertical_modes", IMEX) == pins.pins_hash("vertical_modes") == pins.PINS_HASH
    assert pins.pins_hash("vertical_modes", IMEX) != pins.pins_hash("vertical_modes", "ssprk3")
    assert pins.pins_hash("external", IMEX) != pins.pins_hash("external", "ssprk3")
    assert pins.pins_hash("external", IMEX) != pins.pins_hash("vertical_modes", IMEX)
    assert pins.scheme_of_pins_hash(pins.pins_hash("vertical_modes", IMEX)) == f"vertical_modes/{IMEX}"
    assert pins.arithmetic_label("external", "rk4") == "external"
    document = pins.pin_document("vertical_modes", IMEX)
    assert set(document) == set(pins.pin_document("vertical_modes"))
    assert "imex" in document["semi_implicit"]
    with pytest.raises(ValueError, match="unknown time integrator"):
        pins.pins_hash("vertical_modes", "leapfrog")
    from pathlib import Path

    base_text = Path(CONFIG).read_text(encoding="utf-8")
    path = tmp_path / "imex.toml"
    path.write_text(base_text.replace('integrator = "ssprk3"', f'integrator = "{IMEX}"'), encoding="utf-8")
    assert load_config(path).integrator == IMEX
    path.write_text(base_text.replace('integrator = "ssprk3"', 'integrator = "leapfrog"'), encoding="utf-8")
    with pytest.raises(ValueError, match="time.integrator must be one of"):
        load_config(path)


def test_split_integrators_never_enter_the_imex_path(monkeypatch):
    """ssprk3 and rk4 keep the split of their era: the IMEX stepper and
    the stage solve are never called under them (the checkpoint arrays of
    a 24-step smoke run measured byte-identical to the pre-IMEX tree for
    both, the CPU host 2026-09-02)."""
    import woof.globe.dynamics as dynamics_module

    def forbidden(*args, **kwargs):
        raise AssertionError("the IMEX path ran under a split integrator")

    monkeypatch.setattr(dynamics_module, "imex_step", forbidden)
    monkeypatch.setattr(VerticalModeSemiImplicit, "solve_shifted", forbidden)
    for integrator in ("ssprk3", "rk4"):
        cfg = dataclasses.replace(load_config(CONFIG), integrator=integrator)
        model, state = build_model_and_cold_state(cfg)
        state, _ = model.step(state, cfg.dt_s)


def test_imex_step_metric_reads_the_implicit_divergence_increment():
    cfg = dataclasses.replace(load_config(CONFIG), integrator=IMEX)
    model, state = build_model_and_cold_state(cfg)
    advanced, metrics = imex_step(
        state.atmosphere, cfg.dt_s, model.rhs, model.semi_implicit, model.transform, model.vertical, IMEX_SSP3,
    )
    assert metrics["semi_implicit_max_divergence_increment_s1"] > 0.0
    assert advanced.time_s == state.atmosphere.time_s + cfg.dt_s
    # Fields the operator never touches ride the explicit tableau only:
    # vorticity and every tracer are finite and changed.
    assert float(np.max(np.abs(advanced.vorticity - state.atmosphere.vorticity))) > 0.0
    assert all(np.all(np.isfinite(f)) for f in advanced.fields())


def test_imex_step_keeps_the_state_precision():
    """The stage values, the accumulators and the advanced state stay in
    the state's own complex dtype: a float32 run never lifts to
    complex128 through a numpy float64 tableau factor (module docstring,
    PRECISION)."""
    cfg = dataclasses.replace(load_config(CONFIG), integrator=IMEX, precision="float32")
    model, state = build_model_and_cold_state(cfg)
    expected = model.transform.backend.complex_dtype
    assert state.atmosphere.divergence.dtype == expected
    seen = []

    def rhs(stage):
        seen.append({name: getattr(stage, name).dtype for name in ("divergence", "theta", "log_surface_pressure", "qv")})
        return model.rhs(stage)

    advanced, _ = imex_step(
        state.atmosphere, cfg.dt_s, rhs, model.semi_implicit, model.transform, model.vertical, IMEX_SSP3,
    )
    assert len(seen) == IMEX_SSP3.stages
    for stage in seen:
        assert all(dtype == expected for dtype in stage.values()), stage
    for name in ("vorticity", "divergence", "theta", "log_surface_pressure", "qv"):
        assert getattr(advanced, name).dtype == expected, name
