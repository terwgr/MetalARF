"""Vertical numerics of the moist hybrid dycore (audit 2026-09-01, DN-3 and DN-4).

DN-3: the full-level geopotential integrated the half layer above each
interface with the level temperature alone, omitting R (dTv/dlnp) dlnp^2/8;
its terrain-following variation was a 1.06 m/s per hour spurious top-level
acceleration at rest over a 2 km mountain (T42, 20-level pressure_blend,
T linear in ln p).  The Simmons-Burridge layer-mean pairing was built and
measured at 1.36 on the same test; the shipped form integrates the half
layer with Tv linear in ln p and pairs it with the midpoint pressure
gradient.  Measured after: 4.27e-7 m/s^2 (0.002 m/s per hour) at T42 and
1.4e-11 at T85, flat control 4e-15 unchanged.

DN-4: theta and every tracer rode a first-order donor-cell vertical flux
(numerical diffusivity |omega| dp / 2).  The van Leer limited flux measures
4.19e-5 K/s against the donor 1.31e-4 and the centred 3.85e-5 on the
auditor's smooth-profile test, converges at order 2 in L1 under
refinement, and admits no new extrema.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# CPU only: the device switch is set for each test in this module by
# `conftest._cpu_only_marked_tests` and put back afterwards, because a
# module that set it at import time decided it for the whole session.
pytestmark = pytest.mark.cpu_only

from woof.globe.constants import (  # noqa: E402
    DRY_AIR_GAS_CONSTANT,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
    SPECTRAL_FIELDS,
)
from woof.globe.dynamics import MoistHybridModel  # noqa: E402
from woof.globe.semi_implicit import BarotropicSemiImplicit  # noqa: E402
from woof.globe.state import MoistHybridState  # noqa: E402
from woof.globe.vertical import HybridCoordinate  # noqa: E402
from woof.globe.spectral.transform import SphericalHarmonicTransform  # noqa: E402

R = DRY_AIR_GAS_CONSTANT


def _model(transform, vertical, surface_geopotential=None):
    if surface_geopotential is None:
        surface_geopotential = np.zeros(transform.grid.shape)
    return MoistHybridModel(
        transform=transform, vertical=vertical,
        surface_geopotential=surface_geopotential, physics=None,
        rotation_rate_s=0.0, diffusion=None,
        semi_implicit=BarotropicSemiImplicit(enabled=False),
        mass_fixer=False, water_fixer=False, positivity_repair=False,
        maximum_cfl=1.0e9,
    )


# --------------------------------------------------------------------- DN-3

P_LO, P_HI = 267.1, 95738.1
DT_DLNP = (286.0 - 220.0) / math.log(P_HI / P_LO)


def _temperature_of_pressure(p):
    return 220.0 + DT_DLNP * np.log(np.asarray(p, dtype=np.float64) / P_LO)


def _exact_geopotential(p, ps, phi_s):
    lp = np.linspace(np.log(np.asarray(p, dtype=np.float64)), np.log(ps), 4001, axis=0)
    return phi_s + np.trapezoid(R * _temperature_of_pressure(np.exp(lp)), lp, axis=0)


@pytest.mark.parametrize(
    "nlev,p_top", [(20, 100.0), (40, 100.0), (20, 1000.0)]
)
def test_full_level_geopotential_is_exact_for_temperature_linear_in_ln_p(nlev, p_top):
    """The auditor's v3 quadrature check: coded minus 4001-point exact."""
    transform = SphericalHarmonicTransform.create(5, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(nlev, p_top)
    model = _model(transform, vertical)
    for height in (0.0, 2000.0):
        ps_value = 1.0e5 * math.exp(-GRAVITY_M_S2 * height / (R * 283.0))
        ps = np.full(transform.grid.shape, ps_value)
        pressure = vertical.pressure(ps, transform.backend)
        temperature = _temperature_of_pressure(pressure["p_full"])
        phi_s = np.zeros(transform.grid.shape) + _exact_geopotential(ps_value, 1.0e5, 0.0)
        model.surface_geopotential = phi_s
        coded = model._hydrostatic_geopotential(temperature, pressure["p_half"])
        exact = _exact_geopotential(
            pressure["p_full"][:, 0, 0], ps_value, float(phi_s[0, 0])
        )
        error = coded[:, 0, 0] - exact
        # The audit measured 306 m2/s2 at the top level (nlev=20, p_top=100)
        # and 120 (nlev=40); the quadrature itself is accurate to ~1e-6.
        assert np.max(np.abs(error)) < 1.0e-3, error
        # The loop specification in vertical.py agrees to roundoff.
        loop = vertical.hydrostatic_geopotential(
            temperature, phi_s, pressure["p_half"], gas_constant=R
        )
        np.testing.assert_allclose(loop, coded, rtol=1.0e-13, atol=1.0e-6)


def _rest_state(transform, vertical, mountain_m):
    lat = transform.grid.lat_rad[:, None]
    lon = transform.grid.lon_rad[None, :]

    def temperature(p):
        return 220.0 + (286.0 - 220.0) * np.clip(
            np.log(p / P_LO) / math.log(P_HI / P_LO), -0.3, 1.2
        )

    h = mountain_m * np.exp(
        -(((lat - math.radians(30)) ** 2 + (np.cos(lat) * (lon - math.pi)) ** 2)
          / math.radians(8) ** 2)
    )
    lnps = np.full(transform.grid.shape, math.log(1.0e5))
    for _ in range(50):
        ps = np.exp(lnps)
        mean = 0.5 * (temperature(ps) + temperature(1.0e5))
        lnps = math.log(1.0e5) - GRAVITY_M_S2 * h / (R * mean)
    lnps_spec = transform.forward(lnps)
    ps = np.exp(transform.inverse(lnps_spec))
    lp = np.linspace(np.log(ps), math.log(1.0e5), 400, axis=0)
    phi_s = np.trapezoid(R * temperature(np.exp(lp)), lp, axis=0)
    p_full = vertical.pressure(ps, transform.backend)["p_full"]
    theta = temperature(p_full) / (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
    zeros = np.zeros((vertical.nlev, *transform.spectral_shape), dtype=np.complex128)
    state = MoistHybridState(
        vorticity=zeros.copy(), divergence=zeros.copy(), theta=transform.forward(theta),
        log_surface_pressure=lnps_spec,
        qv=zeros.copy(),
        **_zero_grid_tracers(transform, vertical),
    )
    return state, phi_s


def _zero_grid_tracers(transform, vertical):
    """The ten grid tracers at exact zero on the Gaussian grid."""
    from woof.globe.constants import GRID_TRACERS

    shape = (vertical.nlev, *transform.grid.shape)
    return {name: np.zeros(shape) for name in GRID_TRACERS}


def _max_acceleration(model, state):
    rhs = model.rhs(state)
    u_t, v_t = model.vector.wind_from_vordiv(rhs.vorticity, rhs.divergence)
    return np.sqrt(u_t ** 2 + v_t ** 2)


def test_rest_state_over_a_mountain_stays_at_rest():
    """The auditor's t6 test: T42, 2 km Gaussian mountain, T linear in ln p.

    Audit baseline 2.933e-4 m/s^2 (1.06 m/s per hour) at level 0, the same
    at T85 (not truncation); flat control 4.9e-15.  Gate: an order of
    magnitude under the baseline (measured 4.27e-7, 686x), flat control at
    roundoff.
    """
    transform = SphericalHarmonicTransform.create(42, backend="numpy", precision="float64")
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    state, phi_s = _rest_state(transform, vertical, 2000.0)
    model = _model(transform, vertical, phi_s)
    acceleration = _max_acceleration(model, state)
    assert float(acceleration.max()) < 2.933e-5, (
        f"{acceleration.max():.3e} m/s^2 = {acceleration.max() * 3600:.3f} m/s per hour"
    )
    flat, _phi = _rest_state(transform, vertical, 0.0)
    model.surface_geopotential = np.zeros(transform.grid.shape)
    assert float(_max_acceleration(model, flat).max()) < 1.0e-12


def test_pressure_gradient_factor_is_the_exact_full_level_gradient():
    """grad(ln p_k) = factor_k grad(ln ps): checked against a finite
    difference of ln p_full between two surface pressures."""
    transform = SphericalHarmonicTransform.create(5, backend="numpy", precision="float64")
    vertical = HybridCoordinate.surface_stretched(20)
    model = _model(transform, vertical)
    ps = np.full(transform.grid.shape, 9.0e4)
    delta = 1.0e-3
    lo = vertical.pressure(ps * math.exp(-delta), transform.backend)
    hi = vertical.pressure(ps * math.exp(delta), transform.backend)
    mid = vertical.pressure(ps, transform.backend)
    finite = (np.log(hi["p_full"]) - np.log(lo["p_full"])) / (2.0 * delta)
    factor = model._pressure_gradient_factor(ps, mid["p_half"])
    np.testing.assert_allclose(factor, finite, rtol=1.0e-5)
    # Pure-pressure layers (B = 0 at both interfaces) feel no gradient.
    assert np.all(factor[0] == 0.0)


# --------------------------------------------------------------------- DN-4

P0, W = 50000.0, 15000.0


def _smooth_profile(p):
    return 300.0 + 1.0e-4 * p + 10.0 * np.exp(-((p - P0) / W) ** 2)


def _smooth_gradient(p):
    return 1.0e-4 - 20.0 * (p - P0) / W ** 2 * np.exp(-((p - P0) / W) ** 2)


def _donor_cell(scalar, omega_half, nlev):
    flux = np.zeros((nlev + 1, *scalar.shape[-2:]))
    interior = omega_half[1:nlev]
    flux[1:nlev] = interior * np.where(interior >= 0.0, scalar[:-1], scalar[1:])
    return flux[1:] - flux[:-1]


@pytest.mark.parametrize("omega", [0.5, -0.5, 0.05])
def test_vertical_scalar_flux_error_falls_from_donor_toward_centred(omega):
    """The auditor's v4 test: smooth profile, constant omega, 20 levels.

    Donor cell measured 1.311e-4 K/s (omega 0.5), the centred momentum
    form 3.848e-5; the limited flux measures 4.185e-5.
    """
    transform = SphericalHarmonicTransform.create(5, backend="numpy", precision="float64")
    nlev = 20
    vertical = HybridCoordinate.pressure_blend(nlev, 100.0)
    model = _model(transform, vertical)
    ps = np.full(transform.grid.shape, 1.0e5)
    pressure = vertical.pressure(ps, transform.backend)
    p_half, p_full, dp = pressure["p_half"], pressure["p_full"], pressure["dp"]
    omega_half = np.full((nlev + 1, *transform.grid.shape), omega)
    omega_half[0] = 0.0
    omega_half[-1] = 0.0
    theta = _smooth_profile(p_full)
    exact = -omega * _smooth_gradient(p_full)
    interior = slice(2, nlev - 2)

    limited = -model._vertical_scalar_flux_divergence(theta, omega_half, p_full, p_half) / dp
    donor = -_donor_cell(theta, omega_half, nlev) / dp
    centred = -model._vertical_momentum_advection(theta, omega_half, p_full)
    e_limited = np.max(np.abs(limited[interior, 0, 0] - exact[interior, 0, 0]))
    e_donor = np.max(np.abs(donor[interior, 0, 0] - exact[interior, 0, 0]))
    e_centred = np.max(np.abs(centred[interior, 0, 0] - exact[interior, 0, 0]))
    assert e_limited < 0.4 * e_donor, (e_limited, e_donor)
    assert e_limited < 1.15 * e_centred, (e_limited, e_centred)
    # A linear profile is reconstructed exactly: the interface value is the
    # interface pressure, so the tendency is -omega in every interior layer.
    linear = -model._vertical_scalar_flux_divergence(p_full.copy(), omega_half, p_full, p_half) / dp
    np.testing.assert_allclose(linear[interior], -omega, rtol=1.0e-12)


def test_vertical_scalar_flux_converges_at_second_order_in_l1():
    transform = SphericalHarmonicTransform.create(5, backend="numpy", precision="float64")
    ps = np.full(transform.grid.shape, 1.0e5)
    errors = []
    for nlev in (40, 80, 160):
        vertical = HybridCoordinate.pressure_blend(nlev, 100.0)
        model = _model(transform, vertical)
        pressure = vertical.pressure(ps, transform.backend)
        p_half, p_full, dp = pressure["p_half"], pressure["p_full"], pressure["dp"]
        omega_half = np.full((nlev + 1, *transform.grid.shape), 0.5)
        omega_half[0] = 0.0
        omega_half[-1] = 0.0
        tendency = -model._vertical_scalar_flux_divergence(
            _smooth_profile(p_full), omega_half, p_full, p_half
        ) / dp
        exact = -0.5 * _smooth_gradient(p_full)
        band = (p_full[:, 0, 0] > 1.5e4) & (p_full[:, 0, 0] < 9.0e4)
        weight = dp[band, 0, 0] / np.sum(dp[band, 0, 0])
        errors.append(float(np.sum(weight * np.abs(tendency[band, 0, 0] - exact[band, 0, 0]))))
    orders = [math.log2(a / b) for a, b in zip(errors, errors[1:])]
    # Measured 2.04, 2.03 (the max norm is capped near 1 at the profile's
    # extremum, where every TVD limiter clips to first order).
    assert min(orders) > 1.8, (errors, orders)


@pytest.mark.parametrize("omega", [0.5, -0.5])
def test_vertical_scalar_flux_admits_no_new_extrema(omega):
    """Top hat advected with SSPRK3 at CFL 0.2 through 60 steps (7.4 kPa,
    inside the column): bounds hold and total variation does not grow."""
    transform = SphericalHarmonicTransform.create(5, backend="numpy", precision="float64")
    nlev = 20
    vertical = HybridCoordinate.pressure_blend(nlev, 100.0)
    model = _model(transform, vertical)
    ps = np.full(transform.grid.shape, 1.0e5)
    pressure = vertical.pressure(ps, transform.backend)
    p_half, p_full, dp = pressure["p_half"], pressure["p_full"], pressure["dp"]
    omega_half = np.full((nlev + 1, *transform.grid.shape), omega)
    omega_half[0] = 0.0
    omega_half[-1] = 0.0
    scalar = np.where((p_full > 4.0e4) & (p_full < 6.0e4), 1.0, 0.0)
    dt = 0.2 * float(np.min(dp)) / abs(omega)

    def tendency(s):
        return -model._vertical_scalar_flux_divergence(s, omega_half, p_full, p_half) / dp

    def centroid(s):
        column = s[:, 0, 0] * dp[:, 0, 0]
        return float(np.sum(column * p_full[:, 0, 0]) / np.sum(column))

    variation_before = float(np.sum(np.abs(np.diff(scalar[:, 0, 0]))))
    centroid_before = centroid(scalar)
    for _ in range(60):
        s1 = scalar + dt * tendency(scalar)
        s2 = 0.75 * scalar + 0.25 * (s1 + dt * tendency(s1))
        scalar = scalar / 3.0 + 2.0 / 3.0 * (s2 + dt * tendency(s2))
        assert scalar.min() >= -1.0e-12 and scalar.max() <= 1.0 + 1.0e-12
    variation_after = float(np.sum(np.abs(np.diff(scalar[:, 0, 0]))))
    assert variation_after <= variation_before + 1.0e-12
    # The hat travelled ~7.4 kPa in the direction of omega, so the bounds
    # were tested on a moving discontinuity, not a static field.
    travel = centroid(scalar) - centroid_before
    assert travel * omega > 0.0 and abs(travel) > 5.0e3, travel


def test_stacked_tracers_never_share_a_flux():
    """Every tracer block in the stacked call equals its own single call."""
    transform = SphericalHarmonicTransform.create(5, backend="numpy", precision="float64")
    nlev = 9
    vertical = HybridCoordinate.pressure_blend(nlev, 100.0)
    model = _model(transform, vertical)
    rng = np.random.default_rng(2)
    ps = np.full(transform.grid.shape, 1.0e5)
    pressure = vertical.pressure(ps, transform.backend)
    p_half, p_full = pressure["p_half"], pressure["p_full"]
    omega_half = 0.3 * rng.standard_normal((nlev + 1, *transform.grid.shape))
    omega_half[0] = 0.0
    omega_half[-1] = 0.0
    stacked = rng.standard_normal((4, nlev, *transform.grid.shape))
    together = model._vertical_scalar_flux_divergence(stacked, omega_half, p_full, p_half)
    for index in range(stacked.shape[0]):
        alone = model._vertical_scalar_flux_divergence(stacked[index], omega_half, p_full, p_half)
        assert np.array_equal(together[index], alone)
