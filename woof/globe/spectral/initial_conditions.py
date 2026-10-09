"""Analytic initial conditions for transform and dynamical-core verification."""
from __future__ import annotations

import math

import numpy as np

from .constants import (
    DRY_AIR_GAS_CONSTANT,
    EARTH_ROTATION_RATE_S,
    GRAVITY_M_S2,
    REFERENCE_PRESSURE_PA,
    SECONDS_PER_DAY,
)
from .state import PrimitiveDryState, ShallowWaterState
from .transform import SphericalHarmonicTransform
from .vector import VorticityDivergenceOperator


def williamson2_state(
    transform: SphericalHarmonicTransform,
    *,
    alpha_rad: float = 0.0,
    u0_m_s: float | None = None,
    mean_geopotential_m2_s2: float = 2.94e4,
    rotation_rate_s: float = EARTH_ROTATION_RATE_S,
) -> ShallowWaterState:
    """Williamson et al. test case 2: steady nonlinear geostrophic flow."""
    a = transform.grid.radius_m
    u0 = 2.0 * math.pi * a / (12.0 * SECONDS_PER_DAY) if u0_m_s is None else float(u0_m_s)
    lat, lon = transform.grid.mesh()
    sinp = np.sin(lat)
    cosp = np.cos(lat)
    ca = math.cos(alpha_rad)
    sa = math.sin(alpha_rad)
    u = u0 * (ca * cosp + sa * sinp * np.cos(lon))
    v = -u0 * sa * np.sin(lon)
    rotated_sin_lat = -np.cos(lon) * cosp * sa + sinp * ca
    geopotential = mean_geopotential_m2_s2 - (
        a * rotation_rate_s * u0 + 0.5 * u0 * u0
    ) * rotated_sin_lat * rotated_sin_lat
    if abs(float(alpha_rad)) <= 1.0e-14:
        # The admitted orientation has an independent analytic carrier:
        # solid-body u=u0*cos(phi) gives zeta=2*u0*sin(phi)/a and D=0.
        # Do not derive the verification initial state with the same vector
        # transform whose dynamics it is meant to test.
        analytic_zeta = 2.0 * u0 * sinp / a
        zeta = transform.forward(
            transform.backend.asarray(
                analytic_zeta, dtype=transform.backend.float_dtype
            )
        )
        div = transform.zeros()
    else:
        # Kept as a helper for deriving the future tilted-axis arm.  Config
        # admission refuses it until its discrete Coriolis balance closes.
        vector = VorticityDivergenceOperator(transform)
        zeta, div = vector.vordiv_from_wind(
            transform.backend.asarray(u, dtype=transform.backend.float_dtype),
            transform.backend.asarray(v, dtype=transform.backend.float_dtype),
        )
    phi = transform.forward(
        transform.backend.asarray(geopotential, dtype=transform.backend.float_dtype)
    )
    return ShallowWaterState(zeta, div, transform.project(phi))


def primitive_rest_state(
    transform: SphericalHarmonicTransform,
    sigma_full: np.ndarray,
    *,
    surface_pressure_pa: float = REFERENCE_PRESSURE_PA,
    temperature_surface_k: float = 288.0,
    temperature_top_k: float = 215.0,
    perturbation_k: float = 0.0,
    zonal_wavenumber: int = 4,
) -> PrimitiveDryState:
    """Resting dry sigma atmosphere, optionally with a smooth thermal wave."""
    sigma = np.asarray(sigma_full, dtype=np.float64)
    if np.any(sigma <= 0) or np.any(sigma > 1):
        raise ValueError("sigma_full must lie in (0, 1]")
    nlev = sigma.size
    lat, lon = transform.grid.mesh()
    vertical = temperature_top_k + (
        temperature_surface_k - temperature_top_k
    ) * sigma ** (DRY_AIR_GAS_CONSTANT / 1004.0)
    temp = np.broadcast_to(vertical[:, None, None], (nlev, *transform.grid.shape)).copy()
    if perturbation_k:
        envelope = np.exp(-((sigma - 0.55) / 0.22) ** 2)
        wave = np.cos(lat) ** 2 * np.cos(int(zonal_wavenumber) * lon)
        temp += float(perturbation_k) * envelope[:, None, None] * wave[None]
    zeros = transform.zeros(nlev)
    temperature = transform.forward(
        transform.backend.asarray(temp, dtype=transform.backend.float_dtype)
    )
    logps = transform.constant_coeff(math.log(float(surface_pressure_pa)))
    return PrimitiveDryState(zeros.copy(), zeros.copy(), temperature, logps)
