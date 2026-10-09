"""Monin-Obukhov similarity diagnostics of 2 m temperature/humidity and 10 m wind.

The reference suite, and the render-tape export as a fallback, diagnose the
screen-level state from the skin and the lowest full level instead of
exporting the lowest level itself (audit 2026-09-01, task 1b: ``T2 =
temperature[-1]`` was 378 m up on the 20-level grid and 23 m up on the
40-level one; neither is 2 m).

Formulation (Businger-Dyer flux-profile relations with the Paulson 1970
integrated stability functions, the same family WRF's revised MM5 surface
layer uses):

    U(z)         = (u*/k) [ln(z/z0)  - psi_m(z/L) + psi_m(z0/L)]   =: (u*/k) F(z)
    th(z) - th_s = (th*/k)[ln(z/z0h) - psi_h(z/L) + psi_h(z0h/L)]  =: (th*/k) G(z)
    q(z)  - q_s  = (q*/k) G(z)

with k = 0.4, thermal roughness z0h = z0/10, and

    unstable (zeta = z/L < 0): x = (1 - 16 zeta)^(1/4)
        psi_m = 2 ln((1+x)/2) + ln((1+x^2)/2) - 2 atan(x) + pi/2
        psi_h = 2 ln((1+x^2)/2)
    stable   (zeta >= 0):  psi_m = psi_h = -5 zeta

Given the lowest full level z1 with wind U1, potential temperature th1 and
humidity q1, the skin th_s, q_s, the flux scales cancel and

    U10 = U1 F(10)/F(z1),   th2 = th_s + (th1 - th_s) G(2)/G(z1),
    q2  = q_s  + (q1  - q_s)  G(2)/G(z1),   T2 = th2 (p_2m/p0)^kappa.

zeta follows from the bulk Richardson number of the layer,
Ri_b = g z1 (thv1 - thv_s) / (thv_mean U1^2), through the exact MOST
identity zeta = Ri_b F(z1)^2 / G(z1): a closed-form quadratic on the
stable branch (Ri_b capped at 0.19, just under the 0.2 beyond which the
linear stable functions admit no solution) and five fixed-point sweeps on
the unstable branch (zeta floored at -10, free convection).

Surface humidity q_s is the bucket model's effective value,
q_s = q1 + beta (qsat(T_s, p_s) - q1) with beta = 1 over water and the
soil-wetness fraction over land, so the moisture profile is the one the
evaporation was computed against.
"""
from __future__ import annotations

import math

import numpy as np

from ..constants import (
    DRY_AIR_GAS_CONSTANT,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
)
from .reference import saturation_mixing_ratio

VON_KARMAN = 0.4
#: Diagnostic heights (m).
SCREEN_HEIGHT_M = 2.0
ANEMOMETER_HEIGHT_M = 10.0
#: Thermal roughness as a fraction of momentum roughness.
THERMAL_ROUGHNESS_FRACTION = 0.1
#: Stable-branch Richardson cap: the linear Businger-Dyer stable functions
#: have no solution at Ri_b >= 0.2, so the layer is treated as at most
#: this stable (decoupled beyond it in every scheme of this family).
STABLE_RICHARDSON_CAP = 0.19
#: Stable cap on zeta = z/L: the linear stable functions were fitted to
#: zeta <= ~2 and are used to ~5 in practice; beyond that a 2 km-deep
#: bulk layer at the Richardson cap would solve to zeta ~ 30 and reduce a
#: 2 m/s lowest-level wind to 0.03 m/s at 10 m.
STABLE_ZETA_CAP = 5.0
#: Free-convection floor on zeta = z/L.
UNSTABLE_ZETA_FLOOR = -10.0
#: Wind-speed floor (m/s) so Ri_b stays finite in calm air.
WIND_FLOOR_M_S = 0.1
#: Lowest-level height floor (m): the profile functions need z1 > z0 with
#: ln(z1/z0) away from zero; below this the layer is inside the roughness
#: sublayer where similarity does not hold.
LOWEST_LEVEL_FLOOR_M = 5.0
DIAGNOSTIC_FIELDS = ("t2", "th2", "q2", "u10", "v10")
#: physics_state.metadata key naming which scheme produced the fields.
SOURCE_METADATA_KEY = "surface_diagnostics"
SIMILARITY_SOURCE = "reference-similarity"


def _psi_functions(zeta, xp):
    unstable = zeta < 0.0
    x = (1.0 - 16.0 * xp.minimum(zeta, 0.0)) ** 0.25
    psi_m_unstable = (
        2.0 * xp.log(0.5 * (1.0 + x)) + xp.log(0.5 * (1.0 + x * x))
        - 2.0 * xp.arctan(x) + 0.5 * math.pi
    )
    psi_h_unstable = 2.0 * xp.log(0.5 * (1.0 + x * x))
    stable = -5.0 * xp.maximum(zeta, 0.0)
    psi_m = xp.where(unstable, psi_m_unstable, stable)
    psi_h = xp.where(unstable, psi_h_unstable, stable)
    return psi_m, psi_h


def _profile_functions(z, z0, z0h, zeta_z1, z1, xp):
    """F(z) and G(z) for zeta(z) = zeta_z1 * z / z1."""
    psi_m_z, psi_h_z = _psi_functions(zeta_z1 * z / z1, xp)
    psi_m_0, _ = _psi_functions(zeta_z1 * z0 / z1, xp)
    _, psi_h_0 = _psi_functions(zeta_z1 * z0h / z1, xp)
    f = xp.log(z / z0) - psi_m_z + psi_m_0
    g = xp.log(z / z0h) - psi_h_z + psi_h_0
    return f, g


def effective_surface_humidity(
    skin_temperature_k, surface_pressure_pa, land_fraction, soil_wetness, qv_lowest, xp
):
    """q_s = q1 + beta (qsat(T_s, p_s) - q1), beta = 1 - land + land * wetness."""
    qsat = saturation_mixing_ratio(skin_temperature_k, surface_pressure_pa, xp)
    beta = 1.0 - land_fraction + land_fraction * xp.clip(soil_wetness, 0.0, 1.0)
    return qv_lowest + beta * (qsat - qv_lowest)


def lowest_level_height_m(virtual_temperature_lowest, p_surface_pa, p_full_lowest_pa, xp):
    """Hydrostatic height of the lowest full level above the surface."""
    return (
        DRY_AIR_GAS_CONSTANT * virtual_temperature_lowest / GRAVITY_M_S2
        * xp.log(p_surface_pa / p_full_lowest_pa)
    )


def similarity_surface_diagnostics(
    *,
    u_lowest,
    v_lowest,
    temperature_lowest,
    qv_lowest,
    p_full_lowest_pa,
    p_surface_pa,
    skin_temperature_k,
    surface_humidity,
    roughness_m,
    xp=np,
) -> dict[str, object]:
    """2 m temperature/humidity and 10 m wind by the module formulation.

    Returns ``t2``, ``th2``, ``q2``, ``u10``, ``v10`` plus ``z_lowest_m``
    and ``zeta`` (z1/L) for inspection.  Every input is a 2-D plane on the
    physics grid; arrays stay in the caller's array module.
    """
    theta_lowest = temperature_lowest * (REFERENCE_PRESSURE_PA / p_full_lowest_pa) ** KAPPA
    theta_skin = skin_temperature_k * (REFERENCE_PRESSURE_PA / p_surface_pa) ** KAPPA
    virtual_lowest = temperature_lowest * (1.0 + 0.61 * qv_lowest)
    z1 = xp.maximum(
        lowest_level_height_m(virtual_lowest, p_surface_pa, p_full_lowest_pa, xp),
        LOWEST_LEVEL_FLOOR_M,
    )
    z0 = xp.clip(roughness_m, 1.0e-5, 0.25 * z1)
    z0h = THERMAL_ROUGHNESS_FRACTION * z0
    speed = xp.maximum(xp.sqrt(u_lowest ** 2 + v_lowest ** 2), WIND_FLOOR_M_S)

    thv1 = theta_lowest * (1.0 + 0.61 * qv_lowest)
    thvs = theta_skin * (1.0 + 0.61 * surface_humidity)
    richardson = (
        GRAVITY_M_S2 * z1 * (thv1 - thvs) / (0.5 * (thv1 + thvs) * speed ** 2)
    )
    ln_m = xp.log(z1 / z0)
    ln_h = xp.log(z1 / z0h)

    # Stable branch: zeta (ln_h + 5 zeta) = Ri (ln_m + 5 zeta)^2 (the psi(z0/L)
    # terms are dropped here, z0 << z1), one positive root for Ri < 0.2.
    ri_stable = xp.clip(richardson, 0.0, STABLE_RICHARDSON_CAP)
    qa = 5.0 * (1.0 - 5.0 * ri_stable)
    qb = ln_h - 10.0 * ri_stable * ln_m
    qc = -ri_stable * ln_m ** 2
    zeta_stable = xp.minimum(
        (-qb + xp.sqrt(xp.maximum(qb * qb - 4.0 * qa * qc, 0.0))) / (2.0 * qa),
        STABLE_ZETA_CAP,
    )
    # Unstable branch: fixed point of zeta = Ri F(z1)^2 / G(z1).
    ri_unstable = xp.minimum(richardson, 0.0)
    zeta_unstable = xp.maximum(ri_unstable * ln_m ** 2 / ln_h, UNSTABLE_ZETA_FLOOR)
    for _ in range(5):
        f1, g1 = _profile_functions(z1, z0, z0h, zeta_unstable, z1, xp)
        zeta_unstable = xp.maximum(ri_unstable * f1 ** 2 / g1, UNSTABLE_ZETA_FLOOR)
    zeta = xp.where(richardson < 0.0, zeta_unstable, zeta_stable)

    f1, g1 = _profile_functions(z1, z0, z0h, zeta, z1, xp)
    f10, _ = _profile_functions(
        xp.full_like(z1, ANEMOMETER_HEIGHT_M), z0, z0h, zeta, z1, xp
    )
    _, g2 = _profile_functions(
        xp.full_like(z1, SCREEN_HEIGHT_M), z0, z0h, zeta, z1, xp
    )
    wind_factor = f10 / f1
    scalar_factor = g2 / g1
    th2 = theta_skin + (theta_lowest - theta_skin) * scalar_factor
    q2 = surface_humidity + (qv_lowest - surface_humidity) * scalar_factor
    # 2 m pressure: hydrostatic from the surface over 2 m of skin-side air.
    p2 = p_surface_pa * xp.exp(
        -GRAVITY_M_S2 * SCREEN_HEIGHT_M
        / (DRY_AIR_GAS_CONSTANT * 0.5 * (virtual_lowest + skin_temperature_k))
    )
    t2 = th2 * (p2 / REFERENCE_PRESSURE_PA) ** KAPPA
    return {
        "t2": t2,
        "th2": th2,
        "q2": xp.maximum(q2, 0.0),
        "u10": u_lowest * wind_factor,
        "v10": v_lowest * wind_factor,
        "z_lowest_m": z1,
        "zeta": zeta,
    }


__all__ = [
    "DIAGNOSTIC_FIELDS",
    "SIMILARITY_SOURCE",
    "SOURCE_METADATA_KEY",
    "effective_surface_humidity",
    "lowest_level_height_m",
    "similarity_surface_diagnostics",
]
