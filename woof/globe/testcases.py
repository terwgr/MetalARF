"""Analytic dynamical-core test cases with a published answer.

Two cases live here, and both exist because an integrator that only ever
runs against real data can be wrong in a way nothing measures.

*   The Jablonowski and Williamson 2006 baroclinic wave.  A steady,
    balanced, analytically specified state that a correct core keeps
    steady for days, plus the same state with a 1 m/s wind perturbation
    that grows into a wave train.  The steady arm measures whether the
    core is a fixed point of a state it did not construct; the perturbed
    arm measures whether it grows the instability at the right rate, and
    an integrator that damped the baroclinic growth would pass every
    conservation gate in the tree while forecasting nothing.

*   The Held and Suarez 1994 forcing.  Newtonian relaxation to a
    prescribed equilibrium temperature plus Rayleigh friction in the
    boundary layer, with no radiation, no moisture and no surface: the
    standard way to ask what a dynamical core's CLIMATE looks like after
    a thousand days, which is the timescale on which an over-dissipative
    interpolation shows up and a twenty-four hour forecast does not.

The forcing is written as a state-to-state operator rather than as a
physics suite.  It relaxes temperature and wind and touches nothing else,
so a suite's exchange, closure ledger and water reservoir would all be
machinery with nothing to do; the split is the one the case itself
specifies.
"""
from __future__ import annotations

import math

import numpy as np

from woof.globe.spectral.vector import VorticityDivergenceOperator

from .constants import (
    DRY_AIR_GAS_CONSTANT,
    EARTH_ROTATION_RATE_S,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
)
from .state import ArwenGlobalState, MoistHybridState, PhysicsState, SurfaceState

# --------------------------------------------------------------------------
# Jablonowski and Williamson 2006.

#: Jet speed of the analytic zonal flow, m/s.
JW_U0 = 35.0
#: Reference surface temperature and lapse rate of the mean profile.
JW_T0 = 288.0
JW_LAPSE_K_M = 0.005
#: Stratospheric temperature increment and the level it starts at.
JW_DELTA_T = 4.8e5
JW_ETA_TOP = 0.2
#: Centre of the vertical jet profile.
JW_ETA_0 = 0.252
#: Uniform surface pressure of the case, Pa.
JW_SURFACE_PRESSURE_PA = 1.0e5
#: The perturbation: 1 m/s of zonal wind in a Gaussian of radius a/10
#: centred at 20 E, 40 N.
JW_PERTURBATION_M_S = 1.0
JW_PERTURBATION_RADIUS_FRACTION = 0.1
JW_PERTURBATION_LON_RAD = math.pi / 9.0
JW_PERTURBATION_LAT_RAD = 2.0 * math.pi / 9.0


def _jw_shape(sin_lat, cos_lat):
    """The two latitude shapes both the temperature and the geopotential
    of the analytic state are written in."""
    first = -2.0 * sin_lat ** 6 * (cos_lat ** 2 + 1.0 / 3.0) + 10.0 / 63.0
    second = (
        (8.0 / 5.0) * cos_lat ** 3 * (sin_lat ** 2 + 2.0 / 3.0) - math.pi / 4.0
    )
    return first, second


def jablonowski_williamson_fields(
    eta, lat_rad, *, radius_m: float, rotation_rate_s: float = EARTH_ROTATION_RATE_S,
    gas_constant: float = DRY_AIR_GAS_CONSTANT,
):
    """``(u, temperature, surface_geopotential)`` of the steady state.

    ``eta`` is ``p / ps`` at the model's full levels, shaped to broadcast
    against ``lat_rad``.  Everything is float64 host arithmetic: the case
    is a specification and the model reads it once.
    """
    eta = np.asarray(eta, dtype=np.float64)
    lat = np.asarray(lat_rad, dtype=np.float64)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    first, second = _jw_shape(sin_lat, cos_lat)
    exponent = gas_constant * JW_LAPSE_K_M / GRAVITY_M_S2

    eta_v = (eta - JW_ETA_0) * (math.pi / 2.0)
    cos_v = np.cos(eta_v)
    zonal = JW_U0 * cos_v ** 1.5 * np.sin(2.0 * lat) ** 2

    mean_t = JW_T0 * eta ** exponent
    mean_t = np.where(
        eta < JW_ETA_TOP,
        mean_t + JW_DELTA_T * (JW_ETA_TOP - eta) ** 5,
        mean_t,
    )
    temperature = mean_t + 0.75 * (
        eta * math.pi * JW_U0 / gas_constant
    ) * np.sin(eta_v) * np.sqrt(cos_v) * (
        first * 2.0 * JW_U0 * cos_v ** 1.5
        + second * radius_m * rotation_rate_s
    )

    # The surface geopotential is the same expression at eta = 1, whose
    # mean part is zero: the case's orography IS the balance of its jet.
    surface_v = (1.0 - JW_ETA_0) * (math.pi / 2.0)
    cos_s = math.cos(surface_v)
    surface_geopotential = JW_U0 * cos_s ** 1.5 * (
        first * JW_U0 * cos_s ** 1.5 + second * radius_m * rotation_rate_s
    )
    return zonal, temperature, surface_geopotential


def jablonowski_williamson_perturbation(lat_rad, lon_rad, radius_m: float):
    """The 1 m/s zonal wind bump, as a function of position alone."""
    lat = np.asarray(lat_rad, dtype=np.float64)
    lon = np.asarray(lon_rad, dtype=np.float64)
    cos_r = (
        math.sin(JW_PERTURBATION_LAT_RAD) * np.sin(lat)
        + math.cos(JW_PERTURBATION_LAT_RAD) * np.cos(lat)
        * np.cos(lon - JW_PERTURBATION_LON_RAD)
    )
    great_circle = np.arccos(np.clip(cos_r, -1.0, 1.0))
    scale = JW_PERTURBATION_RADIUS_FRACTION
    return JW_PERTURBATION_M_S * np.exp(-((great_circle / scale) ** 2))


def baroclinic_wave_initial_state(cfg, transform, statics=None):
    """The Jablonowski and Williamson 2006 state, in this model's own
    variables: ``(bundle, surface_geopotential, provenance)``.

    ``cfg.perturbation_amplitude`` scales the wind bump, so the STEADY
    arm of the gate is the same config with the amplitude at zero and
    therefore the same everything else.
    """
    backend = transform.backend
    xp = backend.xp
    grid = transform.grid
    lat = np.asarray(grid.lat_rad, dtype=np.float64)[:, None]
    lon = np.asarray(grid.lon_rad, dtype=np.float64)[None, :]
    ps = np.full(grid.shape, JW_SURFACE_PRESSURE_PA)
    pressure = cfg.vertical.pressure(
        np.asarray(ps, dtype=np.float64), _HostBackend()
    )
    eta = (pressure["p_full"] / JW_SURFACE_PRESSURE_PA)[:, :1, :1]
    zonal, temperature, surface_geopotential = jablonowski_williamson_fields(
        eta, lat[None], radius_m=grid.radius_m,
    )
    u = np.broadcast_to(zonal, pressure["p_full"].shape).copy()
    amplitude = float(cfg.perturbation_amplitude)
    if amplitude != 0.0:
        u = u + amplitude * jablonowski_williamson_perturbation(
            lat, lon, grid.radius_m
        )[None]
    temperature = np.broadcast_to(
        temperature, pressure["p_full"].shape
    ).copy()
    theta = temperature / (pressure["p_full"] / REFERENCE_PRESSURE_PA) ** KAPPA
    surface_geopotential = np.broadcast_to(
        np.asarray(surface_geopotential).reshape(-1)[:, None], grid.shape
    ).copy()

    dtype = backend.float_dtype
    vector = VorticityDivergenceOperator(transform)
    vorticity, divergence = vector.vordiv_from_wind(
        backend.asarray(u, dtype=dtype),
        backend.asarray(np.zeros_like(u), dtype=dtype),
    )
    zeros_grid = xp.zeros(
        (cfg.vertical.nlev, *grid.shape), dtype=dtype
    )
    zeros_spectral = xp.zeros(
        (cfg.vertical.nlev, *transform.spectral_shape),
        dtype=backend.complex_dtype,
    )
    atmosphere = MoistHybridState(
        vorticity=vorticity,
        divergence=divergence,
        theta=transform.project(
            transform.forward(backend.asarray(theta, dtype=dtype))
        ),
        log_surface_pressure=transform.project(
            transform.forward(backend.asarray(np.log(ps), dtype=dtype))
        ),
        qv=zeros_spectral,
        **{
            name: zeros_grid.copy()
            for name in ("qc", "qr", "qi", "qs", "qg",
                         "nc", "nr", "ni", "ns", "ng")
        },
    )
    surface = _dry_surface(transform, temperature[-1])
    provenance = {
        "mode": "baroclinic_wave",
        "case": "jablonowski-williamson-2006",
        "perturbation_m_s": amplitude,
        "surface_pressure_pa": JW_SURFACE_PRESSURE_PA,
    }
    return (
        ArwenGlobalState(atmosphere, surface, PhysicsState()),
        backend.asarray(surface_geopotential, dtype=dtype),
        provenance,
    )


class _HostBackend:
    """The minimal backend ``HybridCoordinate.pressure`` reads, in float64
    on the host: the case is a specification and is built once."""

    name = "numpy"
    xp = np
    float_dtype = np.float64
    complex_dtype = np.complex128

    @staticmethod
    def asarray(value, dtype=None):
        return np.asarray(value, dtype=dtype)

    @staticmethod
    def to_numpy(value):
        return np.asarray(value)


def _dry_surface(transform, surface_temperature) -> SurfaceState:
    """A surface that no scheme reads: the dry cases run with physics off,
    and the state still has to carry a valid one."""
    backend = transform.backend
    xp = backend.xp
    dtype = backend.float_dtype
    shape = transform.grid.shape
    temperature = backend.asarray(
        np.broadcast_to(np.asarray(surface_temperature), shape).copy(),
        dtype=dtype,
    )
    from .statics import synthetic_surface_statics

    nsoil = 4
    soil = xp.stack([temperature for _ in range(nsoil)])
    land_fraction = np.zeros(shape)
    resolved = synthetic_surface_statics(
        land_fraction, np.broadcast_to(
            backend.to_numpy(temperature)[None], (nsoil, *shape)
        ).copy(),
    )
    resolved.pop("land_fraction")
    return SurfaceState(
        temperature_k=temperature,
        water_kg_m2=xp.zeros(shape, dtype=dtype),
        land_fraction=xp.zeros(shape, dtype=dtype),
        heat_capacity_j_m2_k=xp.full(shape, 4.0e7, dtype=dtype),
        soil_temperature_k=soil,
        soil_water_fraction=xp.full((nsoil, *shape), 0.25, dtype=dtype),
        accumulated_rain_kg_m2=xp.zeros(shape, dtype=dtype),
        accumulated_snow_kg_m2=xp.zeros(shape, dtype=dtype),
        accumulated_graupel_kg_m2=xp.zeros(shape, dtype=dtype),
        **{name: backend.asarray(value, dtype=dtype)
           for name, value in resolved.items()},
    )


# --------------------------------------------------------------------------
# Held and Suarez 1994.

#: Relaxation rates, per second: the free atmosphere, the boundary layer,
#: and the Rayleigh friction of the boundary layer.
HS_KA_S = 1.0 / (40.0 * 86400.0)
HS_KS_S = 1.0 / (4.0 * 86400.0)
HS_KF_S = 1.0 / (1.0 * 86400.0)
#: The sigma the boundary-layer terms switch on at.
HS_SIGMA_B = 0.7
#: The equilibrium profile's constants.
HS_T_EQUATOR_K = 315.0
HS_T_FLOOR_K = 200.0
HS_DELTA_T_Y_K = 60.0
HS_DELTA_THETA_Z_K = 10.0


def held_suarez_equilibrium(p_full, sin_lat, cos_lat):
    """``(T_eq, k_T, k_v)`` of the Held and Suarez forcing.

    ``p_full`` is in Pa and the sigma the boundary-layer terms read is
    ``p / p0`` with ``p0`` the reference pressure, which is the case's own
    definition (its surface pressure is 1000 hPa by construction).
    """
    xp = _array_module(p_full)
    sigma = p_full / REFERENCE_PRESSURE_PA
    log_sigma = xp.log(sigma)
    equilibrium = xp.maximum(
        HS_T_FLOOR_K,
        (
            HS_T_EQUATOR_K
            - HS_DELTA_T_Y_K * sin_lat ** 2
            - HS_DELTA_THETA_Z_K * log_sigma * cos_lat ** 2
        ) * sigma ** KAPPA,
    )
    boundary = xp.maximum(0.0, (sigma - HS_SIGMA_B) / (1.0 - HS_SIGMA_B))
    k_t = HS_KA_S + (HS_KS_S - HS_KA_S) * boundary * cos_lat ** 4
    k_v = HS_KF_S * boundary
    return equilibrium, k_t, k_v


def _array_module(value):
    module = type(value).__module__.split(".")[0]
    if module == "cupy":
        import cupy

        return cupy
    return np


def apply_held_suarez(model, bundle: ArwenGlobalState, dt_s: float):
    """One Held and Suarez forcing step, applied to a spectral state.

    Both terms are integrated BACKWARD in time (``x <- (x + dt k x_eq) /
    (1 + dt k)``) rather than forward.  The relaxation is a stiff linear
    term whose fastest rate is one over four days, and a forward step is
    stable at every time step this model runs; the backward form is used
    because it is stable at EVERY time step, so the thousand-day climate
    the case measures cannot depend on the step through the forcing's own
    stability rather than through the dynamics the gate is about.
    """
    transform = model.transform
    backend = transform.backend
    xp = backend.xp
    dtype = backend.float_dtype
    state = bundle.atmosphere
    g = model.grid_state(state, only=("temperature", "p_full", "u", "v"))
    sin_lat = backend.asarray(
        transform.grid.sin_lat[:, None], dtype=dtype
    )[None]
    cos_lat = backend.asarray(
        transform.grid.cos_lat[:, None], dtype=dtype
    )[None]
    equilibrium, k_t, k_v = held_suarez_equilibrium(
        g["p_full"], sin_lat, cos_lat
    )
    dt = dtype(float(dt_s))
    temperature = (g["temperature"] + dt * k_t * equilibrium) / (1.0 + dt * k_t)
    exner = (g["p_full"] / REFERENCE_PRESSURE_PA) ** KAPPA
    theta = transform.project(transform.forward(temperature / exner))
    del temperature, exner, equilibrium, k_t
    damping = 1.0 / (1.0 + dt * k_v)
    vorticity, divergence = model.vector.vordiv_from_wind(
        g["u"] * damping, g["v"] * damping
    )
    del damping, k_v, g
    advanced = state.with_fields(
        (vorticity, divergence, theta, state.log_surface_pressure, state.qv)
    )
    return ArwenGlobalState(advanced, bundle.surface, bundle.physics_state)


__all__ = [
    "HS_KA_S",
    "HS_KF_S",
    "HS_KS_S",
    "HS_SIGMA_B",
    "JW_SURFACE_PRESSURE_PA",
    "JW_U0",
    "apply_held_suarez",
    "baroclinic_wave_initial_state",
    "held_suarez_equilibrium",
    "jablonowski_williamson_fields",
    "jablonowski_williamson_perturbation",
]
