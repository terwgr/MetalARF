"""Smooth analytic initial atmosphere and surface for WOOF global controls."""
from __future__ import annotations

import math

import numpy as np

from woof.globe.spectral.vector import VorticityDivergenceOperator

from .constants import GRAVITY_M_S2, KAPPA, REFERENCE_PRESSURE_PA
from .state import ArwenGlobalState, MoistHybridState, PhysicsState, SurfaceState


def analytic_initial_state(cfg, transform, statics=None):
    """Return ``(state, surface_geopotential, provenance)``.

    The analytic planet's land mask is a formula, so its statics default
    to the declared synthetic planet (printed as such).  A config that
    selects ``[statics] source = "real"`` with a ``valid_time`` puts the
    WPS_GEOG fields -- land fraction included -- onto the analytic
    atmosphere instead; ``statics`` injects that ``(fields, provenance)``
    pair for unit gates.
    """
    from .statics import (
        SYNTHETIC_CONVENTION, cache_paths, load_statics, real_convention,
        real_provenance, resolve_surface_statics, surface_statics_metadata,
        synthetic_provenance, synthetic_surface_statics,
    )

    xp = transform.backend.xp
    b = transform.backend
    lat = b.asarray(transform.grid.lat_rad[:, None], dtype=b.float_dtype)
    lon = b.asarray(transform.grid.lon_rad[None, :], dtype=b.float_dtype)
    coslat = xp.cos(lat)
    wave = xp.cos(float(cfg.zonal_wavenumber) * lon)
    perturb = float(cfg.perturbation_amplitude) * wave * coslat ** 2
    ps = float(cfg.surface_pressure_pa) * (1.0 + perturb)

    pressure = cfg.vertical.pressure(ps, b)
    normalized = xp.log(pressure["p_full"] / pressure["p_full"][0:1]) / xp.log(
        pressure["p_full"][-1:] / pressure["p_full"][0:1]
    )
    temperature = float(cfg.top_temperature_k) + normalized * (
        float(cfg.surface_temperature_k) - float(cfg.top_temperature_k)
    )
    temperature += 0.5 * perturb[None] * xp.sin(math.pi * normalized)
    exner = (pressure["p_full"] / REFERENCE_PRESSURE_PA) ** KAPPA
    theta = temperature / exner

    sigma_like = pressure["p_full"] / ps[None]
    qv = float(cfg.qv_surface) * sigma_like ** 3 * (0.75 + 0.25 * coslat[None] ** 2)
    cloud_shape = xp.maximum(0.0, xp.cos(2.0 * lat)[None] * wave[None])
    qc = 1.0e-5 * cloud_shape * xp.exp(-((sigma_like - 0.65) / 0.15) ** 2)
    zeros = xp.zeros_like(qv)

    u = float(cfg.zonal_wind_m_s) * coslat[None] * (
        0.5 + 0.5 * xp.sin(math.pi * normalized)
    )
    v_pattern = (
        0.1 * float(cfg.zonal_wind_m_s)
        * xp.sin(float(cfg.zonal_wavenumber) * lon)[None]
        * coslat[None]
    )
    v = xp.broadcast_to(v_pattern, u.shape).copy()
    vector = VorticityDivergenceOperator(transform)
    vorticity, divergence = vector.vordiv_from_wind(u, v)

    # The condensate species and the number moments are grid tracers:
    # planted on the Gaussian grid exactly as written, nonnegative, with
    # no spectral round trip.  Cloud droplet number is seeded only where
    # the analytic cloud exists; the remaining categories begin at exact
    # zero and are conservative tracers until physics writes them.
    dtype = b.float_dtype
    atmosphere = MoistHybridState(
        vorticity=vorticity,
        divergence=divergence,
        theta=transform.forward(theta),
        log_surface_pressure=transform.forward(xp.log(ps)),
        qv=transform.forward(qv),
        qc=xp.asarray(qc, dtype=dtype),
        qr=zeros.astype(dtype),
        qi=zeros.astype(dtype),
        qs=zeros.astype(dtype),
        qg=zeros.astype(dtype),
        nc=xp.asarray(xp.where(qc > 0.0, 1.0e8, 0.0), dtype=dtype),
        nr=zeros.astype(dtype),
        ni=zeros.astype(dtype),
        ns=zeros.astype(dtype),
        ng=zeros.astype(dtype),
    )

    mountain = float(cfg.terrain_amplitude_m) * (
        xp.maximum(0.0, xp.cos(lat)) ** 4
        * xp.maximum(0.0, xp.cos(lon - math.pi)) ** 6
    )
    surface_geopotential = GRAVITY_M_S2 * mountain
    land_fraction = xp.clip(0.55 + 0.35 * xp.cos(2.0 * lat) + 0.10 * xp.cos(3.0 * lon), 0.0, 1.0)
    surface_temperature = float(cfg.surface_temperature_k) - 25.0 * xp.sin(lat) ** 2 - 0.0065 * mountain
    nsoil = 4
    soil_temperature = xp.empty((nsoil, *transform.grid.shape), dtype=b.float_dtype)
    soil_water = xp.empty_like(soil_temperature)
    for k in range(nsoil):
        soil_temperature[k] = surface_temperature - 0.5 * k
        soil_water[k] = 0.25 + 0.05 * land_fraction

    if cfg.statics.source == "real":
        if statics is None:
            statics = load_statics(cfg.statics, transform.grid)
        fields, cache_provenance = statics
        resolved, detail = resolve_surface_statics(
            fields, cache_provenance, valid_time=cfg.statics.valid_time_utc,
            latitude_deg=transform.grid.latitude_deg,
            terrain_height_m=b.to_numpy(mountain),
            soil_temperature_k=b.to_numpy(soil_temperature),
            skin_temperature_k=b.to_numpy(surface_temperature),
        )
        land_fraction = b.asarray(resolved.pop("land_fraction"), dtype=b.float_dtype)
        soil_water = 0.25 + 0.05 * land_fraction + 0.0 * soil_water
        statics_provenance = real_provenance(
            cfg.statics, cache_paths(cfg.statics, transform.grid)[0],
            cache_provenance, detail,
        )
        statics_provenance["land_fraction_source"] = "static-water-fraction"
        statics_metadata = surface_statics_metadata(
            "real", real_convention(cache_provenance)
        )
    else:
        resolved = synthetic_surface_statics(
            b.to_numpy(land_fraction), b.to_numpy(soil_temperature)
        )
        # The formula mask, held on the side of one half its water
        # columns' categories name (statics.consistent_land_fraction).
        land_fraction = b.asarray(resolved.pop("land_fraction"), dtype=b.float_dtype)
        statics_provenance = synthetic_provenance(cfg.statics)
        statics_provenance["land_fraction_source"] = "analytic-formula"
        statics_metadata = surface_statics_metadata("synthetic", SYNTHETIC_CONVENTION)

    surface = SurfaceState(
        temperature_k=surface_temperature.astype(b.float_dtype),
        water_kg_m2=xp.full(transform.grid.shape, float(cfg.surface_water_kg_m2), dtype=b.float_dtype),
        land_fraction=land_fraction.astype(b.float_dtype),
        # Same land/ocean split as the analysis path: a 2e7 land capacity
        # is five metres of water equivalent and makes deserts thermally
        # oceanic (the f021 Sonoran-cyclone defect, 2026-08-31).
        heat_capacity_j_m2_k=(
            4.0e7 * (1.0 - land_fraction) + 2.0e5 * land_fraction
        ).astype(b.float_dtype),
        soil_temperature_k=soil_temperature,
        soil_water_fraction=soil_water.astype(b.float_dtype),
        accumulated_rain_kg_m2=xp.zeros(transform.grid.shape, dtype=b.float_dtype),
        accumulated_snow_kg_m2=xp.zeros(transform.grid.shape, dtype=b.float_dtype),
        accumulated_graupel_kg_m2=xp.zeros(transform.grid.shape, dtype=b.float_dtype),
        **{name: b.asarray(value, dtype=b.float_dtype)
           for name, value in resolved.items()},
    )
    provenance = {"mode": "analytic", "statics": statics_provenance}
    return (
        ArwenGlobalState(atmosphere, surface, PhysicsState(metadata=statics_metadata)),
        surface_geopotential,
        provenance,
    )
