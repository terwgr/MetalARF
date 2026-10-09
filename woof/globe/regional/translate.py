"""Translate one Level-5 parent export into an Arwen regional state frame."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..constants import (
    DRY_AIR_GAS_CONSTANT,
    GRAVITY_M_S2,
    NUMBER_MOMENTS,
    WATER_SPECIES,
)
from ..export import read_parent_export
from .artifact import (
    file_hash,
    read_regional_target,
    write_regional_frame,
)
from .interpolation import (
    STANDARD_LAPSE_RATE_K_M,
    extrapolation_fractions,
    log_pressure_interpolate,
    periodic_bilinear,
    rotate_earth_to_grid,
    standard_lapse_theta_below,
    stagger_u_nonperiodic,
    stagger_v_nonperiodic,
)

_THETA_OFFSET_K = 300.0
_REFERENCE_PRESSURE_PA = 100_000.0
_KAPPA = 287.0 / 1004.0
#: Largest |mu'| the base/perturbation split admits.  mu' is the departure of
#: the translated dry column from the target's own base column at the same
#: terrain, so it is a surface-pressure anomaly: the observed global extremes
#: referred to the 101.325 kPa standard atmosphere are 87.0 kPa and 108.3 kPa,
#: giving 14.3 kPa, plus at most 0.4 kPa for the total-water removal of a
#: 40 kg/m2 column.  Beyond that mub2d is not this column's base state and
#: php carries hundreds of metres of spurious base-state offset.
_MAXIMUM_MU_PERTURBATION_PA = 15_000.0


def _expand_vertical_base(value, nlev, ny, nx, name):
    array = np.asarray(value, np.float64)
    if array.shape == (nlev,):
        return np.broadcast_to(array[:, None, None], (nlev, ny, nx)).copy()
    if array.shape == (nlev, ny, nx):
        return array.copy()
    raise ValueError(f"target {name} has incompatible shape {array.shape}")


def _horizontal(parent, name, target_lat, target_lon):
    return periodic_bilinear(
        parent["latitude_deg"], parent["longitude_deg"], parent[name],
        target_lat, target_lon,
    )


def _vertical(parent_pressure, parent_value, target_pressure, bottom_values=None):
    return log_pressure_interpolate(
        parent_pressure, parent_value, target_pressure,
        bottom_values=bottom_values,
    )


def _hydrostatic_half(virtual_temperature, p_half, terrain_geopotential):
    nlev = virtual_temperature.shape[0]
    half = np.empty((nlev + 1, *terrain_geopotential.shape), np.float64)
    half[-1] = terrain_geopotential
    for k in range(nlev - 1, -1, -1):
        half[k] = half[k + 1] + DRY_AIR_GAS_CONSTANT * virtual_temperature[k] * np.log(
            p_half[k + 1] / p_half[k]
        )
    full = np.empty_like(virtual_temperature)
    for k in range(nlev):
        p_full = np.sqrt(p_half[k] * p_half[k + 1])
        full[k] = half[k + 1] + DRY_AIR_GAS_CONSTANT * virtual_temperature[k] * np.log(
            p_half[k + 1] / p_full
        )
    return half, full


def _mu_at_u(mu):
    out = np.empty((mu.shape[0], mu.shape[1] + 1), np.float64)
    out[:, 1:-1] = 0.5 * (mu[:, :-1] + mu[:, 1:])
    out[:, 0] = mu[:, 0]
    out[:, -1] = mu[:, -1]
    return out


def _mu_at_v(mu):
    out = np.empty((mu.shape[0] + 1, mu.shape[1]), np.float64)
    out[1:-1] = 0.5 * (mu[:-1] + mu[1:])
    out[0] = mu[0]
    out[-1] = mu[-1]
    return out


def translate_parent_to_regional_frame(
    parent_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
) -> Path:
    parent_meta, parent = read_parent_export(parent_path)
    target_meta, target = read_regional_target(target_path)
    target_lat = target["latitude_deg"]
    target_lon = target["longitude_deg"]
    ny, nx = target_lat.shape
    a = target["a_half_pa"]
    b = target["b_half"]
    nlev = a.size - 1

    parent_ps = _horizontal(parent, "surface_pressure_pa", target_lat, target_lon)
    parent_phi_surface = _horizontal(
        parent, "surface_geopotential_m2_s2", target_lat, target_lon
    )
    parent_tv_surface = _horizontal(
        parent, "virtual_temperature_k", target_lat, target_lon
    )[-1]
    target_phi_surface = GRAVITY_M_S2 * target["terrain_height_m"]
    ps = parent_ps * np.exp(
        -(target_phi_surface - parent_phi_surface)
        / np.maximum(DRY_AIR_GAS_CONSTANT * parent_tv_surface, 1.0)
    )
    if np.any(ps < 30_000.0) or np.any(ps > 120_000.0):
        raise ValueError("terrain-adjusted regional surface pressure is outside bounds")
    p_half = a[:, None, None] + b[:, None, None] * ps[None]
    if np.any(np.diff(p_half, axis=0) <= 0.0):
        raise ValueError("target hybrid pressure is not monotonic")
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    dp = p_half[1:] - p_half[:-1]

    parent_p_full = _horizontal(parent, "p_full_pa", target_lat, target_lon)
    parent_theta = _horizontal(
        parent, "potential_temperature_k", target_lat, target_lon
    )
    theta = _vertical(
        parent_p_full,
        parent_theta,
        p_full,
        standard_lapse_theta_below(
            parent_p_full, parent_theta, p_full,
            reference_pressure_pa=_REFERENCE_PRESSURE_PA,
            kappa=_KAPPA,
            gas_constant=DRY_AIR_GAS_CONSTANT,
            gravity=GRAVITY_M_S2,
        ),
    )
    extrapolated = extrapolation_fractions(parent_p_full, p_full)
    tracers = {}
    for name in WATER_SPECIES:
        tracers[name] = np.maximum(
            _vertical(
                parent_p_full,
                _horizontal(parent, f"{name}_kg_kg", target_lat, target_lon),
                p_full,
            ),
            0.0,
        )
    for name in NUMBER_MOMENTS:
        tracers[name] = np.maximum(
            _vertical(
                parent_p_full,
                _horizontal(parent, f"{name}_kg1", target_lat, target_lon),
                p_full,
            ),
            0.0,
        )
    east = _vertical(
        parent_p_full,
        _horizontal(parent, "eastward_wind_m_s", target_lat, target_lon),
        p_full,
    )
    north = _vertical(
        parent_p_full,
        _horizontal(parent, "northward_wind_m_s", target_lat, target_lon),
        p_full,
    )
    grid_u_mass, grid_v_mass = rotate_earth_to_grid(
        east, north, target["cosa"][None], target["sina"][None]
    )
    w_full = _vertical(
        parent_p_full,
        _horizontal(parent, "vertical_velocity_full_m_s", target_lat, target_lon),
        p_full,
    )
    w_half = np.zeros((nlev + 1, ny, nx), np.float64)
    if nlev > 1:
        w_half[1:-1] = 0.5 * (w_full[:-1] + w_full[1:])

    temperature = theta * (p_full / _REFERENCE_PRESSURE_PA) ** _KAPPA
    condensate = sum(tracers[name] for name in ("qc", "qr", "qi", "qs", "qg"))
    virtual_temperature = temperature * (1.0 + 0.61 * tracers["qv"] - condensate)
    phi_half, phi_full = _hydrostatic_half(
        virtual_temperature, p_half, target_phi_surface
    )
    density = p_full / np.maximum(DRY_AIR_GAS_CONSTANT * virtual_temperature, 1.0)
    total_water = sum(tracers[name] for name in WATER_SPECIES)
    # ``density`` is the total (moist) density p/(Rd*Tv); Arwen's state.alt is
    # the DRY specific volume alpha_d, and rho_moist = rho_dry*(1 + q_total).
    dry_inverse_density = (1.0 + total_water) / density
    dry_mu = np.sum(dp / np.maximum(1.0 + total_water, 1.0e-12), axis=0)

    # Regional Arwen state is surface-to-top.
    theta_r = theta[::-1]
    temperature_r = temperature[::-1]
    pressure_r = p_full[::-1]
    p_half_r = p_half[::-1]
    phi_full_r = phi_full[::-1]
    phi_half_r = phi_half[::-1]
    density_r = density[::-1]
    dry_inverse_density_r = dry_inverse_density[::-1]
    u_mass_r = grid_u_mass[::-1]
    v_mass_r = grid_v_mass[::-1]
    u = stagger_u_nonperiodic(u_mass_r)
    v = stagger_v_nonperiodic(v_mass_r)
    w = w_half[::-1]
    tracer_r = {name: value[::-1] for name, value in tracers.items()}

    thb = _expand_vertical_base(target["thb"], nlev, ny, nx, "thb")
    phb = _expand_vertical_base(target["phb"], nlev + 1, ny, nx, "phb")
    mup = dry_mu - target["mub2d"]
    if np.max(np.abs(mup)) > _MAXIMUM_MU_PERTURBATION_PA:
        raise ValueError(
            "regional mu' exceeds the admissible column-mass anomaly "
            f"({float(np.max(np.abs(mup))):.1f} Pa > "
            f"{_MAXIMUM_MU_PERTURBATION_PA:.1f} Pa): the target's mub2d is not "
            "the base column mass of this coordinate at this terrain"
        )
    thp = theta_r - thb
    php = phi_half_r - phb

    c1h = target["c1h"][:, None, None]
    c2h = target["c2h"][:, None, None]
    c1f = target["c1f"][:, None, None]
    c2f = target["c2f"][:, None, None]
    mux = _mu_at_u(dry_mu)
    muy = _mu_at_v(dry_mu)
    chm = c1h * dry_mu[None] + c2h
    chf = c1f * dry_mu[None] + c2f
    coupled_u = (c1h * mux[None] + c2h) * u
    coupled_v = (c1h * muy[None] + c2h) * v
    if not np.all(target["msfu"] == 1.0):
        coupled_u /= target["msfu"][None]
    if not np.all(target["msfv"] == 1.0):
        coupled_v /= target["msfv"][None]
    coupled = {
        "coupled__u": coupled_u,
        "coupled__v": coupled_v,
        "coupled__theta": chm * (theta_r - _THETA_OFFSET_K),
        "coupled__phi": chf * php,
        "coupled__mu": mup[None],
        "coupled__qv": chm * tracer_r["qv"],
    }

    arrays = {
        "u": u, "v": v, "w": w, "theta": theta_r,
        "temperature": temperature_r, "pressure": pressure_r,
        "p_half": p_half_r, "geopotential": phi_full_r,
        "geopotential_half": phi_half_r, "density": density_r,
        "dry_inverse_density": dry_inverse_density_r,
        "surface_pressure": ps, "dry_mu": dry_mu,
        "mup": mup, "thp": thp, "php": php,
        **{name: value for name, value in tracer_r.items()},
        **coupled,
    }
    metadata = {
        "source_parent_path": str(parent_path),
        "source_parent_file_sha256": file_hash(parent_path),
        "source_parent_self_sha256": parent_meta["self_sha256"],
        "target_path": str(target_path),
        "target_file_sha256": file_hash(target_path),
        "target_self_sha256": target_meta["self_sha256"],
        "target_grid_id": target_meta["grid_id"],
        "time_s": float(parent_meta["time_s"]),
        "step": int(parent_meta["step"]),
        "methods": {
            "horizontal": "periodic-regular-latlon-bilinear-v1",
            "vertical": (
                "independent-column-log-pressure-theta-standard-lapse-below-"
                "parent-bottom-others-held-v1"
            ),
            "surface_pressure": "single-layer-hypsometric-virtual-temperature-v1",
            "wind": (
                "earth-to-grid-rotation-then-nonperiodic-C-grid-average-with-"
                "zeroth-order-outer-faces-v1"
            ),
            "geopotential": "target-terrain-hydrostatic-reintegration-v1",
            "vertical_velocity": "interpolated-parent-w-with-zero-boundary-flux-v1",
            "dry_mass": "sum-dp-over-one-plus-total-water-v1",
            "coupling": "existing-WOOF-WRF-u-v-theta-phi-mu-qv-units-v1",
        },
        "extrapolated_fraction": extrapolated,
        "standard_lapse_rate_k_m": STANDARD_LAPSE_RATE_K_M,
        "admission": "experimental one-way parent frame; regional A/B required",
    }
    return write_regional_frame(output_path, metadata, arrays)


__all__ = ["translate_parent_to_regional_frame"]
