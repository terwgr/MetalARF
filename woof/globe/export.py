"""Hash-bound regular-lat/lon export for global-to-regional experiments."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from woof.globe.spectral.sampling import (
    regular_latlon_coordinates,
    sample_scalar,
    sample_wind,
)

from .checkpoint import read_checkpoint, state_from_checkpoint
from .constants import (
    DRY_AIR_GAS_CONSTANT,
    EXPORT_SCHEMA,
    GRAVITY_M_S2,
    GRID_TRACERS,
    KAPPA,
    NUMBER_MOMENTS,
    REFERENCE_PRESSURE_PA,
    WATER_SPECIES,
)
from .pins import ACCEPTED_PINS_HASHES, pins_hash
from .runner import build_model_and_cold_state, build_transform
from .transport import sample_grid_field


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def _sample_grid_field(transform, value, lat2, lon2):
    return sample_scalar(transform, transform.forward(value), lat2, lon2)


def export_parent(
    cfg,
    checkpoint: str | Path,
    output: str | Path,
    *,
    nlat: int,
    nlon: int,
) -> Path:
    """Export complete moist/moment state plus diagnosed continuity carriers."""
    transform = build_transform(cfg)
    model, _cold = build_model_and_cold_state(
        cfg, transform, scratch_destination=Path(output))
    metadata, arrays = read_checkpoint(
        checkpoint,
        expected_config_hash=cfg.config_hash,
        semi_implicit_scheme=cfg.semi_implicit_scheme,
        integrator=cfg.integrator,
    )
    bundle = state_from_checkpoint(metadata, arrays, transform.backend)
    model.enforce(bundle)
    lat, lon = regular_latlon_coordinates(nlat, nlon, include_poles=False)
    lon2, lat2 = np.meshgrid(lon, lat)

    atmosphere = bundle.atmosphere
    theta = sample_scalar(transform, atmosphere.theta, lat2, lon2)
    logps = sample_scalar(
        transform, atmosphere.log_surface_pressure, lat2, lon2
    )
    ps = np.exp(logps)
    vorticity = sample_scalar(transform, atmosphere.vorticity, lat2, lon2)
    divergence = sample_scalar(transform, atmosphere.divergence, lat2, lon2)
    u, v = sample_wind(
        transform, atmosphere.vorticity, atmosphere.divergence, lat2, lon2
    )
    # Vapor is sampled through the spherical basis like every spectral
    # field; the grid tracers are sampled bilinearly on the Gaussian grid
    # (convex weights: a nonnegative field samples nonnegative, so the
    # condensate a regional child receives never rings the way a spectral
    # synthesis of the same field did).
    tracers = {
        "qv": sample_scalar(transform, atmosphere.qv, lat2, lon2),
        **{
            name: sample_grid_field(
                transform.grid,
                transform.backend.to_numpy(getattr(atmosphere, name)),
                lat2, lon2,
            )
            for name in GRID_TRACERS
        },
    }
    p_half = (
        np.asarray(cfg.a_half_pa)[:, None, None]
        + np.asarray(cfg.b_half)[:, None, None] * ps[None]
    )
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    dp = p_half[1:] - p_half[:-1]
    temperature = theta * (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
    virtual_temperature = temperature * (
        1.0 + 0.61 * tracers["qv"]
        - sum(tracers[name] for name in ("qc", "qr", "qi", "qs", "qg"))
    )

    terrain_coeff = transform.forward(model.surface_geopotential)
    surface_geopotential = sample_scalar(transform, terrain_coeff, lat2, lon2)
    geopotential = np.empty_like(temperature)
    lower = surface_geopotential
    for k in range(model.nlev - 1, -1, -1):
        lower_p = p_half[k + 1]
        upper_p = p_half[k]
        full_p = p_full[k]
        geopotential[k] = lower + model.gas_constant * virtual_temperature[k] * np.log(
            lower_p / full_p
        )
        if k:
            lower = lower + model.gas_constant * virtual_temperature[k] * np.log(
                lower_p / upper_p
            )

    # Diagnose the parent-time continuity carriers on the model's native
    # Gaussian grid, then sample them through the same spherical basis.
    native = model.grid_state(bundle.atmosphere)
    _, div_mass = model._mass_flux_divergence(
        native["dp"], native["u"], native["v"]
    )
    xp = transform.backend.xp
    ps_t_native = -xp.sum(div_mass, axis=0)
    omega_half_native, closure_native = model.vertical.continuity(
        div_mass, ps_t_native, transform.backend
    )
    omega_full_native = 0.5 * (
        omega_half_native[:-1] + omega_half_native[1:]
    )
    rho_full_native = native["p_full"] / (
        DRY_AIR_GAS_CONSTANT * native["virtual_temperature"]
    )
    rho_half_native = xp.empty_like(omega_half_native)
    rho_half_native[0] = rho_full_native[0]
    rho_half_native[-1] = rho_full_native[-1]
    rho_half_native[1:-1] = 0.5 * (
        rho_full_native[:-1] + rho_full_native[1:]
    )
    w_half_native = -omega_half_native / xp.maximum(
        rho_half_native * GRAVITY_M_S2, 1.0e-12
    )
    w_full_native = -omega_full_native / xp.maximum(
        rho_full_native * GRAVITY_M_S2, 1.0e-12
    )
    pressure_tendency_native = (
        transform.backend.asarray(model.vertical.delta_b, dtype=transform.backend.float_dtype)
        [:, None, None]
        * ps_t_native[None]
    )
    total_water_native = sum(native[name] for name in WATER_SPECIES)
    dry_column_native = xp.sum(
        native["dp"] / xp.maximum(1.0 + total_water_native, 1.0e-12), axis=0
    ) / GRAVITY_M_S2

    sampled: dict[str, np.ndarray] = {
        "latitude_deg": lat,
        "longitude_deg": lon,
        "a_half_pa": np.asarray(cfg.a_half_pa, dtype=np.float64),
        "b_half": np.asarray(cfg.b_half, dtype=np.float64),
        "surface_pressure_pa": ps,
        "p_half_pa": p_half,
        "p_full_pa": p_full,
        "dp_pa": dp,
        "potential_temperature_k": theta,
        "temperature_k": temperature,
        "virtual_temperature_k": virtual_temperature,
        "geopotential_m2_s2": geopotential,
        "height_m": geopotential / GRAVITY_M_S2,
        "surface_geopotential_m2_s2": surface_geopotential,
        "eastward_wind_m_s": u,
        "northward_wind_m_s": v,
        "relative_vorticity_s1": vorticity,
        "divergence_s1": divergence,
        "surface_pressure_tendency_pa_s": _sample_grid_field(
            transform, ps_t_native, lat2, lon2
        ),
        "pressure_tendency_pa_s": _sample_grid_field(
            transform, pressure_tendency_native, lat2, lon2
        ),
        "omega_half_pa_s": _sample_grid_field(
            transform, omega_half_native, lat2, lon2
        ),
        "omega_full_pa_s": _sample_grid_field(
            transform, omega_full_native, lat2, lon2
        ),
        "vertical_velocity_half_m_s": _sample_grid_field(
            transform, w_half_native, lat2, lon2
        ),
        "vertical_velocity_full_m_s": _sample_grid_field(
            transform, w_full_native, lat2, lon2
        ),
        "continuity_residual_pa_s": _sample_grid_field(
            transform, closure_native, lat2, lon2
        ),
        "dry_column_mass_kg_m2": _sample_grid_field(
            transform, dry_column_native, lat2, lon2
        ),
        **{
            (f"{name}_kg_kg" if name in WATER_SPECIES else f"{name}_kg1"):
            value
            for name, value in tracers.items()
        },
    }
    for name, value in bundle.surface.arrays().items():
        sampled[name] = _sample_grid_field(transform, value, lat2, lon2)

    export_metadata: dict[str, object] = {
        "schema": EXPORT_SCHEMA,
        "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
        "config_hash": cfg.config_hash,
        "source_checkpoint": str(Path(checkpoint)),
        "source_checkpoint_self_sha256": metadata["self_sha256"],
        "step": int(metadata["step"]),
        "time_s": float(metadata["time_s"]),
        "target_grid": {
            "kind": "regular-latlon-cell-centres",
            "nlat": int(nlat),
            "nlon": int(nlon),
            "includes_poles": False,
        },
        "continuity": {
            "pressure_velocity": "hybrid-layer mass continuity; top/bottom pinned zero",
            "vertical_velocity": "hydrostatic -omega/(rho*g) approximation",
            "dry_column_mass": "sum(dp/(1+total-water))/g",
        },
        "arrays": {
            name: {
                "shape": list(np.asarray(value).shape),
                "dtype": np.asarray(value).dtype.str,
                "sha256": _array_hash(np.asarray(value)),
            }
            for name, value in sampled.items()
        },
        "admission": (
            "neutral research parent export; this is not a WOOF regional "
            "initial-condition or LBC file. Translation through a hash-bound "
            "regional target is required before WOOF initialization or LBC use"
        ),
    }
    export_metadata["self_sha256"] = hashlib.sha256(
        _canonical(export_metadata)
    ).hexdigest()

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            __metadata__=np.asarray(
                json.dumps(export_metadata, sort_keys=True, allow_nan=False)
            ),
            **sampled,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return target


def read_parent_export(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "__metadata__" not in archive:
            raise ValueError(f"parent export {source} has no metadata")
        metadata = json.loads(str(archive["__metadata__"].item()))
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "__metadata__"
        }
    if metadata.get("schema") != EXPORT_SCHEMA:
        raise ValueError("parent export schema mismatch")
    if metadata.get("pins_hash") not in ACCEPTED_PINS_HASHES:
        raise ValueError("parent export arithmetic pins mismatch")
    self_hash = metadata.pop("self_sha256", None)
    if self_hash != hashlib.sha256(_canonical(metadata)).hexdigest():
        raise ValueError("parent export metadata self-hash mismatch")
    metadata["self_sha256"] = self_hash
    expected = metadata.get("arrays")
    if not isinstance(expected, dict) or set(expected) != set(arrays):
        raise ValueError("parent export array inventory mismatch")
    for name, array in arrays.items():
        row = expected[name]
        if list(array.shape) != row["shape"] or array.dtype.str != row["dtype"]:
            raise ValueError(f"parent export array {name} shape/dtype mismatch")
        if _array_hash(array) != row["sha256"]:
            raise ValueError(f"parent export array {name} hash mismatch")
        if not np.isfinite(array).all():
            raise ValueError(f"parent export array {name} contains non-finite values")
    required = {
        "latitude_deg", "longitude_deg", "surface_pressure_pa",
        "p_half_pa", "p_full_pa", "potential_temperature_k",
        "eastward_wind_m_s", "northward_wind_m_s", "omega_half_pa_s",
        *{f"{name}_kg_kg" for name in WATER_SPECIES},
        *{f"{name}_kg1" for name in NUMBER_MOMENTS},
    }
    if not required <= set(arrays):
        raise ValueError("parent export Level-5 field inventory is incomplete")
    return metadata, arrays


__all__ = ["export_parent", "read_parent_export"]
