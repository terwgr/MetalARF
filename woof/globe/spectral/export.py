"""Hash-bound regular-lat/lon export for regional-parent experiments."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .checkpoint import read_checkpoint, state_from_checkpoint
from .config import GlobalSpectralRunConfig
from .constants import GRAVITY_M_S2
from .pins import ACCEPTED_PINS_HASHES, PINS_HASH
from .sampling import regular_latlon_coordinates, sample_scalar, sample_wind
from .transform import SphericalHarmonicTransform

EXPORT_SCHEMA = "gpuwm.global-spectral-latlon-export/v1"


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


def _hydrostatic_geopotential(temperature: np.ndarray, sigma_half: tuple[float, ...]):
    from .constants import DRY_AIR_GAS_CONSTANT

    half = np.asarray(sigma_half, dtype=np.float64)
    full = 0.5 * (half[:-1] + half[1:])
    result = np.empty_like(temperature, dtype=np.float64)
    lower = np.zeros(temperature.shape[1:], dtype=np.float64)
    for k in range(full.size - 1, -1, -1):
        result[k] = lower + DRY_AIR_GAS_CONSTANT * temperature[k] * math.log(
            half[k + 1] / full[k]
        )
        if k:
            lower = lower + DRY_AIR_GAS_CONSTANT * temperature[k] * math.log(
                half[k + 1] / half[k]
            )
    return result


def export_checkpoint_latlon(
    cfg: GlobalSpectralRunConfig,
    transform: SphericalHarmonicTransform,
    checkpoint: str | Path,
    output: str | Path,
    *,
    nlat: int,
    nlon: int,
) -> Path:
    """Export one checkpoint onto a cell-centred regular global grid.

    The export is intentionally a neutral parent handoff, not a claim that
    Arwen can already consume it as lateral boundaries.  All sampled state,
    coordinates, identities, and hashes are present so a later interpolation
    adapter can be proven against a stable source artifact.
    """
    metadata, arrays = read_checkpoint(
        checkpoint, expected_config_hash=cfg.config_hash
    )
    state = state_from_checkpoint(metadata, arrays, transform.backend)
    lat, lon = regular_latlon_coordinates(nlat, nlon, include_poles=False)
    lon2, lat2 = np.meshgrid(lon, lat)

    sampled: dict[str, np.ndarray] = {
        "latitude_deg": lat,
        "longitude_deg": lon,
    }
    if cfg.model == "shallow-water":
        geopotential = sample_scalar(
            transform, state.geopotential, lat2, lon2
        )
        vorticity = sample_scalar(transform, state.vorticity, lat2, lon2)
        divergence = sample_scalar(transform, state.divergence, lat2, lon2)
        u, v = sample_wind(
            transform, state.vorticity, state.divergence, lat2, lon2
        )
        sampled.update(
            geopotential_m2_s2=geopotential,
            depth_m=geopotential / GRAVITY_M_S2,
            relative_vorticity_s1=vorticity,
            divergence_s1=divergence,
            eastward_wind_m_s=u,
            northward_wind_m_s=v,
        )
    else:
        temperature = sample_scalar(
            transform, state.temperature, lat2, lon2
        )
        logps = sample_scalar(
            transform, state.log_surface_pressure, lat2, lon2
        )
        surface_pressure = np.exp(logps)
        vorticity = sample_scalar(transform, state.vorticity, lat2, lon2)
        divergence = sample_scalar(transform, state.divergence, lat2, lon2)
        u, v = sample_wind(
            transform, state.vorticity, state.divergence, lat2, lon2
        )
        sigma_half = np.asarray(cfg.sigma_half, dtype=np.float64)
        sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
        pressure = sigma_full[:, None, None] * surface_pressure[None]
        geopotential = _hydrostatic_geopotential(temperature, cfg.sigma_half)
        sampled.update(
            sigma_half=sigma_half,
            sigma_full=sigma_full,
            temperature_k=temperature,
            surface_pressure_pa=surface_pressure,
            pressure_pa=pressure,
            geopotential_m2_s2=geopotential,
            relative_vorticity_s1=vorticity,
            divergence_s1=divergence,
            eastward_wind_m_s=u,
            northward_wind_m_s=v,
        )

    export_metadata: dict[str, object] = {
        "schema": EXPORT_SCHEMA,
        "pins_hash": PINS_HASH,
        "config_hash": cfg.config_hash,
        "model": cfg.model,
        "source_checkpoint": str(Path(checkpoint)),
        "source_checkpoint_self_sha256": metadata["self_sha256"],
        "step": int(metadata["step"]),
        "time_s": float(metadata["time_s"]),
        "spectral_geometry": transform.geometry_identity,
        "spectral_geometry_hash": transform.geometry_hash,
        "target_grid": {
            "kind": "regular-latlon-cell-centres",
            "nlat": int(nlat),
            "nlon": int(nlon),
            "includes_poles": False,
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
            "neutral research export; no claim of a completed WOOF lateral-"
            "boundary or initial-condition adapter"
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


def read_latlon_export(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "__metadata__" not in archive:
            raise ValueError(f"lat/lon export {source} has no metadata")
        metadata = json.loads(str(archive["__metadata__"].item()))
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "__metadata__"
        }
    if metadata.get("schema") != EXPORT_SCHEMA:
        raise ValueError("lat/lon export schema mismatch")
    if metadata.get("pins_hash") not in ACCEPTED_PINS_HASHES:
        raise ValueError("lat/lon export arithmetic pins mismatch")
    self_hash = metadata.pop("self_sha256", None)
    if hashlib.sha256(_canonical(metadata)).hexdigest() != self_hash:
        raise ValueError("lat/lon export metadata self-hash mismatch")
    metadata["self_sha256"] = self_hash
    expected = metadata.get("arrays", {})
    if set(expected) != set(arrays):
        raise ValueError("lat/lon export array inventory mismatch")
    for name, array in arrays.items():
        row = expected[name]
        if list(array.shape) != row["shape"] or array.dtype.str != row["dtype"]:
            raise ValueError(f"lat/lon export array {name} shape/dtype mismatch")
        if _array_hash(array) != row["sha256"]:
            raise ValueError(f"lat/lon export array {name} hash mismatch")
    return metadata, arrays


__all__ = ["EXPORT_SCHEMA", "export_checkpoint_latlon", "read_latlon_export"]
