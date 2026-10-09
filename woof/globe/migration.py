"""Explicit migration of recovered Level-4 checkpoints into Level 5.

The new number moments and adapter-owned state did not exist in Level 4.  They
are therefore never inferred during ordinary checkpoint loading.  Migration is
an explicit, hash-bound operation that seeds the five moments to exact zero and
records that scientific choice in both checkpoint metadata and a separate
receipt.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .checkpoint import TRACKER_KEYS, read_checkpoint, write_checkpoint
from .constants import (
    CONDENSATE_SPECIES,
    LEVEL4_CHECKPOINT_SCHEMA,
    LEVEL4_MIGRATION_SCHEMA,
    LEVEL4_SPECTRAL_FIELDS,
    NUMBER_MOMENTS,
    PHYSICS_STATE_SCHEMA,
    SPECTRAL_FIELDS,
)
from .pins import ACCEPTED_PINS_HASHES, DEFAULT_INTEGRATOR, pins_hash
from .state import ArwenGlobalState, MoistHybridState, PhysicsState, SurfaceState
from .water import SOIL_LAYER_THICKNESS_M

LEVEL4_PINS_HASH = "0f4f5bb173813ca196b904de70ab3a21b82f0c28ead67e005fcc29a687b043d5"
LEVEL4_TRACKER_KEYS = (
    "maximum_spectral_cfl",
    "maximum_mass_fixer_log_offset",
    "maximum_global_water_fixer_kg_m2",
    "maximum_repaired_negative_mixing_ratio",
    "maximum_semi_implicit_divergence_increment_s1",
    "maximum_physics_water_repair_kg_m2",
)
_SURFACE_MAP = {
    "temperature_k": "surface_temperature_k",
    "water_kg_m2": "surface_water_kg_m2",
    "land_fraction": "land_fraction",
    "albedo": "surface_albedo",
    "emissivity": "surface_emissivity",
    "roughness_m": "surface_roughness_m",
    "heat_capacity_j_m2_k": "surface_heat_capacity_j_m2_k",
    "soil_temperature_k": "soil_temperature_k",
    "soil_water_fraction": "soil_water_fraction",
    "accumulated_rain_kg_m2": "accumulated_rain_kg_m2",
    "accumulated_snow_kg_m2": "accumulated_snow_kg_m2",
    "accumulated_graupel_kg_m2": "accumulated_graupel_kg_m2",
}


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


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def read_level4_checkpoint(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "__metadata__" not in archive:
            raise ValueError(f"Level-4 checkpoint {source} has no metadata")
        metadata = json.loads(str(archive["__metadata__"].item()))
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "__metadata__"
        }
    required = {
        "schema", "config_hash", "pins_hash", "time_s", "step",
        "run_trackers", "arrays", "self_sha256",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("Level-4 checkpoint metadata inventory mismatch")
    if metadata["schema"] != LEVEL4_CHECKPOINT_SCHEMA:
        raise ValueError("input is not a WOOF global Level-4 checkpoint")
    if metadata["pins_hash"] != LEVEL4_PINS_HASH:
        raise ValueError("Level-4 checkpoint arithmetic pins mismatch")
    if not _is_sha(metadata["config_hash"]):
        raise ValueError("Level-4 checkpoint config hash is malformed")
    if (
        isinstance(metadata["step"], bool)
        or not isinstance(metadata["step"], int)
        or metadata["step"] < 0
    ):
        raise ValueError("Level-4 checkpoint step must be nonnegative")
    time_s = float(metadata["time_s"])
    if not math.isfinite(time_s) or time_s < 0.0:
        raise ValueError("Level-4 checkpoint time is invalid")
    trackers = metadata["run_trackers"]
    if not isinstance(trackers, dict) or set(trackers) != set(LEVEL4_TRACKER_KEYS):
        raise ValueError("Level-4 checkpoint tracker inventory mismatch")
    for name, raw in trackers.items():
        value = float(raw)
        if isinstance(raw, bool) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Level-4 tracker {name} is invalid")
        trackers[name] = value
    self_hash = metadata.pop("self_sha256")
    if self_hash != hashlib.sha256(_canonical(metadata)).hexdigest():
        raise ValueError("Level-4 checkpoint metadata self-hash mismatch")
    metadata["self_sha256"] = self_hash
    expected = metadata["arrays"]
    if not isinstance(expected, dict) or set(expected) != set(arrays):
        raise ValueError("Level-4 checkpoint array inventory mismatch")
    spectral = {f"atmosphere__{name}" for name in LEVEL4_SPECTRAL_FIELDS}
    surface = {f"surface__{name}" for name in _SURFACE_MAP.values()}
    if set(arrays) != spectral | surface:
        raise ValueError("Level-4 atmosphere/surface array inventory is incomplete")
    for name, array in arrays.items():
        row = expected[name]
        if (
            not isinstance(row, dict)
            or set(row) != {"shape", "dtype", "sha256"}
            or list(array.shape) != row["shape"]
            or array.dtype.str != row["dtype"]
            or _array_hash(array) != row["sha256"]
        ):
            raise ValueError(f"Level-4 checkpoint array {name} failed validation")
        if name.startswith("atmosphere__") and array.dtype.kind != "c":
            raise ValueError(f"Level-4 spectral array {name} must be complex")
        if not np.isfinite(array.real).all() or (
            np.iscomplexobj(array) and not np.isfinite(array.imag).all()
        ):
            raise ValueError(f"Level-4 checkpoint array {name} is non-finite")
    return metadata, arrays


def _write_migration_receipt(
    path: Path, payload: dict[str, object], *, semi_implicit_scheme: str,
    integrator: str = DEFAULT_INTEGRATOR,
) -> Path:
    row = dict(payload)
    row["schema"] = LEVEL4_MIGRATION_SCHEMA
    row["level5_pins_hash"] = pins_hash(semi_implicit_scheme, integrator)
    row["self_sha256"] = hashlib.sha256(_canonical(row)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def read_migration_receipt(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != LEVEL4_MIGRATION_SCHEMA:
        raise ValueError("migration receipt schema mismatch")
    if payload.get("level5_pins_hash") not in ACCEPTED_PINS_HASHES:
        raise ValueError("migration receipt Level-5 pins mismatch")
    self_hash = payload.pop("self_sha256", None)
    if self_hash != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError("migration receipt self-hash mismatch")
    payload["self_sha256"] = self_hash
    return payload


def _target_geometry(target_config) -> dict[str, object]:
    """Spectral and grid shapes the migrated checkpoint must already have."""
    from woof.globe.spectral.grid import GaussianGrid

    truncation = int(target_config.truncation)
    nlev = int(target_config.vertical.nlev)
    grid = GaussianGrid.create(
        truncation,
        nlat=target_config.nlat,
        nlon=target_config.nlon,
        dealias_factor=target_config.dealias_factor,
    )
    return {
        "truncation": truncation,
        "nlev": nlev,
        "spectral_shape": (nlev, truncation + 1, truncation + 1),
        "surface_shape": (int(grid.nlat), int(grid.nlon)),
        "soil_shape": (len(SOIL_LAYER_THICKNESS_M), int(grid.nlat), int(grid.nlon)),
    }


def _check_geometry(arrays: dict[str, np.ndarray], geometry: dict[str, object]) -> None:
    spectral = tuple(geometry["spectral_shape"])
    surface = tuple(geometry["surface_shape"])
    soil = tuple(geometry["soil_shape"])
    for name, array in sorted(arrays.items()):
        if name.startswith("atmosphere__"):
            expected = spectral if array.ndim == 3 else spectral[1:]
        elif array.ndim == 3:
            expected = soil
        else:
            expected = surface
        if tuple(array.shape) != expected:
            raise ValueError(
                f"Level-4 checkpoint array {name} has shape {tuple(array.shape)}, "
                f"incompatible with the target config's T{geometry['truncation']} / "
                f"{geometry['nlev']}-level geometry (expected {expected}); "
                "migrate a checkpoint written on the target's own grid"
            )


def migrate_level4_checkpoint(
    source_path: str | Path,
    output_path: str | Path,
    *,
    target_config,
    allow_native_zero_moments: bool = False,
    receipt_path: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Migrate one Level-4 checkpoint, explicitly seeding missing moments.

    A native Morrison run normally requires scientifically initialized number
    moments.  Migrating directly into native physics is therefore refused
    unless the caller explicitly acknowledges exact-zero moment seeding.  The
    refusal and the output checkpoint's stamped identity are both taken from
    ``target_config``: a mode supplied independently of the hash lets a caller
    stamp a native run's identity onto a checkpoint the refusal never saw.
    """
    target = Path(output_path)
    receipt = (
        Path(receipt_path)
        if receipt_path is not None
        else target.with_suffix(target.suffix + ".migration.json")
    )
    for path in (target, receipt):
        if path.exists() and not overwrite:
            raise FileExistsError(f"output {path} exists; pass overwrite=True")
    new_config_hash = target_config.config_hash
    if not _is_sha(new_config_hash):
        raise ValueError("target config hash must be a SHA-256 digest")
    mode = str(target_config.physics_mode).strip().lower()
    if mode not in {"reference", "none", "arwen-native"}:
        raise ValueError("target physics mode must be reference, none, or arwen-native")
    if mode == "arwen-native" and not allow_native_zero_moments:
        raise ValueError(
            "Level-4 -> native-physics migration would seed nc/nr/ni/ns/ng "
            "to zero; pass allow_native_zero_moments=True only for an explicit "
            "spin-up experiment"
        )
    geometry = _target_geometry(target_config)

    old_metadata, old_arrays = read_level4_checkpoint(source_path)
    _check_geometry(old_arrays, geometry)
    atmosphere_values = {
        name: old_arrays[f"atmosphere__{name}"]
        for name in SPECTRAL_FIELDS
    }
    # Level 4 carried the condensate species as spectral coefficients;
    # Level 5 carries them on the Gaussian grid (finding 2026-09-02), so
    # they are synthesized once here and floored at zero (the clip is the
    # old representation's ringing).  The moments are seeded to exact
    # zero on the same grid.
    from .runner import build_transform

    transform = build_transform(target_config)
    condensate_negative = 0.0
    for name in CONDENSATE_SPECIES:
        grid = np.asarray(transform.backend.to_numpy(transform.inverse(
            transform.backend.asarray(
                old_arrays[f"atmosphere__{name}"],
                dtype=transform.backend.complex_dtype,
            )
        )), dtype=np.float64)
        condensate_negative = max(condensate_negative, -float(grid.min()))
        atmosphere_values[name] = np.maximum(grid, 0.0)
    template = atmosphere_values["qc"]
    zero_digest = _array_hash(np.zeros_like(template))
    atmosphere_values.update({
        name: np.zeros_like(template) for name in NUMBER_MOMENTS
    })
    atmosphere = MoistHybridState(
        **atmosphere_values,
        time_s=float(old_metadata["time_s"]),
        step=int(old_metadata["step"]),
    )
    carried = {
        member: old_arrays[f"surface__{stored}"]
        for member, stored in _SURFACE_MAP.items()
    }
    # A Level-4 checkpoint predates the static surface fields; the
    # migrated state is seeded with the synthetic planet (the constants
    # Level 4 ran on) and the receipt says so.  Real statics enter only
    # through a fresh cold start on the target config.
    from .statics import (
        SYNTHETIC_CONVENTION, surface_statics_metadata, synthetic_provenance,
        synthetic_surface_statics,
    )

    seeded_statics = synthetic_surface_statics(
        carried["land_fraction"], carried["soil_temperature_k"]
    )
    # The checkpoint's own albedo, emissivity, roughness and land fraction
    # are kept; the categories were derived from that land fraction with
    # the runtime's own water test, so they agree with it.
    for name in ("albedo", "emissivity", "roughness_m", "land_fraction"):
        seeded_statics.pop(name)
    surface = SurfaceState(**carried, **{
        name: np.asarray(value, dtype=carried["land_fraction"].dtype)
        for name, value in seeded_statics.items()
    })
    physics = PhysicsState(
        schema=PHYSICS_STATE_SCHEMA,
        arrays={},
        metadata={
            "migration": {
                "schema": LEVEL4_MIGRATION_SCHEMA,
                "source_checkpoint_self_sha256": old_metadata["self_sha256"],
                "source_level4_pins_hash": LEVEL4_PINS_HASH,
                "seeded_number_moments": list(NUMBER_MOMENTS),
                "seeded_number_moment_sha256": zero_digest,
                "condensate_synthesized_to_grid": list(CONDENSATE_SPECIES),
                "condensate_ringing_clipped_kg_kg": float(condensate_negative),
                "target_physics_mode": mode,
                "native_zero_moments_acknowledged": bool(
                    allow_native_zero_moments
                ),
                "seeded_surface_statics": {
                    **synthetic_provenance(target_config.statics),
                    "fields": sorted(seeded_statics),
                },
            },
            **surface_statics_metadata("synthetic", SYNTHETIC_CONVENTION),
        },
    )
    bundle = ArwenGlobalState(atmosphere, surface, physics)
    trackers = {name: 0.0 for name in TRACKER_KEYS}
    trackers.update(old_metadata["run_trackers"])
    write_checkpoint(
        target,
        bundle,
        config_hash=new_config_hash,
        to_numpy=np.asarray,
        trackers=trackers,
        semi_implicit_scheme=target_config.semi_implicit_scheme,
        integrator=target_config.integrator,
    )
    new_metadata, _ = read_checkpoint(
        target,
        expected_config_hash=new_config_hash,
        semi_implicit_scheme=target_config.semi_implicit_scheme,
        integrator=target_config.integrator,
    )
    _write_migration_receipt(receipt, {
        "source_path": str(source_path),
        "source_file_sha256": _file_hash(source_path),
        "source_checkpoint_self_sha256": old_metadata["self_sha256"],
        "source_level4_pins_hash": LEVEL4_PINS_HASH,
        "source_config_hash": old_metadata["config_hash"],
        "output_path": str(target),
        "output_file_sha256": _file_hash(target),
        "output_checkpoint_self_sha256": new_metadata["self_sha256"],
        "new_config_hash": new_config_hash,
        "target_truncation": geometry["truncation"],
        "target_nlev": geometry["nlev"],
        "target_surface_shape": list(geometry["surface_shape"]),
        "seeded_number_moments": list(NUMBER_MOMENTS),
        "seeded_number_moment_sha256": zero_digest,
        "condensate_synthesized_to_grid": list(CONDENSATE_SPECIES),
        "condensate_ringing_clipped_kg_kg": float(condensate_negative),
        "target_physics_mode": mode,
        "native_zero_moments_acknowledged": bool(allow_native_zero_moments),
        "tracker_policy": "preserve-level4-and-zero-level5-native-trackers",
        "status": "migrated",
    }, semi_implicit_scheme=target_config.semi_implicit_scheme,
       integrator=target_config.integrator)
    read_migration_receipt(receipt)
    return target, receipt


__all__ = [
    "LEVEL4_PINS_HASH",
    "migrate_level4_checkpoint",
    "read_level4_checkpoint",
    "read_migration_receipt",
]
