"""Hash-bound artifacts for Level-5 global-to-regional translation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..constants import (
    REGIONAL_FRAME_SCHEMA,
    REGIONAL_SERIES_SCHEMA,
    REGIONAL_TARGET_SCHEMA,
)
from ..pins import PINS_HASH, accepted_pins_hashes


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode())
    digest.update(str(arr.shape).encode())
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_npz(path: Path, metadata: dict, arrays: dict[str, object]) -> Path:
    packed = {name: np.asarray(value) for name, value in arrays.items()}
    for name, value in packed.items():
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError(f"artifact array {name} must be finite numeric data")
    metadata = dict(metadata)
    metadata["arrays"] = {
        name: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": array_hash(value),
        }
        for name, value in packed.items()
    }
    metadata["self_sha256"] = hashlib.sha256(canonical(metadata)).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            __metadata__=np.asarray(
                json.dumps(metadata, sort_keys=True, allow_nan=False)
            ),
            **packed,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def _read_npz(path: Path, schema: str) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        if "__metadata__" not in archive:
            raise ValueError(f"artifact {path} has no metadata")
        metadata = json.loads(str(archive["__metadata__"].item()))
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "__metadata__"
        }
    if metadata.get("schema") != schema:
        raise ValueError(f"artifact schema mismatch: expected {schema!r}")
    if metadata.get("pins_hash") not in accepted_pins_hashes():
        raise ValueError("artifact arithmetic/coupling pins mismatch")
    self_hash = metadata.pop("self_sha256", None)
    if self_hash != hashlib.sha256(canonical(metadata)).hexdigest():
        raise ValueError("artifact metadata self-hash mismatch")
    metadata["self_sha256"] = self_hash
    expected = metadata.get("arrays")
    if not isinstance(expected, dict) or set(expected) != set(arrays):
        raise ValueError("artifact array inventory mismatch")
    for name, value in arrays.items():
        row = expected[name]
        if (
            list(value.shape) != row.get("shape")
            or value.dtype.str != row.get("dtype")
            or array_hash(value) != row.get("sha256")
        ):
            raise ValueError(f"artifact array {name} hash/shape/dtype mismatch")
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError(f"artifact array {name} is invalid")
    return metadata, arrays


def write_regional_target(
    path: str | Path,
    arrays: dict[str, object],
    *,
    name: str,
    grid_id: str,
    source_identity: dict[str, object] | None = None,
) -> Path:
    required = {
        "latitude_deg", "longitude_deg", "terrain_height_m", "cosa", "sina",
        "a_half_pa", "b_half", "mub2d", "c1h", "c2h", "c1f", "c2f",
        "thb", "phb", "msft", "msfu", "msfv",
    }
    if set(arrays) != required:
        raise ValueError(
            f"regional target arrays {sorted(arrays)} do not match {sorted(required)}"
        )
    metadata = {
        "schema": REGIONAL_TARGET_SCHEMA,
        "pins_hash": PINS_HASH,
        "name": str(name),
        "grid_id": str(grid_id),
        "source_identity": dict(source_identity or {}),
        "vertical_order": {
            "a_half_pa_b_half": "top-to-surface",
            "regional_state_and_base": "surface-to-top",
        },
    }
    return _write_npz(Path(path), metadata, arrays)


def read_regional_target(path: str | Path):
    metadata, arrays = _read_npz(Path(path), REGIONAL_TARGET_SCHEMA)
    lat = arrays["latitude_deg"]
    lon = arrays["longitude_deg"]
    if lat.ndim != 2 or lon.shape != lat.shape:
        raise ValueError("regional target latitude/longitude must be equal 2-D arrays")
    ny, nx = lat.shape
    a = arrays["a_half_pa"]
    b = arrays["b_half"]
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size < 3:
        raise ValueError("regional target A/B coefficients must be equal 1-D arrays")
    if np.any(a < 0.0) or np.any(b < 0.0) or np.any(b > 1.0):
        raise ValueError("regional target A/B coefficients are outside physical bounds")
    # The translator reads the column dry mass off these coefficients as
    # sum(dp) = p_half[-1] - p_half[0] = ps - p_top.  That identity needs the
    # hybrid endpoints exactly: without them dry_mu, mub2d and every coupled
    # field built on c1*mu+c2 measure different quantities.
    if b[0] != 0.0 or b[-1] != 1.0 or a[-1] != 0.0:
        raise ValueError(
            "regional target A/B endpoints must be (a[0]=p_top, b[0]=0) and "
            "(a[-1]=0, b[-1]=1) for the column mass to be ps - p_top"
        )
    p_top_pa = float(a[0])
    for representative_ps in (50_000.0, 100_000.0, 120_000.0):
        pressure = a + b * representative_ps
        if np.any(pressure <= 0.0) or np.any(np.diff(pressure) <= 0.0):
            raise ValueError("regional target A/B pressure is not top-to-surface monotonic")
    if np.any(arrays["msft"] <= 0.0) or np.any(arrays["msfu"] <= 0.0) or np.any(arrays["msfv"] <= 0.0):
        raise ValueError("regional target map factors must be positive")
    angle_norm = arrays["cosa"] ** 2 + arrays["sina"] ** 2
    # Targets are written from state arrays that may be float32, so the norm
    # carries one float32 round-trip: |c^2+s^2-1| <= 2*eps_f32 = 2.384e-7
    # analytically, measured max 8.39e-8 over 500001 angles spanning the full
    # circle.  The gate catches a cosa/sina pair that is not a rotation at all
    # (unnormalized, truncated, or a mis-shaped copy); it cannot detect a
    # unit-norm pair evaluated at the wrong point.
    if not np.allclose(angle_norm, 1.0, rtol=0.0, atol=2.4e-7):
        raise ValueError("regional target wind rotation is not unit-normalized")
    nlev = a.size - 1
    shapes = {
        "terrain_height_m": (ny, nx), "cosa": (ny, nx), "sina": (ny, nx),
        "b_half": (nlev + 1,), "mub2d": (ny, nx),
        "c1h": (nlev,), "c2h": (nlev,),
        "c1f": (nlev + 1,), "c2f": (nlev + 1,),
        "msft": (ny, nx), "msfu": (ny, nx + 1), "msfv": (ny + 1, nx),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"regional target {name} shape {arrays[name].shape} != {shape}")
    if arrays["thb"].shape not in {(nlev,), (nlev, ny, nx)}:
        raise ValueError("regional target thb has the wrong shape")
    if arrays["phb"].shape not in {(nlev + 1,), (nlev + 1, ny, nx)}:
        raise ValueError("regional target phb has the wrong shape")
    base_surface_pa = arrays["mub2d"] + p_top_pa
    if np.any(base_surface_pa < 30_000.0) or np.any(base_surface_pa > 120_000.0):
        # Same admissible surface-pressure band the translator enforces on the
        # terrain-adjusted ps: mub2d is the base column of the same coordinate,
        # so a value outside it is not a base state for these coefficients, and
        # mu' = dry_mu - mub2d is then the size of the column itself.
        raise ValueError(
            "regional target mub2d implies a base surface pressure outside "
            "[30 kPa, 120 kPa] for its own A/B coefficients"
        )
    base_half = a[:, None, None] + b[:, None, None] * base_surface_pa[None]
    if np.any(np.diff(base_half, axis=0) <= 0.0):
        # WRF's compute_vcoord_1d_coeffs validity check applied to the target's
        # own base column: a hybrid coordinate whose reference dry pressure
        # folds at this mub2d cannot order the levels it is about to fill.
        raise ValueError(
            "regional target base dry pressure from mub2d is not "
            "top-to-surface monotonic"
        )
    return metadata, arrays


def write_regional_frame(path: str | Path, metadata: dict, arrays: dict[str, object]):
    row = dict(metadata)
    row.update(schema=REGIONAL_FRAME_SCHEMA, pins_hash=PINS_HASH)
    return _write_npz(Path(path), row, arrays)


def read_regional_frame(path: str | Path):
    metadata, arrays = _read_npz(Path(path), REGIONAL_FRAME_SCHEMA)
    required = {
        "u", "v", "w", "theta", "temperature", "pressure", "p_half",
        "geopotential", "geopotential_half", "density", "dry_inverse_density",
        "surface_pressure", "dry_mu", "mup", "thp", "php",
        "coupled__u", "coupled__v", "coupled__theta", "coupled__phi",
        "coupled__mu", "coupled__qv",
    }
    if not required <= set(arrays):
        raise ValueError("translated regional frame inventory is incomplete")
    theta = arrays["theta"]
    if theta.ndim != 3 or theta.shape[0] < 2:
        raise ValueError("translated regional theta must be (nz,ny,nx)")
    nz, ny, nx = theta.shape
    expected = {
        "u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
        "w": (nz + 1, ny, nx), "temperature": (nz, ny, nx),
        "pressure": (nz, ny, nx), "p_half": (nz + 1, ny, nx),
        "geopotential": (nz, ny, nx),
        "geopotential_half": (nz + 1, ny, nx),
        "density": (nz, ny, nx), "dry_inverse_density": (nz, ny, nx),
        "surface_pressure": (ny, nx),
        "dry_mu": (ny, nx), "mup": (ny, nx), "thp": (nz, ny, nx),
        "php": (nz + 1, ny, nx),
        "coupled__u": (nz, ny, nx + 1),
        "coupled__v": (nz, ny + 1, nx),
        "coupled__theta": (nz, ny, nx),
        "coupled__phi": (nz + 1, ny, nx),
        "coupled__mu": (1, ny, nx),
        "coupled__qv": (nz, ny, nx),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"translated regional {name} shape {arrays[name].shape} != {shape}")
    for name, value in arrays.items():
        if name in {"theta", "temperature", "pressure", "geopotential", "density", "thp"} or name in {"qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng"}:
            if value.shape != (nz, ny, nx):
                raise ValueError(f"translated regional scalar {name} shape mismatch")
    if np.any(arrays["pressure"] <= 0.0) or np.any(arrays["density"] <= 0.0):
        raise ValueError("translated regional pressure/density must be positive")
    if np.any(arrays["dry_inverse_density"] <= 0.0):
        raise ValueError("translated regional dry alpha must be positive")
    return metadata, arrays


def write_parent_series(
    path: str | Path,
    frames: list[tuple[str | Path, dict]],
    *,
    target_path: str | Path,
    target_self_sha256: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    previous = None
    for frame_path, frame_metadata in frames:
        source = Path(frame_path)
        time_s = float(frame_metadata["time_s"])
        if previous is not None and time_s <= previous:
            raise ValueError("parent-series frame times must be strictly increasing")
        previous = time_s
        frame_target = frame_metadata["target_self_sha256"]
        if frame_target != target_self_sha256:
            raise ValueError(f"frame {source} belongs to a different target")
        try:
            stored = str(source.relative_to(target.parent))
        except ValueError:
            stored = str(source.resolve())
        rows.append({
            "path": stored,
            "file_sha256": file_hash(source),
            "frame_self_sha256": frame_metadata["self_sha256"],
            "time_s": time_s,
            "source_parent_self_sha256": frame_metadata["source_parent_self_sha256"],
            "target_self_sha256": frame_target,
        })
    if len(rows) < 2:
        raise ValueError("parent series requires at least two translated frames")
    target_source = Path(target_path)
    try:
        stored_target = str(target_source.relative_to(target.parent))
    except ValueError:
        stored_target = str(target_source.resolve())
    payload = {
        "schema": REGIONAL_SERIES_SCHEMA,
        "pins_hash": PINS_HASH,
        "target_path": stored_target,
        "target_file_sha256": file_hash(target_source),
        "target_self_sha256": target_self_sha256,
        "frames": rows,
    }
    payload["self_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    temporary = target.with_name(f".{target.name}.partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def read_parent_series(path: str | Path):
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != REGIONAL_SERIES_SCHEMA:
        raise ValueError("parent-series schema mismatch")
    if payload.get("pins_hash") not in accepted_pins_hashes():
        raise ValueError("parent-series pins mismatch")
    self_hash = payload.pop("self_sha256", None)
    if self_hash != hashlib.sha256(canonical(payload)).hexdigest():
        raise ValueError("parent-series self-hash mismatch")
    payload["self_sha256"] = self_hash
    if not isinstance(payload.get("frames"), list) or len(payload["frames"]) < 2:
        raise ValueError("parent-series frames are incomplete")
    target_path = Path(payload.get("target_path", ""))
    if not target_path.is_absolute():
        target_path = source.parent / target_path
    if file_hash(target_path) != payload.get("target_file_sha256"):
        raise ValueError("parent-series target file hash mismatch")
    target_metadata, _target_arrays = read_regional_target(target_path)
    if target_metadata["self_sha256"] != payload.get("target_self_sha256"):
        raise ValueError("parent-series target metadata identity mismatch")
    resolved = []
    previous = None
    for row in payload["frames"]:
        frame = Path(row["path"])
        if not frame.is_absolute():
            frame = source.parent / frame
        if file_hash(frame) != row["file_sha256"]:
            raise ValueError(f"parent-series frame file hash mismatch: {frame}")
        metadata, arrays = read_regional_frame(frame)
        if metadata["self_sha256"] != row["frame_self_sha256"]:
            raise ValueError("parent-series frame metadata identity mismatch")
        if metadata["source_parent_self_sha256"] != row["source_parent_self_sha256"]:
            raise ValueError("parent-series parent identity mismatch")
        # The series declares one regional target and every consumer -- the
        # LBC builder, the installer, the runtime attach -- reads the frames
        # as if they all share it.  A frame translated for another target
        # carries another grid's terrain, A/B and base column mass, so its
        # boundary tables would be forced onto this domain undetected.
        if row.get("target_self_sha256") != payload.get("target_self_sha256"):
            raise ValueError(f"frame {frame} belongs to a different target")
        if metadata["target_self_sha256"] != payload.get("target_self_sha256"):
            raise ValueError(f"frame {frame} belongs to a different target")
        time_s = float(row["time_s"])
        if time_s != float(metadata["time_s"]):
            raise ValueError("parent-series frame time mismatch")
        if previous is not None and time_s <= previous:
            raise ValueError("parent-series times are not strictly increasing")
        previous = time_s
        resolved.append((frame, metadata, arrays))
    return payload, resolved


__all__ = [
    "array_hash", "canonical", "file_hash", "read_parent_series",
    "read_regional_frame", "read_regional_target", "write_parent_series",
    "write_regional_frame", "write_regional_target",
]
