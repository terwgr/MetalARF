"""Hash-bound NPZ checkpoints for standalone global-spectral runs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .constants import CHECKPOINT_SCHEMA
from .pins import ACCEPTED_PINS_HASHES, PINS_HASH
from .state import PrimitiveDryState, ShallowWaterState

_TRACKER_KEYS = (
    "maximum_spectral_cfl",
    "maximum_mass_fixer_log_offset",
)


def _canonical(value: dict) -> bytes:
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


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _run_trackers(value: dict[str, object] | None) -> dict[str, float]:
    if value is None:
        return {name: 0.0 for name in _TRACKER_KEYS}
    if not isinstance(value, dict) or set(value) != set(_TRACKER_KEYS):
        raise ValueError(
            f"run_trackers must contain exactly {list(_TRACKER_KEYS)}"
        )
    result: dict[str, float] = {}
    for name in _TRACKER_KEYS:
        raw = value[name]
        if isinstance(raw, bool):
            raise ValueError(f"run tracker {name} must be a finite number")
        number = float(raw)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"run tracker {name} must be finite and nonnegative")
        result[name] = number
    return result


def state_arrays(state, to_numpy) -> dict[str, np.ndarray]:
    if isinstance(state, ShallowWaterState):
        names = ("vorticity", "divergence", "geopotential")
    elif isinstance(state, PrimitiveDryState):
        names = (
            "vorticity",
            "divergence",
            "temperature",
            "log_surface_pressure",
        )
    else:
        raise TypeError(f"unsupported checkpoint state {type(state)!r}")
    return {name: np.asarray(to_numpy(getattr(state, name))) for name in names}


def write_checkpoint(
    path: str | Path,
    state,
    *,
    model: str,
    config_hash: str,
    to_numpy,
    run_trackers: dict[str, object] | None = None,
) -> Path:
    if model not in {"shallow-water", "primitive-dry"}:
        raise ValueError(f"unknown checkpoint model {model!r}")
    if not _is_sha256(config_hash):
        raise ValueError("config_hash must be a 64-character SHA-256 hex digest")
    if isinstance(state.step, bool) or int(state.step) != state.step or state.step < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    if not math.isfinite(float(state.time_s)) or float(state.time_s) < 0.0:
        raise ValueError("checkpoint time_s must be finite and nonnegative")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = state_arrays(state, to_numpy)
    for name, array in arrays.items():
        if array.dtype.kind != "c":
            raise ValueError(f"checkpoint spectral array {name} must be complex")
        if not np.isfinite(array.real).all() or not np.isfinite(array.imag).all():
            raise ValueError(f"checkpoint spectral array {name} is non-finite")
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "model": model,
        "config_hash": config_hash,
        "pins_hash": PINS_HASH,
        "time_s": float(state.time_s),
        "step": int(state.step),
        "run_trackers": _run_trackers(run_trackers),
        "arrays": {
            name: {
                "shape": list(array.shape),
                "dtype": array.dtype.str,
                "sha256": _array_hash(array),
            }
            for name, array in arrays.items()
        },
    }
    metadata["self_sha256"] = hashlib.sha256(_canonical(metadata)).hexdigest()
    temp = target.with_name(f".{target.name}.partial-{os.getpid()}")
    with temp.open("wb") as stream:
        np.savez_compressed(
            stream,
            __metadata__=np.asarray(
                json.dumps(metadata, sort_keys=True, allow_nan=False)
            ),
            **arrays,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, target)
    return target


def read_checkpoint(
    path: str | Path,
    *,
    expected_config_hash: str | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        if "__metadata__" not in data:
            raise ValueError(f"checkpoint {source} has no metadata")
        metadata = json.loads(str(data["__metadata__"].item()))
        arrays = {
            name: np.array(data[name], copy=True)
            for name in data.files
            if name != "__metadata__"
        }
    required = {
        "schema",
        "model",
        "config_hash",
        "pins_hash",
        "time_s",
        "step",
        "run_trackers",
        "arrays",
        "self_sha256",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError(
            f"checkpoint metadata keys {sorted(metadata) if isinstance(metadata, dict) else type(metadata).__name__} "
            f"do not match {sorted(required)}"
        )
    if metadata.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"checkpoint schema mismatch: {metadata.get('schema')!r}")
    if metadata.get("model") not in {"shallow-water", "primitive-dry"}:
        raise ValueError(f"unknown checkpoint model {metadata.get('model')!r}")
    if metadata.get("pins_hash") not in ACCEPTED_PINS_HASHES:
        raise ValueError("checkpoint arithmetic pin does not match this implementation")
    if not _is_sha256(metadata.get("config_hash")):
        raise ValueError("checkpoint config hash is not a SHA-256 digest")
    if expected_config_hash is not None and metadata.get("config_hash") != expected_config_hash:
        raise ValueError("checkpoint config identity does not match this run")
    if isinstance(metadata["step"], bool) or not isinstance(metadata["step"], int) or metadata["step"] < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    if isinstance(metadata["time_s"], bool):
        raise ValueError("checkpoint time_s must be finite and nonnegative")
    time_s = float(metadata["time_s"])
    if not math.isfinite(time_s) or time_s < 0.0:
        raise ValueError("checkpoint time_s must be finite and nonnegative")
    metadata["run_trackers"] = _run_trackers(metadata["run_trackers"])
    self_hash = metadata.pop("self_sha256")
    if not _is_sha256(self_hash) or self_hash != hashlib.sha256(_canonical(metadata)).hexdigest():
        raise ValueError("checkpoint metadata self-hash mismatch")
    metadata["self_sha256"] = self_hash
    expected = metadata.get("arrays")
    if not isinstance(expected, dict) or set(expected) != set(arrays):
        raise ValueError("checkpoint array inventory mismatch")
    expected_names = (
        {"vorticity", "divergence", "geopotential"}
        if metadata["model"] == "shallow-water"
        else {"vorticity", "divergence", "temperature", "log_surface_pressure"}
    )
    if set(arrays) != expected_names:
        raise ValueError(
            f"checkpoint model {metadata['model']!r} requires arrays {sorted(expected_names)}"
        )
    for name, array in arrays.items():
        row = expected[name]
        if not isinstance(row, dict) or set(row) != {"shape", "dtype", "sha256"}:
            raise ValueError(f"checkpoint array metadata for {name} is malformed")
        if list(array.shape) != row["shape"] or array.dtype.str != row["dtype"]:
            raise ValueError(f"checkpoint array {name} shape/dtype mismatch")
        if array.dtype.kind != "c":
            raise ValueError(f"checkpoint spectral array {name} must be complex")
        if not _is_sha256(row["sha256"]) or _array_hash(array) != row["sha256"]:
            raise ValueError(f"checkpoint array {name} hash mismatch")
        if not np.isfinite(array.real).all() or not np.isfinite(array.imag).all():
            raise ValueError(f"checkpoint spectral array {name} is non-finite")
    return metadata, arrays


def state_from_checkpoint(metadata: dict, arrays: dict[str, np.ndarray], backend):
    kwargs = {
        name: backend.asarray(value, dtype=backend.complex_dtype)
        for name, value in arrays.items()
    }
    kwargs.update(time_s=float(metadata["time_s"]), step=int(metadata["step"]))
    if metadata["model"] == "shallow-water":
        return ShallowWaterState(**kwargs)
    if metadata["model"] == "primitive-dry":
        return PrimitiveDryState(**kwargs)
    raise ValueError(f"unknown checkpoint model {metadata['model']!r}")


__all__ = [
    "read_checkpoint",
    "state_arrays",
    "state_from_checkpoint",
    "write_checkpoint",
]
