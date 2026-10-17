"""Restart checkpoints for the public Hex runner.

The checkpoint stores the model state and the saved time-level diagnostics
that seed the next RK step.  Mesh, reference columns, and run configuration
remain caller-owned setup, just as they do for :func:`run_dry_dycore`.

A physics run adds its column-physics state as a nested mapping: array
leaves are stored as archive members, everything else as JSON, so the file
still loads without pickle.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .driver import DrySavedDiagnostics
from .state import PrognosticState


_SCHEMA = "woof.hex-dry-checkpoint/v1"
_STATE_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
_DIAGNOSTIC_FIELDS = tuple(DrySavedDiagnostics.__slots__)
_PHYSICS_PREFIX = "physics/"


def _flatten_physics(value: Any, path: str, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"checkpoint physics mapping {path!r} needs string keys")
        return {key: _flatten_physics(item, f"{path}/{key}", arrays)
                for key, item in value.items()}
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    if isinstance(value, np.ndarray):
        name = _PHYSICS_PREFIX + path.lstrip("/")
        arrays[name] = np.ascontiguousarray(value)
        return {"__array__": name}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_flatten_physics(item, f"{path}/{index}", arrays)
                for index, item in enumerate(value)]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"checkpoint physics leaf {path!r} is a {type(value).__name__}, "
        "which a checkpoint cannot carry without pickle")


def _rebuild_physics(value: Any, archive: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__array__"}:
            return np.array(archive[value["__array__"]], copy=True)
        return {key: _rebuild_physics(item, archive) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebuild_physics(item, archive) for item in value]
    return value


def write_checkpoint(
    path: str | Path,
    state: PrognosticState,
    *,
    saved_diagnostics: DrySavedDiagnostics | None = None,
    physics: dict[str, Any] | None = None,
) -> Path:
    """Atomically write one restart checkpoint for a committed Hex state."""
    destination = Path(path).expanduser().resolve()
    arrays: dict[str, np.ndarray] = {}
    for name in _STATE_FIELDS:
        value = getattr(state, name)
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        array = np.asarray(value)
        if array.dtype.kind != "f" or not np.isfinite(array).all():
            raise ValueError(f"checkpoint state {name} must be finite floating data")
        arrays[f"state_{name}"] = np.ascontiguousarray(array)
    if not np.isfinite(float(state.time_seconds)):
        raise ValueError("checkpoint time_seconds must be finite")

    diagnostic_names: list[str] = []
    if saved_diagnostics is not None:
        for name in _DIAGNOSTIC_FIELDS:
            value = getattr(saved_diagnostics, name)
            if hasattr(value, "to_numpy"):
                value = value.to_numpy()
            array = np.asarray(value)
            if array.dtype.kind != "f" or not np.isfinite(array).all():
                raise ValueError(
                    f"checkpoint saved diagnostic {name} must be finite floating data")
            arrays[f"diagnostic_{name}"] = np.ascontiguousarray(array)
            diagnostic_names.append(name)

    metadata = {
        "schema": _SCHEMA,
        "time_seconds": float(state.time_seconds),
        "state_fields": list(_STATE_FIELDS),
        "state_shapes": {name: list(arrays[f"state_{name}"].shape)
                         for name in _STATE_FIELDS},
        "state_dtypes": {name: arrays[f"state_{name}"].dtype.str
                         for name in _STATE_FIELDS},
        "saved_diagnostic_fields": diagnostic_names,
    }
    if physics is not None:
        physics_arrays: dict[str, np.ndarray] = {}
        metadata["physics"] = _flatten_physics(physics, "", physics_arrays)
        metadata["physics_arrays"] = sorted(physics_arrays)
        arrays.update(physics_arrays)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w+b", prefix=f".{destination.name}.", suffix=".tmp",
                dir=destination.parent, delete=False) as temporary:
            temporary_name = temporary.name
            np.savez_compressed(
                temporary,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                **arrays,
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        return destination
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def read_checkpoint(
    path: str | Path,
) -> tuple[PrognosticState, DrySavedDiagnostics | None]:
    """Load and validate a Hex checkpoint without pickle or backend imports."""
    source = Path(path).expanduser().resolve(strict=True)
    with np.load(source, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("Hex checkpoint is missing metadata")
        try:
            metadata: dict[str, Any] = json.loads(str(archive["metadata"].item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("Hex checkpoint metadata is malformed") from error
        if not isinstance(metadata, dict) or metadata.get("schema") != _SCHEMA:
            raise ValueError("unsupported Hex checkpoint schema")
        if tuple(metadata.get("state_fields", ())) != _STATE_FIELDS:
            raise ValueError("Hex checkpoint prognostic field inventory differs")
        time_seconds = float(metadata.get("time_seconds", float("nan")))
        if not np.isfinite(time_seconds):
            raise ValueError("Hex checkpoint time_seconds is not finite")
        arrays: dict[str, np.ndarray] = {}
        expected_files = {"metadata"}
        for name in _STATE_FIELDS:
            key = f"state_{name}"
            expected_files.add(key)
            if key not in archive.files:
                raise ValueError(f"Hex checkpoint is missing {name}")
            array = np.array(archive[key], copy=True)
            if (list(array.shape) != metadata.get("state_shapes", {}).get(name)
                    or array.dtype.str != metadata.get("state_dtypes", {}).get(name)):
                raise ValueError(f"Hex checkpoint {name} shape or dtype differs from metadata")
            if array.dtype.kind != "f" or not np.isfinite(array).all():
                raise ValueError(f"Hex checkpoint {name} is not finite floating data")
            arrays[name] = array

        diagnostic_names = tuple(metadata.get("saved_diagnostic_fields", ()))
        if diagnostic_names not in ((), _DIAGNOSTIC_FIELDS):
            raise ValueError("Hex checkpoint saved-diagnostic inventory differs")
        diagnostics_arrays: dict[str, np.ndarray] = {}
        for name in diagnostic_names:
            key = f"diagnostic_{name}"
            expected_files.add(key)
            if key not in archive.files:
                raise ValueError(f"Hex checkpoint is missing saved diagnostic {name}")
            array = np.array(archive[key], copy=True)
            if array.dtype.kind != "f" or not np.isfinite(array).all():
                raise ValueError(f"Hex checkpoint diagnostic {name} is not finite floating data")
            diagnostics_arrays[name] = array
        expected_files.update(metadata.get("physics_arrays", ()))
        if set(archive.files) != expected_files:
            raise ValueError("Hex checkpoint contains an unexpected payload field")

    state = PrognosticState(time_seconds=time_seconds, **arrays)
    state.validate(
        n_cells=state.rho.shape[1], n_edges=state.rho_u.shape[1],
        n_vert_levels=state.rho.shape[0],
    )
    diagnostics = None
    if diagnostics_arrays:
        diagnostics = DrySavedDiagnostics(**diagnostics_arrays)
        diagnostics.validate(state.rho.shape, state.rho.dtype, state.rho_u.shape[1])
    return state, diagnostics


def read_checkpoint_physics(path: str | Path) -> dict[str, Any] | None:
    """The column-physics mapping a physics run stored, or ``None``."""
    source = Path(path).expanduser().resolve(strict=True)
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("schema") != _SCHEMA:
            raise ValueError("unsupported Hex checkpoint schema")
        if "physics" not in metadata:
            return None
        return _rebuild_physics(metadata["physics"], archive)


__all__ = ["read_checkpoint", "read_checkpoint_physics", "write_checkpoint"]
