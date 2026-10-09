"""Restart checkpoints for the public Hex dry-dynamics runner.

The checkpoint stores the model state and the saved time-level diagnostics
that seed the next RK step.  Mesh, reference columns, and run configuration
remain caller-owned setup, just as they do for :func:`run_dry_dycore`.
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


def write_checkpoint(
    path: str | Path,
    state: PrognosticState,
    *,
    saved_diagnostics: DrySavedDiagnostics | None = None,
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


__all__ = ["read_checkpoint", "write_checkpoint"]
