"""Runtime seam from Level-5 parent artifacts into regional Arwen.

Nothing in this module invents a second lateral-boundary format.  Translated
frames are validated, converted into the existing ``LateralBoundaries``
objects, and attached through the established Arwen ingest functions.
Imports of the regional CUDA model are lazy so artifact inspection and target
construction remain usable in a CPU-only environment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from ..constants import (
    NUMBER_MOMENTS,
    REGIONAL_ATTACH_SCHEMA,
    REGIONAL_INSTALL_SCHEMA,
    WATER_SPECIES,
)
from ..pins import PINS_HASH, accepted_pins_hashes
from .artifact import (
    canonical,
    file_hash,
    read_parent_series,
    read_regional_frame,
    read_regional_target,
    write_regional_target,
)
from .interpolation import side_tables

_COUPLED_FRAME_FIELDS = {
    "u": "coupled__u",
    "v": "coupled__v",
    "theta": "coupled__theta",
    "phi": "coupled__phi",
    "mu": "coupled__mu",
    "qv": "coupled__qv",
}
_INITIAL_FIELDS = {
    "u": "u",
    "v": "v",
    "w": "w",
    "thp": "thp",
    "php": "php",
    "mup": "mup",
    **{name: name for name in (*WATER_SPECIES, *NUMBER_MOMENTS)},
}


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    result = np.ascontiguousarray(np.asarray(value))
    if result.dtype.hasobject or not np.isfinite(result).all():
        raise ValueError("regional runtime fields must be finite numeric arrays")
    return result


def _write_receipt(path: str | Path, payload: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    row = dict(payload)
    row["pins_hash"] = PINS_HASH
    row["self_sha256"] = hashlib.sha256(canonical(row)).hexdigest()
    temporary = target.with_name(f".{target.name}.partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def read_runtime_receipt(path: str | Path, schema: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != schema or payload.get("pins_hash") not in accepted_pins_hashes():
        raise ValueError("regional runtime receipt schema/pins mismatch")
    self_hash = payload.pop("self_sha256", None)
    if self_hash != hashlib.sha256(canonical(payload)).hexdigest():
        raise ValueError("regional runtime receipt self-hash mismatch")
    payload["self_sha256"] = self_hash
    return payload


def write_regional_target_from_state(
    path: str | Path,
    state,
    *,
    latitude_deg,
    longitude_deg,
    a_half_pa,
    b_half,
    name: str,
    grid_id: str,
    source_identity: Mapping[str, object] | None = None,
) -> Path:
    """Freeze the regional base/coordinate identity needed by translation.

    ``a_half_pa``/``b_half`` are the target *total-pressure* hybrid
    coefficients in top-to-surface order.  State/base arrays retain Arwen's
    native surface-to-top ordering.  The distinction is serialized in the
    target artifact and checked by the translator.
    """
    required = (
        "ht", "sina", "cosa", "mub2d", "c1h", "c2h", "c1f", "c2f",
        "thb", "phb", "msft", "msfu", "msfv",
    )
    missing = [item for item in required if getattr(state, item, None) is None]
    if missing:
        raise ValueError("regional state lacks target carriers: " + ", ".join(missing))
    latitude = _host(latitude_deg).astype(np.float64, copy=False)
    longitude = _host(longitude_deg).astype(np.float64, copy=False)
    if latitude.ndim != 2 or longitude.shape != latitude.shape:
        raise ValueError("regional latitude/longitude must be equal 2-D arrays")
    ny, nx = latitude.shape
    if tuple(_host(state.mub2d).shape) != (ny, nx):
        raise ValueError("regional target coordinates do not match state.mub2d")
    a = _host(a_half_pa).astype(np.float64, copy=False)
    b = _host(b_half).astype(np.float64, copy=False)
    if a.ndim != 1 or b.shape != a.shape or a.size < 3:
        raise ValueError("regional A/B coefficients must be equal 1-D arrays")
    nz = a.size - 1
    if np.any(np.diff(a + b * 100_000.0) <= 0.0):
        raise ValueError("regional A/B pressure must increase top-to-surface")
    if _host(state.c1h).shape != (nz,) or _host(state.c1f).shape != (nz + 1,):
        raise ValueError("regional A/B level count does not match WOOF state")
    arrays = {
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "terrain_height_m": _host(state.ht).astype(np.float64),
        "cosa": _host(state.cosa).astype(np.float64),
        "sina": _host(state.sina).astype(np.float64),
        "a_half_pa": a,
        "b_half": b,
        "mub2d": _host(state.mub2d).astype(np.float64),
        "c1h": _host(state.c1h).astype(np.float64),
        "c2h": _host(state.c2h).astype(np.float64),
        "c1f": _host(state.c1f).astype(np.float64),
        "c2f": _host(state.c2f).astype(np.float64),
        "thb": _host(state.thb).astype(np.float64),
        "phb": _host(state.phb).astype(np.float64),
        "msft": _host(state.msft).astype(np.float64),
        "msfu": _host(state.msfu).astype(np.float64),
        "msfv": _host(state.msfv).astype(np.float64),
    }
    return write_regional_target(
        path,
        arrays,
        name=name,
        grid_id=grid_id,
        source_identity=dict(source_identity or {}),
    )


def _array_module(value):
    if isinstance(value, np.ndarray):
        return np
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:  # pragma: no cover - target-device path
        raise RuntimeError("installing into a CUDA DomainState requires CuPy") from exc
    if isinstance(value, cp.ndarray):
        return cp
    raise TypeError(f"unsupported regional state array type {type(value)!r}")


def install_regional_initial_frame(
    state,
    frame_path: str | Path,
    target_path: str | Path,
    *,
    receipt_path: str | Path | None = None,
) -> dict[str, object]:
    """Transactionally install one translated frame into an existing state."""
    frame_meta, frame = read_regional_frame(frame_path)
    target_meta, _target = read_regional_target(target_path)
    if frame_meta.get("target_self_sha256") != target_meta["self_sha256"]:
        raise ValueError("regional frame was translated for a different target")

    staged: dict[str, object] = {}
    installed: list[str] = []
    field_map = dict(_INITIAL_FIELDS)
    # P and ALT are diagnostic state in Arwen but are consumed by physics
    # before the first completed step.  Seed them when the state carries the
    # arrays; the normal model preparation remains free to recompute them.
    if getattr(state, "p", None) is not None:
        field_map["p"] = "pressure"
    if getattr(state, "alt", None) is not None:
        # state.alt is the dry specific volume alpha_d; the frame's own
        # ``density`` is the total (moist) density, low by (1 + q_total).
        field_map["alt"] = "dry_inverse_density"
    for state_name, frame_name in field_map.items():
        target = getattr(state, state_name, None)
        if target is None:
            # Inactive optional hydrometeors/moments are not invented.
            if state_name in (*WATER_SPECIES, *NUMBER_MOMENTS):
                continue
            raise ValueError(f"regional state lacks required field {state_name}")
        if frame_name not in frame:
            if state_name in (*WATER_SPECIES, *NUMBER_MOMENTS):
                raise ValueError(f"translated frame lacks active field {state_name}")
            raise ValueError(f"translated frame lacks required field {frame_name}")
        source = frame[frame_name]
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                f"translated {frame_name} shape {source.shape} != state {state_name} {target.shape}"
            )
        xp = _array_module(target)
        staged[state_name] = xp.ascontiguousarray(
            xp.asarray(source, dtype=target.dtype)
        )
        installed.append(state_name)

    # Only commit after every active field has been converted successfully.
    for name in installed:
        getattr(state, name)[...] = staged[name]
    # The frame is the initial condition, so the model clock starts at it.
    # Both attach paths in woof.ingest.lateral_bc end by zeroing this same
    # field; writing the frame's absolute parent time here instead would make
    # the receipt below assert a model time no state ever holds.
    state.elapsed_seconds = 0.0
    payload: dict[str, object] = {
        "schema": REGIONAL_INSTALL_SCHEMA,
        "frame_path": str(frame_path),
        "frame_file_sha256": file_hash(frame_path),
        "frame_self_sha256": frame_meta["self_sha256"],
        "target_path": str(target_path),
        "target_self_sha256": target_meta["self_sha256"],
        "installed_fields": installed,
        "model_time_s": float(state.elapsed_seconds),
        "parent_time_s": float(frame_meta["time_s"]),
        "status": "installed",
    }
    payload["pins_hash"] = PINS_HASH
    payload["self_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    if receipt_path is not None:
        _write_receipt(receipt_path, {
            key: value for key, value in payload.items()
            if key not in {"pins_hash", "self_sha256"}
        })
        payload = read_runtime_receipt(receipt_path, REGIONAL_INSTALL_SCHEMA)
    return payload


def build_lateral_boundaries_from_parent_series(
    series_path: str | Path,
    *,
    spec_bdy_width: int = 5,
    spec_zone: int = 1,
    relax_zone: int = 4,
):
    """Build the existing Arwen ``LateralBoundaries`` object from a series."""
    series, resolved = read_parent_series(series_path)
    if spec_zone < 1 or relax_zone < 2:
        raise ValueError("spec_zone must be >=1 and relax_zone >=2")
    if spec_bdy_width < spec_zone + relax_zone:
        raise ValueError("spec_bdy_width must cover spec_zone + relax_zone")
    try:
        from woof.ingest.lateral_bc import (
            BoundaryInterval,
            FieldBoundary,
            LateralBoundaries,
            SideBoundary,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "building live WOOF LBCs requires the complete WOOF repository"
        ) from exc

    origin = float(resolved[0][1]["time_s"])
    intervals = []
    for (_path0, meta0, arrays0), (_path1, meta1, arrays1) in zip(
        resolved[:-1], resolved[1:]
    ):
        start = float(meta0["time_s"])
        end = float(meta1["time_s"])
        duration = end - start
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("parent series contains a nonpositive interval")
        fields = {}
        for lbc_name, frame_name in _COUPLED_FRAME_FIELDS.items():
            first = side_tables(arrays0[frame_name], spec_bdy_width)
            second = side_tables(arrays1[frame_name], spec_bdy_width)
            sides = {
                side: SideBoundary(
                    first[side],
                    np.ascontiguousarray((second[side] - first[side]) / duration),
                )
                for side in ("west", "east", "south", "north")
            }
            fields[lbc_name] = FieldBoundary(**sides)
        intervals.append(
            BoundaryInterval(start - origin, end - origin, fields)
        )
    boundaries = LateralBoundaries(
        tuple(intervals), int(spec_bdy_width), int(spec_zone), int(relax_zone)
    )
    return boundaries, series, resolved


def attach_parent_series(
    state,
    series_path: str | Path,
    *,
    spec_bdy_width: int = 5,
    spec_zone: int = 1,
    relax_zone: int = 4,
    streaming: bool = False,
    receipt_path: str | Path | None = None,
) -> dict[str, object]:
    boundaries, series, resolved = build_lateral_boundaries_from_parent_series(
        series_path,
        spec_bdy_width=spec_bdy_width,
        spec_zone=spec_zone,
        relax_zone=relax_zone,
    )
    try:
        from woof.ingest.lateral_bc import (
            attach_lateral_boundaries,
            attach_streaming_lateral_boundaries,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "attaching global parent forcing requires the complete WOOF repository"
        ) from exc
    if streaming:
        attach_streaming_lateral_boundaries(state, boundaries)
        mode = "streaming-external"
    else:
        attach_lateral_boundaries(state, boundaries)
        mode = "eager-resident"
    payload = {
        "schema": REGIONAL_ATTACH_SCHEMA,
        "series_path": str(series_path),
        "series_file_sha256": file_hash(series_path),
        "series_self_sha256": series["self_sha256"],
        "first_frame_self_sha256": resolved[0][1]["self_sha256"],
        "last_frame_self_sha256": resolved[-1][1]["self_sha256"],
        "interval_count": len(boundaries.intervals),
        "field_inventory": sorted(boundaries.intervals[0].fields),
        "attachment_mode": mode,
        "spec_bdy_width": int(spec_bdy_width),
        "spec_zone": int(spec_zone),
        "relax_zone": int(relax_zone),
        "status": "attached",
    }
    payload["pins_hash"] = PINS_HASH
    payload["self_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    if receipt_path is not None:
        _write_receipt(receipt_path, {
            key: value for key, value in payload.items()
            if key not in {"pins_hash", "self_sha256"}
        })
        payload = read_runtime_receipt(receipt_path, REGIONAL_ATTACH_SCHEMA)
    return payload


def install_initial_and_attach_parent(
    state,
    series_path: str | Path,
    target_path: str | Path,
    *,
    spec_bdy_width: int = 5,
    spec_zone: int = 1,
    relax_zone: int = 4,
    streaming: bool = False,
    receipt_directory: str | Path | None = None,
) -> dict[str, object]:
    series, resolved = read_parent_series(series_path)
    first_path = resolved[0][0]
    install_receipt = None
    attach_receipt = None
    if receipt_directory is not None:
        directory = Path(receipt_directory)
        install_receipt = directory / "arwen_global_regional_install.json"
        attach_receipt = directory / "arwen_global_regional_attach.json"
    installed = install_regional_initial_frame(
        state, first_path, target_path, receipt_path=install_receipt
    )
    attached = attach_parent_series(
        state,
        series_path,
        spec_bdy_width=spec_bdy_width,
        spec_zone=spec_zone,
        relax_zone=relax_zone,
        streaming=streaming,
        receipt_path=attach_receipt,
    )
    return {"install": installed, "attach": attached, "series": series}


__all__ = [
    "attach_parent_series",
    "build_lateral_boundaries_from_parent_series",
    "install_initial_and_attach_parent",
    "install_regional_initial_frame",
    "read_runtime_receipt",
    "write_regional_target_from_state",
]
