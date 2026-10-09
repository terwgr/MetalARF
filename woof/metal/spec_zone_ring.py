"""Typed in-device copies for the specified/nested microphysics ring guard."""
from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_spec_zone_ring_copy(source, target, *, row_start: int,
                               rows: int, column_start: int, columns: int,
                               direction: int) -> None:
    if not isinstance(source, MetalArray) or not isinstance(target, MetalArray):
        raise TypeError("ring-copy resources must be MetalArrays")
    if (source.dtype != np.dtype(np.float32)
            or target.dtype != np.dtype(np.float32)
            or source.ndim not in (2, 3)
            or not source.flags.c_contiguous
            or not target.flags.c_contiguous):
        raise TypeError("ring-copy requires contiguous 2-D/3-D float32 fields")
    if source.buffer.context is not target.buffer.context:
        raise ValueError("ring-copy arrays must share a Metal context")
    if source.buffer is target.buffer:
        raise ValueError("ring-copy source and target must be disjoint")
    ny, nx = source.shape[-2:]
    leading = int(np.prod(source.shape[:-2], dtype=np.int64)) or 1
    if (rows <= 0 or columns <= 0 or row_start < 0 or column_start < 0
            or row_start + rows > ny or column_start + columns > nx):
        raise ValueError("ring-copy section is outside the source field")
    if direction != 2 and target.size != leading * rows * columns:
        raise ValueError("ring-copy target has the wrong packed section size")
    if direction == 2 and target.size < 1:
        raise ValueError("ring zero-fill requires a nonempty scratch target")
    if direction not in (0, 1, 2):
        raise ValueError("ring-copy direction must be gather, scatter, or zero")
    count = leading * rows * columns
    expected_target_shape = (tuple(source.shape[:-2]) + (rows, columns)
                             if direction != 2 else None)
    if (expected_target_shape is not None
            and target.shape != expected_target_shape):
        target = target.view(expected_target_shape)
    try:
        source.buffer.context.submit_kernel(
            "spec_zone_ring_copy", {"source": source, "target": target},
            {"count": count, "row_length": nx, "plane": ny * nx,
             "ny": ny, "nz": leading, "row_start": row_start,
             "rows": rows, "column_start": column_start,
             "columns": columns, "direction": direction},
            exact_policy=False, grid_size=count, threads_per_group=256)
    except ValueError as exc:
        raise ValueError(
            f"ring-copy ABI rejected source {source.shape} and target "
            f"{target.shape} (size {target.size}) for rows={rows}, "
            f"columns={columns}, leading={leading}, direction={direction}, "
            f"count={count}, source-bytes={source.nbytes}, "
            f"target-bytes={target.nbytes}") from exc
