"""Typed launcher for the production classic Thompson phase cleanup."""
from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_thompson_final_phase_cleanup_with_columns(
        *, qc, qi, ni, temperature, pressure, qv, micro_columns):
    """Apply WRF's terminal cleanup using the caller's entry microphysics mask."""
    if not isinstance(qc, MetalArray) or qc.ndim != 3:
        raise TypeError("Thompson cleanup qc must be a 3-D MetalArray")
    shape = tuple(map(int, qc.shape))
    nz, ny, nx = shape
    n = int(np.prod(shape, dtype=np.int64))
    if min(shape) <= 0 or max(n, nz, ny, nx, ny * nx) > 0x7fffffff:
        raise ValueError(f"Thompson cleanup dimensions are outside int32: {shape}")
    arrays = {"qc": qc, "qi": qi, "ni": ni, "temperature": temperature,
              "pressure": pressure, "qv": qv,
              "micro_columns": micro_columns}
    context = qc.buffer.context
    for name, value in arrays.items():
        if not isinstance(value, MetalArray):
            raise TypeError(f"{name} must be a MetalArray")
        expected = (ny, nx) if name == "micro_columns" else shape
        if (value.shape != expected or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"{name} must be contiguous float32 with shape {expected}")
        if value.buffer.context is not context:
            raise ValueError("Thompson cleanup resources must share one Metal context")
    names = tuple(arrays)
    for pos, name in enumerate(names):
        left = arrays[name]
        for other_name in names[pos + 1:]:
            right = arrays[other_name]
            if (left.buffer is right.buffer
                    and left.byte_offset < right.byte_offset + right.nbytes
                    and right.byte_offset < left.byte_offset + left.nbytes):
                raise ValueError(f"restricted Thompson cleanup resources overlap: {name}, {other_name}")
    return context.submit_kernel(
        "thompson_final_phase_cleanup_with_columns", arrays,
        {"ncol": ny * nx, "size": n, "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False,
        grid_size=n, threads_per_group=256)


__all__ = ["submit_thompson_final_phase_cleanup_with_columns"]
