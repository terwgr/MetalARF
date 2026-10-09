"""Typed Metal launcher for the full Noah-MP glacier column."""

from __future__ import annotations

import numpy as np

from .array import MetalArray

_THREADS = 64
_N_INPUT = 52
_N_OUTPUT = 68


def submit_glacier_columns(*, x, isnow, zsoil, dt, y, status, context):
    """Submit one packed glacier column batch on `context`.

    The resources preserve `noahmp_glacier_column`'s CUDA layout: 52 float
    inputs, one snow index, a shared four-layer soil-depth row, 68 float
    outputs, and a two-int snow-index/error result per column.
    """
    arrays = {"in": x, "ii": isnow, "zsoil_row": zsoil,
              "out": y, "io": status}
    ncase = int(x.shape[0])
    expected = {
        "in": ((ncase, _N_INPUT), np.dtype(np.float32)),
        "ii": ((ncase,), np.dtype(np.int32)),
        "zsoil_row": ((4,), np.dtype(np.float32)),
        "out": ((ncase, _N_OUTPUT), np.dtype(np.float32)),
        "io": ((ncase, 2), np.dtype(np.int32)),
    }
    for name, array in arrays.items():
        shape, dtype = expected[name]
        if not isinstance(array, MetalArray):
            raise TypeError(f"Noah-MP glacier {name} must be a MetalArray")
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"Noah-MP glacier {name} requires {dtype.name} {shape}, "
                f"got {array.dtype.name} {array.shape}")
        if not array.flags.c_contiguous:
            raise ValueError(f"Noah-MP glacier {name} must be contiguous")
        if array.buffer.context is not context:
            raise ValueError(
                f"Noah-MP glacier {name} belongs to another Metal context")
    if ncase == 0:
        return None
    if ncase > np.iinfo(np.int32).max - (_THREADS - 1):
        raise ValueError("Noah-MP glacier dispatch extent exceeds int32")
    return context.submit_kernel(
        "noahmp_glacier_column", arrays,
        {"dt": np.float32(dt), "ncase": ncase}, exact_policy=True,
        grid_size=((ncase + _THREADS - 1) // _THREADS) * _THREADS,
        threads_per_group=_THREADS)


__all__ = ["submit_glacier_columns"]
