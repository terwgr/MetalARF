"""Typed Metal launcher for the production Noah-MP RADIATION batch leaf."""

from __future__ import annotations

import numpy as np

from woof.wrf_exact import ENABLED as WRF_EXACT


_THREADS = 64
_I32_MAX = np.iinfo(np.int32).max


def submit_noahmp_radiation_batch(in_rows, ist, out_rows, *, n=None):
    """Submit ``[n,56] + [n] -> [n,19]`` radiation rows on one context."""
    from .array import MetalArray

    if WRF_EXACT:
        raise NotImplementedError(
            "Noah-MP radiation Metal is disabled under WRF exact mode")
    if not isinstance(in_rows, MetalArray) or in_rows.ndim != 2 \
            or in_rows.shape[1] != 56:
        raise TypeError("Noah-MP radiation input must have shape (n, 56)")
    count = int(in_rows.shape[0]) if n is None else int(n)
    if count <= 0 or count > _I32_MAX:
        raise ValueError("n must be a positive int32 value")
    arrays = {"in": in_rows, "ist": ist, "out": out_rows}
    shapes = {"in": (count, 56), "ist": (count,), "out": (count, 19)}
    dtypes = {"in": np.dtype(np.float32), "ist": np.dtype(np.int32),
              "out": np.dtype(np.float32)}
    context = in_rows.buffer.context
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray) or array.shape != shapes[name]
                or array.dtype != dtypes[name]
                or not array.flags.c_contiguous):
            raise TypeError(
                f"{name} must be contiguous {dtypes[name]} with shape "
                f"{shapes[name]}")
        if array.buffer.context is not context:
            raise ValueError("Noah-MP radiation resources must share one context")
    return context.submit_kernel(
        "noahmp_radiation_batch", arrays, {"n": count}, exact_policy=False,
        grid_size=((count + _THREADS - 1) // _THREADS) * _THREADS,
        threads_per_group=_THREADS)


__all__ = ["submit_noahmp_radiation_batch"]
