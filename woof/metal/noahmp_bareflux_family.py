"""Typed Metal launcher for the Noah-MP BARE_FLUX caller slice."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_THREADS = 64
_I32_MAX = np.iinfo(np.int32).max
_FLOAT = np.dtype(np.float32)


def submit_bare_flux(context, in_rows: MetalArray, ii: MetalArray,
                     out_rows: MetalArray):
    """Submit the packed opt_sfc=1 / opt_stc=1 BARE_FLUX Metal slice."""
    arrays = {"in": in_rows, "ii": ii, "out": out_rows}
    expected = {"in": (_FLOAT, 53), "ii": (np.dtype(np.int32), 5),
                "out": (_FLOAT, 13)}
    if any(not isinstance(a, MetalArray) for a in arrays.values()):
        raise TypeError("Noah-MP BARE_FLUX inputs must be MetalArray values")
    if in_rows.ndim != 2:
        raise ValueError("BARE_FLUX input must have shape (ncase, 53)")
    n = int(in_rows.shape[0])
    for name, arr in arrays.items():
        dtype, width = expected[name]
        if arr.dtype != dtype or arr.shape != (n, width) or not arr.flags.c_contiguous:
            raise ValueError(f"BARE_FLUX {name} must be contiguous {dtype} shape {(n, width)}")
        if arr.buffer.context is not context:
            raise ValueError(f"BARE_FLUX {name} belongs to a different context")
    if not n:
        return None
    return context.submit_kernel(
        "noahmp_bare_flux", arrays, {"n": n}, exact_policy=False,
        grid_size=((n + _THREADS - 1) // _THREADS) * _THREADS,
        threads_per_group=_THREADS)


