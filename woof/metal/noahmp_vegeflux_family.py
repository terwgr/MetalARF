"""Typed Metal launcher for the Noah-MP VEGE_FLUX caller slice."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray

_FLOAT = np.dtype(np.float32)
_THREADS = 64


def submit_vegeflux(context, in_rows: MetalArray, params: MetalArray,
                    isnow: MetalArray, dzsnso: MetalArray,
                    stc: MetalArray, df: MetalArray, out: MetalArray):
    """Submit one packed case per grid thread for VEGE_FLUX."""
    arrays = {"in": in_rows, "pin": params, "isnow": isnow,
              "dzsnso": dzsnso, "stc": stc, "df": df, "out": out}
    if any(not isinstance(a, MetalArray) for a in arrays.values()):
        raise TypeError("VEGE_FLUX inputs must be MetalArray values")
    if in_rows.ndim != 2:
        raise ValueError("VEGE_FLUX in must have shape (ncase, 50)")
    n = int(in_rows.shape[0])
    expected = {"in": (_FLOAT, 50), "pin": (_FLOAT, 14),
                "isnow": (np.dtype(np.int32), None),
                "dzsnso": (_FLOAT, 7), "stc": (_FLOAT, 7),
                "df": (_FLOAT, 7), "out": (_FLOAT, 30)}
    for name, array in arrays.items():
        dtype, width = expected[name]
        shape = (n,) if width is None else (n, width)
        if (array.dtype != dtype or array.shape != shape
                or not array.flags.c_contiguous):
            raise ValueError(
                f"VEGE_FLUX {name} must be contiguous {dtype} shape {shape}")
        if array.buffer.context is not context:
            raise ValueError(f"VEGE_FLUX {name} belongs to another Metal context")
    if not n:
        return None
    return context.submit_kernel(
        "noahmp_vegeflux_flux", arrays, {"n": n}, exact_policy=False,
        grid_size=((n + _THREADS - 1) // _THREADS) * _THREADS,
        threads_per_group=_THREADS)


__all__ = ["submit_vegeflux"]
