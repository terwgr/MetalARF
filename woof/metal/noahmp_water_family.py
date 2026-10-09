"""Typed launcher for the composed Noah-MP WATER Metal kernel."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray

_F32 = np.dtype(np.float32)
_I32 = np.dtype(np.int32)
_THREADS = 64


def submit_water(context, par: MetalArray, fin: MetalArray, iin: MetalArray,
                 fout: MetalArray, iout: MetalArray):
    """Submit one complete WATER column per Metal grid thread."""
    arrays = {"par": par, "fin": fin, "iin": iin, "fout": fout,
              "iout": iout}
    if any(not isinstance(a, MetalArray) for a in arrays.values()):
        raise TypeError("WATER inputs must be MetalArray values")
    if par.ndim != 2:
        raise ValueError("WATER par must have shape (ncase, 26)")
    n = int(par.shape[0])
    expected = {"par": (_F32, (n, 26)), "fin": (_F32, (n, 76)),
                "iin": (_I32, (n, 9)), "fout": (_F32, (n, 73)),
                "iout": (_I32, (n,))}
    for name, array in arrays.items():
        dtype, shape = expected[name]
        if (array.dtype != dtype or array.shape != shape
                or not array.flags.c_contiguous):
            raise ValueError(
                f"WATER {name} must be contiguous {dtype} shape {shape}")
        if array.buffer.context is not context:
            raise ValueError(f"WATER {name} belongs to another Metal context")
    if not n:
        return None
    return context.submit_kernel(
        "noahmp_water_batch", arrays, {"n": n}, exact_policy=False,
        grid_size=((n + _THREADS - 1) // _THREADS) * _THREADS,
        threads_per_group=_THREADS)


def submit_water_expf(*args, **kwargs):
    """Standalone glibc expf is not part of the public caller ABI."""
    raise NotImplementedError("standalone WATER expf is not registered")


def submit_water_powf(*args, **kwargs):
    """Standalone glibc powf is not part of the public caller ABI."""
    raise NotImplementedError("standalone WATER powf is not registered")


__all__ = ["submit_water", "submit_water_expf", "submit_water_powf"]
