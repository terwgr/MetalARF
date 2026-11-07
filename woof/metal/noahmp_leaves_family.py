"""Metal launcher for the independently supportable Noah-MP ATM leaf."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_leaf_atm(xs: MetalArray, ys: MetalArray) -> MetalArray:
    """Run the packed 11-to-17-slot ATM leaf on Metal.

    Native MSL ``pow``/``exp`` serve ATM's r_pow/r_exp operations under the
    default Metal arithmetic policy.
    """
    if not isinstance(xs, MetalArray) or not isinstance(ys, MetalArray):
        raise TypeError("Noah-MP ATM requires MetalArray inputs and output")
    if (xs.dtype != np.dtype(np.float32) or ys.dtype != np.dtype(np.float32)
            or not xs.flags.c_contiguous or not ys.flags.c_contiguous):
        raise TypeError("Noah-MP ATM requires contiguous float32 arrays")
    if xs.ndim != 2 or xs.shape[1] != 11:
        raise ValueError("Noah-MP ATM input must have shape (ncase, 11)")
    ncase = int(xs.shape[0])
    if ys.shape != (ncase, 17):
        raise ValueError(f"Noah-MP ATM output must have shape {(ncase, 17)}")
    if xs.buffer.context is not ys.buffer.context:
        raise ValueError("Noah-MP ATM arrays must share one Metal context")
    if ncase:
        xs.buffer.context.submit_kernel(
            "noahmp_leaf_atm", {"xs": xs, "ys": ys},
            {"ncase": ncase}, exact_policy=False,
            grid_size=((ncase + 63) // 64) * 64, threads_per_group=64)
    return ys


