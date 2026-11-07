"""Pinned aerosol Exner arithmetic on Metal device storage."""
from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_aerosol_exner(pressure, pii_out):
    """Form the current aerosol source's REAL(4) Exner word."""
    for name, value in (("pressure", pressure), ("pii_out", pii_out)):
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"aerosol Exner {name} requires contiguous float32 Metal storage")
    if pressure.shape != pii_out.shape:
        raise ValueError("aerosol Exner input and output shapes differ")
    context = pressure.buffer.context
    if pii_out.buffer.context is not context:
        raise ValueError("aerosol Exner arrays must share one Metal context")
    if pressure.buffer.handle == pii_out.buffer.handle:
        left, right = pressure.byte_offset, pii_out.byte_offset
        if left < right + pii_out.nbytes and right < left + pressure.nbytes:
            raise ValueError("aerosol Exner input and output storage overlaps")
    n = pressure.size
    if n > np.iinfo(np.int32).max:
        raise ValueError("aerosol Exner extent exceeds its int32 ABI")
    if not n:
        return None
    return context.submit_kernel(
        "thompson_aa_exner",
        {"pressure": pressure.view((n,)), "pii_out": pii_out.view((n,))},
        {"n": n}, exact_policy=False, grid_size=n, threads_per_group=128)
