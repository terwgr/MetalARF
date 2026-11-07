"""Resident binary32 arithmetic at the analysis/forecast boundary."""
from __future__ import annotations

import numpy as np

from .array import MetalArray, MetalArrayModule


def binary(left, right, *, subtract=False):
    """Return an independently owned FP32 sum or difference, rounded once.

    The integer implementation preserves gradual underflow. Neither operand
    is written, so a caller can validate every staged result before committing
    an analysis to its live state.
    """
    for value in (left, right):
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError("Metal analysis arithmetic needs contiguous FP32 operands")
    context = left.buffer.context
    if right.buffer.context is not context:
        raise ValueError("Metal analysis operands must share one context")
    if left.shape != right.shape:
        raise ValueError("Metal analysis operand shapes differ")
    if left.size > np.iinfo(np.int32).max:
        raise ValueError("Metal analysis extent exceeds the int32 indexing ABI")
    out = MetalArrayModule(context).empty(left.shape, dtype=np.float32)
    if left.size:
        context.submit_kernel(
            "da_state_binary",
            {"left": left.view((left.size,)), "right": right.view((right.size,)),
             "out": out.view((out.size,))},
            {"n": left.size, "subtract": int(subtract)},
            exact_policy=False, grid_size=left.size, threads_per_group=128)
    return out
