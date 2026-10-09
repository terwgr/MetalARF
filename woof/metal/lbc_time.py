"""Metal launcher for supported lateral-boundary time evaluation."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_linear_boundary(state, value, tendency, out_value, out_tendency,
                           seconds):
    """Evaluate one resident FP32 linear boundary side at a fresh clock."""
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal boundary evaluation requires a Metal DomainState")

    arrays = (value, tendency, out_value, out_tendency)
    if any(not isinstance(array, MetalArray) for array in arrays):
        raise TypeError("Metal boundary evaluation requires MetalArray resources")
    if any(array.dtype != np.dtype(np.float32) for array in arrays):
        raise TypeError("Metal boundary evaluation requires FP32 resources")
    count = value.size
    if count == 0:
        return None
    if any(array.size != count for array in arrays[1:]):
        raise ValueError("boundary evaluation arrays must have equal size")

    # The CUDA API accepts arbitrary same-shaped contiguous side tables. The
    # Metal signature uses a one-dimensional count, so submit no-copy views
    # over each immutable input and resident output allocation.
    flat = {name: array.view((count,)) for name, array in zip(
        ("value", "tendency", "out_value", "out_tendency"), arrays)}
    return state._array_module.context.submit_kernel(
        "evaluate_linear_boundary", flat,
        {"seconds": float(np.float32(seconds)), "count": int(count)},
        exact_policy=False, grid_size=count)
