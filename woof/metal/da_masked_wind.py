"""Masked resident binary32 wind increments."""
import numpy as np

from .array import MetalArray, MetalArrayModule


def add_masked(original, increment, active):
    """Return a 3-D FP32 sum where the 2-D boolean mask is true.

    The mask broadcasts over axis 0. Inactive words are preserved exactly;
    the original array is never written.
    """
    if (not isinstance(original, MetalArray)
            or original.dtype != np.dtype(np.float32)
            or original.ndim != 3 or not original.flags.c_contiguous):
        raise TypeError("original must be a contiguous 3-D FP32 Metal array")
    increment = np.asarray(increment)
    active = np.asarray(active)
    if increment.dtype != np.dtype(np.float32) or increment.shape != original.shape:
        raise ValueError("increment must be FP32 and match the original shape")
    if active.dtype != np.dtype(bool) or active.shape != original.shape[1:]:
        raise ValueError("active must be boolean and match the horizontal shape")
    if original.size > np.iinfo(np.int32).max:
        raise ValueError("wind extent exceeds the int32 indexing ABI")
    context = original.buffer.context
    xp = MetalArrayModule(context)
    out = xp.empty(original.shape, dtype=np.float32)
    if original.size:
        delta = xp.asarray(np.ascontiguousarray(increment))
        mask = xp.asarray(np.ascontiguousarray(active, dtype=np.uint32))
        context.submit_kernel(
            "da_masked_wind_add",
            {"left": original.view((original.size,), np.uint32), "right": delta.view((delta.size,), np.uint32),
             "mask": mask.view((mask.size,)), "out": out.view((out.size,), np.uint32)},
            {"n": original.size, "plane": mask.size}, exact_policy=False,
            grid_size=original.size, threads_per_group=128)
    return out
