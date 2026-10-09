"""Metal launchers for the small SAXPY and gravity-constant kernels."""

from __future__ import annotations

import numpy as np

from woof.core import constants as c
from woof.metal.array import MetalArray


def submit_emit_g(out):
    """Write the pinned float32 ``G`` constant to element zero of ``out``."""
    if not isinstance(out, MetalArray) or out.dtype != np.dtype(np.float32):
        raise TypeError("Metal emit_g requires a float32 MetalArray")
    if out.size < 1:
        raise ValueError("Metal emit_g output must contain at least one element")
    scalar_out = out.view((1,), byte_offset=out.byte_offset)
    return scalar_out.buffer.context.submit_kernel(
        "emit_g", {"out": scalar_out},
        {"gravity": float(np.float32(c.G)), "n": 1},
        exact_policy=False, grid_size=1)


def submit_saxpy(a, x, y, out, n, *, block_size=256):
    """Submit the pinned CUDA SAXPY indexing for contiguous float32 buffers.

    Only the first ``n`` values participate, matching the CUDA pointer/count
    interface. The alignment flag is derived from each view's byte offset and
    assumes each Metal buffer base satisfies float4 alignment; integration
    must confirm that allocator contract.
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError("Metal saxpy n must be an integer")
    n = int(n)
    if n <= 0:  # CUDA returns before reading any pointers for n <= 0.
        return None
    if n > 0x7FFFFFFF:
        raise ValueError("Metal saxpy n must fit the CUDA int32 count")
    if (not isinstance(block_size, (int, np.integer))
            or not 0 < int(block_size) <= 1024):
        raise ValueError("Metal saxpy block_size must be in 1..1024")
    block_size = int(block_size)

    arrays = (x, y, out)
    if any(not isinstance(array, MetalArray)
           or array.dtype != np.dtype(np.float32)
           or not array.flags.c_contiguous for array in arrays):
        raise TypeError("Metal saxpy requires contiguous float32 MetalArrays")
    if any(array.size < n for array in arrays):
        raise ValueError("Metal saxpy buffers must each contain at least n values")
    context = out.buffer.context
    if any(array.buffer.context is not context for array in arrays):
        raise ValueError("Metal saxpy arrays must belong to one context")

    # Flat views preserve pointer offsets and support any contiguous source
    # shape without allocating or changing alias relationships.
    xv = x.view((n,), byte_offset=x.byte_offset)
    yv = y.view((n,), byte_offset=y.byte_offset)
    ov = out.view((n,), byte_offset=out.byte_offset)
    vector_aligned = int(all(array.byte_offset % 16 == 0
                             for array in (x, y, out)))
    grid_size = ((n + block_size * 4 - 1) // (block_size * 4)) * block_size
    if grid_size > 0xFFFFFFFF:
        raise ValueError("Metal saxpy launch grid exceeds the uint thread index")
    return context.submit_kernel(
        "saxpy", {"x": xv, "y": yv, "out": ov},
        {"a": float(np.float32(a)), "n": n,
         "block_size": block_size, "vector_aligned": vector_aligned},
        exact_policy=False, grid_size=grid_size,
        threads_per_group=block_size)
