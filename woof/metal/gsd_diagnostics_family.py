"""Typed Metal launchers for the GSD operational diagnostics."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray


def submit_gsd(name, arrays, params, grid_size):
    """Launch one GSD kernel; scalar values are rounded to their ABI type."""
    context = next(iter(arrays.values())).buffer.context
    for field, value in arrays.items():
        if (not isinstance(value, MetalArray) or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous or value.buffer.context is not context):
            raise TypeError(f"Metal GSD {name}.{field} must be contiguous float32 "
                            "on one context")
    return context.submit_kernel(
        name, arrays, {key: (int(value) if isinstance(value, (int, np.integer))
                             else float(np.float32(value))) for key, value in params.items()},
        exact_policy=False, grid_size=int(grid_size), threads_per_group=128)


def copy_inner_boundary(field):
    """WRF's x-then-y boundary copy of a 2-D plane."""
    ny, nx = map(int, field.shape)
    field.buffer.context.submit_kernel(
        "uh_boundary_copy", {"field": field}, {"ny": ny, "nx": nx, "ncol": ny * nx},
        exact_policy=True, grid_size=ny * nx, threads_per_group=128)
