"""Metal launcher for the initialized real-case surface diagnostics."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from .runtime import MetalContext


def submit_surface_diagnostics_seed(
        context: MetalContext, *, t2: MetalArray, psfc: MetalArray,
        u10_staggered: MetalArray, v10_staggered: MetalArray,
        th2: MetalArray, u10: MetalArray, v10: MetalArray) -> None:
    """Seed mass-grid TH2/U10/V10 from analyzed fields on Metal.

    T2, PSFC, and the staggered 10 m winds are read-only inputs; outputs are
    written by one thread per mass-grid column. This is a device operation,
    with no host-side numerical substitute.
    """
    if not isinstance(context, MetalContext):
        raise TypeError("surface diagnostic seeding requires a MetalContext")
    if not isinstance(t2, MetalArray) or t2.ndim != 2:
        raise TypeError("T2 must be a 2-D MetalArray")
    ny, nx = t2.shape
    expected = {
        "psfc": (psfc, (ny, nx)),
        "u10_staggered": (u10_staggered, (ny, nx + 1)),
        "v10_staggered": (v10_staggered, (ny + 1, nx)),
        "th2": (th2, (ny, nx)),
        "u10": (u10, (ny, nx)),
        "v10": (v10, (ny, nx)),
    }
    arrays = {"t2": t2}
    for name, (value, shape) in expected.items():
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or value.shape != shape
                or not value.flags.c_contiguous):
            raise TypeError(
                f"{name} must be contiguous float32 MetalArray with shape "
                f"{shape}")
        arrays[name] = value
    for name, value in arrays.items():
        if (value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"{name} must be contiguous float32")
        if value.buffer.context is not context:
            raise ValueError(f"{name} belongs to a different Metal context")
    if not nx or not ny:
        return
    context.submit_kernel(
        "surface_diagnostics_seed", arrays,
        {"nx": nx, "ny": ny, "n": nx * ny},
        exact_policy=False, grid_size=nx * ny)
