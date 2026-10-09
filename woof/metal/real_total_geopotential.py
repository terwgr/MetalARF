"""Metal launcher for real-initialization total geopotential."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_total_geopotential(state) -> MetalArray:
    """Compute ``phb + php`` into a new FP32 Metal array.

    The source expression is ordinary FP32 addition with NumPy/CuPy
    broadcasting: WRF's one-dimensional base profile may be added to the
    three-dimensional perturbation field, or the base field may already be
    fully expanded. This routine keeps the addition on the device and leaves
    the caller to perform any explicitly requested setup readback.
    """
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("total geopotential Metal submission requires a Metal state")
    xp = getattr(state, "_array_module", None)
    context = getattr(xp, "context", None)
    if context is None:
        raise TypeError("Metal state has no Metal array context")

    phb = getattr(state, "phb", None)
    php = getattr(state, "php", None)
    if not isinstance(php, MetalArray) or php.ndim != 3:
        raise TypeError("Metal php must be a contiguous three-dimensional FP32 array")
    nz1, ny, nx = php.shape
    if nz1 < 1 or ny < 1 or nx < 1:
        raise ValueError("total geopotential requires nonempty dimensions")
    if php.dtype != np.dtype(np.float32) or not php.flags.c_contiguous:
        raise TypeError("Metal php must be contiguous float32 storage")
    if php.buffer.context is not context:
        raise ValueError("Metal php belongs to a different context")
    profile_shape = (nz1,)
    full_shape = (nz1, ny, nx)
    if (not isinstance(phb, MetalArray)
            or phb.dtype != np.dtype(np.float32)
            or phb.shape not in (profile_shape, full_shape)
            or not phb.flags.c_contiguous):
        raise TypeError(
            "Metal phb must be contiguous float32 storage with shape "
            f"{profile_shape} or {full_shape}")
    if phb.buffer.context is not context:
        raise ValueError("Metal phb belongs to a different context")
    nz = nz1 - 1
    n = php.size
    if n > 2_147_483_647 or ny * nx > 2_147_483_647:
        raise ValueError("total geopotential dimensions exceed the Metal ABI")
    total_phi = xp.zeros(full_shape, dtype=np.float32)
    context.submit_kernel(
        "real_total_geopotential",
        {"phb": phb, "php": php, "total_phi": total_phi},
        {"nz": nz, "ny": ny, "nx": nx,
         "phb_3d": int(phb.shape == full_shape)},
        exact_policy=False, grid_size=n)
    return total_phi
