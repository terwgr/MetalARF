"""Column-owned Metal submission for native NSSL sedimentation."""

from __future__ import annotations

import math
import numpy as np

from woof.metal.array import MetalArray


_TPB = 32


def submit_nssl2_sediment_all(
        context, air_density, temperature_k, state, dz,
        ignored_accumulator, category_export, dt_s, *, nz, ny, nx):
    """Run all six default sedimentation categories in source order.

    The kernel uses one lane per horizontal column and runs the admitted
    single-column WRF sedimentation routines in order. Category exports are
    overwritten, while the ignored receipt is cleared before its staged use.
    """
    dims = (nz, ny, nx)
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer)) or value < 1
           for value in dims):
        raise ValueError("NSSL sedimentation dimensions must be positive integers")
    nz, ny, nx = map(int, dims)
    if nz < 2 or nz > 256:
        raise ValueError("NSSL sedimentation requires 2 <= nz <= 256")
    if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise ValueError("NSSL dt_s must be positive and finite")
    dt = np.float32(dt_s)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("NSSL dt_s must be representable as positive float32")
    expected = (nz, ny, nx)
    for name, value in (("air_density", air_density),
                        ("temperature_k", temperature_k), ("dz", dz)):
        if not isinstance(value, MetalArray):
            raise TypeError(f"{name} must be a MetalArray")
        if value.shape != expected or value.dtype != np.dtype(np.float32):
            raise ValueError(f"{name} must be FP32 with shape {expected}")
        if value.buffer.context is not context or not value.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous in the submitting context")
    if not isinstance(state, MetalArray) or state.shape != (16, *expected):
        raise ValueError("state must have shape (16, nz, ny, nx)")
    if state.dtype != np.dtype(np.float32) or state.buffer.context is not context:
        raise ValueError("state must be FP32 storage in the submitting context")
    if not isinstance(ignored_accumulator, MetalArray) or ignored_accumulator.shape != (ny, nx):
        raise ValueError("ignored_accumulator must have shape (ny, nx)")
    if not isinstance(category_export, MetalArray) or category_export.shape != (5, ny, nx):
        raise ValueError("category_export must have shape (5, ny, nx)")
    for name, value in (("ignored_accumulator", ignored_accumulator),
                        ("category_export", category_export)):
        if value.dtype != np.dtype(np.float32) or value.buffer.context is not context:
            raise ValueError(f"{name} must be FP32 storage in the submitting context")
    ignored_accumulator.fill(0.0)
    kernel = "nssl2_sediment_all_64" if nz <= 64 else "nssl2_sediment_all_256"
    return context.submit_kernel(
        kernel,
        {"air_density": air_density, "temperature_k": temperature_k,
         "state": state, "dz": dz,
         "ignored_accumulator": ignored_accumulator,
         "category_export": category_export},
        {"dt": dt, "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=_TPB)


__all__ = ["submit_nssl2_sediment_all"]
