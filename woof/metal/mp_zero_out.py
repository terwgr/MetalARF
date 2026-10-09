"""Typed Metal launcher for WRF's in-place microphysics zero-out pass."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray


def submit_mp_zero_out(field: MetalArray, *, threshold: float, mode: int,
                       spec_zone: int, first_species: bool):
    """Floor boundary values and apply WRF threshold rules on-device."""
    if not isinstance(field, MetalArray) or field.ndim != 3:
        raise TypeError("Metal mp_zero_out fields must be 3-D MetalArrays")
    if field.dtype != np.dtype(np.float32) or not field.flags.c_contiguous:
        raise TypeError("Metal mp_zero_out fields require contiguous float32")
    nz, ny, nx = map(int, field.shape)
    count = nz * ny * nx
    if min(nz, ny, nx) <= 0 or count > 0x7FFFFFFF:
        raise ValueError("Metal mp_zero_out dimensions are empty or too large")
    if mode not in (1, 2):
        raise ValueError(f"Metal mp_zero_out mode must be 1 or 2, got {mode}")
    if spec_zone < 0:
        raise ValueError("Metal mp_zero_out spec_zone must be nonnegative")
    threshold32 = np.float32(threshold)
    if not np.isfinite(threshold32):
        raise ValueError("Metal mp_zero_out threshold must be finite float32")
    context = field.buffer.context
    return context.submit_kernel(
        "mp_zero_out", {"field": field},
        {"count": count, "nz": nz, "ny": ny, "nx": nx,
         "spec_zone": int(spec_zone), "mode": int(mode),
         "first_species": int(bool(first_species)),
         "threshold": float(threshold32)},
        exact_policy=False, grid_size=count, threads_per_group=128)


__all__ = ["submit_mp_zero_out"]
