"""Typed launcher for classic Thompson held-density cloud fallout."""
from __future__ import annotations

import numpy as np

from .array import MetalArray


def _overlap(a: MetalArray, b: MetalArray) -> bool:
    if a.buffer is not b.buffer:
        return False
    return (a.byte_offset < b.byte_offset + b.nbytes
            and b.byte_offset < a.byte_offset + a.nbytes)


def submit_thompson_cloud_sedimentation_with_density_and_masks(
        *, qc, temperature, pressure, qv, reference_density,
        rain_active_columns, cloud_active_columns, vertical_velocity, dz, dt):
    """Submit the caller's exact held-density and entry-mask fallout path.

    A single Metal entry handles the complete source contract (2<=nz<=256);
    CUDA's 64-level and generic 256-level launches share this scalar column
    operation graph.
    """
    if not isinstance(qc, MetalArray) or qc.ndim != 3:
        raise TypeError("Metal Thompson cloud qc must be a 3-D MetalArray")
    shape = tuple(qc.shape)
    nz, ny, nx = shape
    if nz < 2 or nz > 256 or ny <= 0 or nx <= 0:
        raise ValueError(f"Thompson cloud sedimentation requires 2<=nz<=256 and nonempty horizontal dimensions, got {shape}")
    n = int(np.prod(shape, dtype=np.int64))
    if n > 0x7fffffff:
        raise ValueError("Thompson cloud sedimentation indexing exceeds int32")
    surface_shape = (ny, nx)
    arrays = {
        "qc": qc,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "reference_density": reference_density,
        "rain_active_columns": rain_active_columns,
        "cloud_active_columns": cloud_active_columns,
        "vertical_velocity": vertical_velocity,
        "dz": dz,
    }
    context = qc.buffer.context
    for name, value in arrays.items():
        if not isinstance(value, MetalArray):
            raise TypeError(f"Metal Thompson cloud sedimentation {name} must be a MetalArray")
        expected_shape = surface_shape if name.endswith("_columns") else shape
        if (value.shape != expected_shape or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"{name} must be contiguous float32 with shape {expected_shape}")
        if value.buffer.context is not context:
            raise ValueError("Thompson cloud sedimentation resources must share one Metal context")
    names = tuple(arrays)
    for pos, left_name in enumerate(names):
        for right_name in names[pos + 1:]:
            if _overlap(arrays[left_name], arrays[right_name]):
                raise ValueError(f"restricted Thompson cloud resources overlap: {left_name}, {right_name}")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(f"Thompson cloud sedimentation dt must be finite positive float32, got {dt!r}")
    return context.submit_kernel(
        "thompson_cloud_sediment_with_density_and_masks", arrays,
        {"dt": float(dt32), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=32)
