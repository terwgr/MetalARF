"""Typed Metal submission for classic Thompson graupel fallout with shadow number."""
from __future__ import annotations

import numpy as np

from .array import MetalArray


def _overlap(a: MetalArray, b: MetalArray) -> bool:
    return (a.buffer is b.buffer
            and a.byte_offset < b.byte_offset + b.nbytes
            and b.byte_offset < a.byte_offset + a.nbytes)


def submit_thompson_graupel_sedimentation(
        *, qg, temperature, pressure, qv, reference_density, dz,
        rainnc, rainncv, graupelnc, graupelncv, active_columns,
        graupel_number_shadow, dt, accumulate_surface=False):
    """Submit the guarded classic-graupel fallout operation.

    Resource ordering and alias rejection mirror the CUDA restrict-qualified
    kernel. ``active_columns`` preserves WRF's held column gate; the private
    number shadow receives its independent sedimentation tendency.
    """
    if not isinstance(qg, MetalArray) or qg.ndim != 3:
        raise TypeError("Metal Thompson graupel qg must be a 3-D MetalArray")
    shape = tuple(qg.shape)
    nz, ny, nx = shape
    if nz < 2 or nz > 256 or ny <= 0 or nx <= 0:
        raise ValueError(f"Thompson graupel sedimentation requires 2<=nz<=256 and nonempty horizontal dimensions, got {shape}")
    if ny * nx > 0x7fffffff or nz * ny * nx > 0x7fffffff:
        raise ValueError("Thompson graupel dimensions exceed the int32 ABI")
    if not isinstance(accumulate_surface, (bool, np.bool_)):
        raise TypeError("accumulate_surface must be boolean")
    arrays = {
        "qg": qg, "temperature": temperature, "pressure": pressure,
        "qv": qv, "reference_density": reference_density, "dz": dz,
        "rainnc": rainnc, "rainncv": rainncv,
        "graupelnc": graupelnc, "graupelncv": graupelncv,
        "active_columns": active_columns,
        "graupel_number_shadow": graupel_number_shadow,
    }
    surface = {"rainnc", "rainncv", "graupelnc", "graupelncv", "active_columns"}
    context = qg.buffer.context
    for name, value in arrays.items():
        if not isinstance(value, MetalArray):
            raise TypeError(f"Metal Thompson graupel {name} must be a MetalArray")
        expected = (ny, nx) if name in surface else shape
        if (value.shape != expected or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"{name} must be contiguous float32 with shape {expected}")
        if value.buffer.context is not context:
            raise ValueError("Thompson graupel resources must share one Metal context")
    names = tuple(arrays)
    for pos, left_name in enumerate(names):
        for right_name in names[pos + 1:]:
            if _overlap(arrays[left_name], arrays[right_name]):
                raise ValueError(f"restricted Thompson graupel resources overlap: {left_name}, {right_name}")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(f"Thompson graupel sedimentation dt must be finite positive float32, got {dt!r}")
    return context.submit_kernel(
        "thompson_graupel_sediment_with_density_column_mask_shadow", arrays,
        {"accumulate_surface": int(accumulate_surface), "dt": float(dt32),
         "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=32)
