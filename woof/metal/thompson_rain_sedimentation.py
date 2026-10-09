"""Typed Metal launcher for classic Thompson held-density rain fallout."""
from __future__ import annotations

import numpy as np

from .array import MetalArray


def _overlap(a: MetalArray, b: MetalArray) -> bool:
    return (a.buffer is b.buffer
            and a.byte_offset < b.byte_offset + b.nbytes
            and b.byte_offset < a.byte_offset + a.nbytes)


def submit_thompson_rain_sediment_with_density_and_presence(
        *, qr, nr, temperature, pressure, qv, reference_density, dz,
        rainnc, rainncv, dt, accumulate_surface=False):
    """Submit Thompson's source-selected rain fallout operation.

    ``reference_density`` is the held rain density from the evaporation
    source. Zero marks failed ``L_qr``; a negative value marks a source
    rewrite. The shader retains that ABI while updating the current ``qr``
    and ``nr`` arrays and both surface precipitation accumulators.
    """
    if not isinstance(qr, MetalArray) or qr.ndim != 3:
        raise TypeError("Metal Thompson rain qr must be a 3-D MetalArray")
    shape = tuple(qr.shape)
    nz, ny, nx = shape
    if nz < 2 or nz > 256 or ny <= 0 or nx <= 0:
        raise ValueError(f"Thompson rain sedimentation requires 2<=nz<=256 and nonempty horizontal dimensions, got {shape}")
    if not isinstance(accumulate_surface, (bool, np.bool_)):
        raise TypeError("accumulate_surface must be boolean")
    arrays = {
        "qr": qr, "nr": nr, "temperature": temperature,
        "pressure": pressure, "qv": qv,
        "reference_density": reference_density, "dz": dz,
        "rainnc": rainnc, "rainncv": rainncv,
    }
    context = qr.buffer.context
    for name, value in arrays.items():
        if not isinstance(value, MetalArray):
            raise TypeError(f"Metal Thompson rain {name} must be a MetalArray")
        expected_shape = (ny, nx) if name in ("rainnc", "rainncv") else shape
        if (value.shape != expected_shape or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"{name} must be contiguous float32 with shape {expected_shape}")
        if value.buffer.context is not context:
            raise ValueError("Thompson rain resources must share one Metal context")
    names = tuple(arrays)
    for pos, left_name in enumerate(names):
        for right_name in names[pos + 1:]:
            if _overlap(arrays[left_name], arrays[right_name]):
                raise ValueError(f"restricted Thompson rain resources overlap: {left_name}, {right_name}")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(f"Thompson rain sedimentation dt must be finite positive float32, got {dt!r}")
    return context.submit_kernel(
        "thompson_rain_sediment_with_density_and_presence", arrays,
        {"accumulate_surface": int(accumulate_surface), "dt": float(dt32),
         "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=32)
