"""Typed Metal submission for Thompson snow fallout with melt-rain state."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def _overlap(left: MetalArray, right: MetalArray) -> bool:
    if left.buffer is not right.buffer:
        return False
    left_start = left.byte_offset
    right_start = right.byte_offset
    return (left_start < right_start + right.nbytes
            and right_start < left_start + left.nbytes)


def submit_thompson_snow_sedimentation_with_rain_profile_and_presence(
        *, qs, snow_melt_marker, melt_rain_qr, melt_rain_nr,
        melt_rain_density, temperature, pressure, qv, reference_density,
        reference_temperature, velocity_boost, dz, rainnc, rainncv,
        snownc, snowncv, dt, accumulate_surface=False):
    """Submit the complete held-state, melting-snow fallout operation.

    ``melt_rain_density`` carries the rain fall pass's reference density and
    rain-presence decision: zero means no source-selected rain, while a
    negative value marks a rewritten rain pair. All array resources are
    restrict-qualified in the CUDA implementation, so overlapping views are
    rejected even when they refer to different logical fields.
    """
    if not isinstance(qs, MetalArray) or qs.ndim != 3:
        raise TypeError("Metal Thompson snow qs must be a 3-D MetalArray")
    shape = tuple(qs.shape)
    nz, ny, nx = shape
    if nz < 2 or nz > 256 or ny <= 0 or nx <= 0:
        raise ValueError(
            "Thompson snow sedimentation requires 2<=nz<=256 and "
            f"nonempty horizontal dimensions, got {shape}")
    if ny * nx > 0x7FFFFFFF or nz * ny * nx > 0x7FFFFFFF:
        raise ValueError("Thompson snow sedimentation dimensions exceed int32 ABI")
    if not isinstance(accumulate_surface, (bool, np.bool_)):
        raise TypeError("accumulate_surface must be boolean")

    arrays = {
        "qs": qs,
        "snow_melt_marker": snow_melt_marker,
        "melt_rain_qr": melt_rain_qr,
        "melt_rain_nr": melt_rain_nr,
        "melt_rain_density": melt_rain_density,
        "temperature": temperature,
        "pressure": pressure,
        "qv": qv,
        "reference_density": reference_density,
        "reference_temperature": reference_temperature,
        "velocity_boost": velocity_boost,
        "dz": dz,
        "rainnc": rainnc,
        "rainncv": rainncv,
        "snownc": snownc,
        "snowncv": snowncv,
    }
    context = qs.buffer.context
    surface_shape = (ny, nx)
    surface_names = {"rainnc", "rainncv", "snownc", "snowncv"}
    for name, value in arrays.items():
        if not isinstance(value, MetalArray):
            raise TypeError(f"Metal Thompson snow {name} must be a MetalArray")
        expected_shape = surface_shape if name in surface_names else shape
        if (value.shape != expected_shape
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"{name} must be contiguous float32 with shape {expected_shape}")
        if value.buffer.context is not context:
            raise ValueError("Thompson snow resources must share one Metal context")

    names = tuple(arrays)
    for pos, left_name in enumerate(names):
        for right_name in names[pos + 1:]:
            if _overlap(arrays[left_name], arrays[right_name]):
                raise ValueError(
                    "restricted Thompson snow resources overlap: "
                    f"{left_name}, {right_name}")

    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(
            f"Thompson snow sedimentation dt must be finite positive float32, got {dt!r}")
    return context.submit_kernel(
        "thompson_snow_sediment_with_rain_profile_and_presence", arrays,
        {"accumulate_surface": int(accumulate_surface), "dt": float(dt32),
         "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=32)
