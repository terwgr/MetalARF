"""Typed Metal submission for classic Thompson's production rain evaporation."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def _overlap(left: MetalArray, right: MetalArray) -> bool:
    if left.buffer is not right.buffer:
        return False
    a0, a1 = left.byte_offset, left.byte_offset + left.nbytes
    b0, b1 = right.byte_offset, right.byte_offset + right.nbytes
    return a0 < b1 and b0 < a1


def submit_thompson_rain_evaporation_with_density_history_and_presence(
        *, qr, nr, temperature, pressure, qv, reference_density,
        graupel_melt_marker, source_density, condensation_marker, dt):
    """Submit the output-presence history form used by classic mp=8.

    The three held fields and the density output are separate live buffers in
    the CUDA contract. Reject overlapping Metal views before submission too.
    """
    arrays = {
        "qr": qr, "nr": nr, "temperature": temperature,
        "pressure": pressure, "qv": qv,
        "reference_density": reference_density,
        "graupel_melt_marker": graupel_melt_marker,
        "source_density": source_density,
        "condensation_marker": condensation_marker,
    }
    if not isinstance(temperature, MetalArray) or temperature.ndim != 3:
        raise TypeError("Metal Thompson rain evaporation temperature must be a 3-D MetalArray")
    shape = tuple(temperature.shape)
    nz, ny, nx = shape
    if min(shape) <= 0:
        raise ValueError("Metal Thompson rain evaporation requires nonempty dimensions")
    if nz > 256:
        raise ValueError(f"Metal Thompson rain evaporation supports nz <= 256, got {nz}")
    if int(np.prod(shape, dtype=np.int64)) > 0x7FFFFFFF:
        raise ValueError("Metal Thompson rain evaporation size exceeds the int32 ABI")
    for name, value in arrays.items():
        if not isinstance(value, MetalArray):
            raise TypeError(f"Metal Thompson rain evaporation {name} must be a MetalArray")
        if (value.shape != shape or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"Metal Thompson rain evaporation {name} must be matching contiguous FP32 storage")
    context = temperature.buffer.context
    if any(value.buffer.context is not context for value in arrays.values()):
        raise ValueError("Metal Thompson rain evaporation fields must share one context")
    names = tuple(arrays)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            if _overlap(arrays[left_name], arrays[right_name]):
                raise ValueError(
                    f"Metal Thompson rain evaporation {left_name} must not alias {right_name}")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(f"Metal Thompson rain evaporation dt must be finite and positive, got {dt!r}")
    return context.submit_kernel(
        "thompson_rain_evaporation_with_density_history_and_presence",
        arrays, {"dt": float(dt32), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=int(np.prod(shape, dtype=np.int64)),
        threads_per_group=256)
