"""Typed launcher for classic mp=8 Thompson radiative effective radii."""
from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_thompson_effective_radius(
        *, temperature, pressure, qv, qc, qi, ni, qs, effc, effi, effs):
    """Diagnose Thompson cloud, ice, and snow radii into caller outputs."""
    arrays = {"temperature": temperature, "pressure": pressure, "qv": qv,
              "qc": qc, "qi": qi, "ni": ni, "qs": qs,
              "effc": effc, "effi": effi, "effs": effs}
    first = temperature
    if not isinstance(first, MetalArray) or first.ndim not in (1, 2, 3):
        raise TypeError("Thompson effective-radius fields must be MetalArrays")
    shape = tuple(map(int, first.shape))
    n = int(np.prod(shape, dtype=np.int64))
    if not shape or min(shape) <= 0 or n > 0x7fffffff:
        raise ValueError(f"Thompson effective-radius dimensions are invalid: {shape}")
    context = first.buffer.context
    for name, value in arrays.items():
        if not isinstance(value, MetalArray):
            raise TypeError(f"{name} must be a MetalArray")
        if (value.shape != shape or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"{name} must be contiguous float32 with shape {shape}")
        if value.buffer.context is not context:
            raise ValueError("Thompson effective-radius fields must share one Metal context")
    names = tuple(arrays)
    for pos, name in enumerate(names):
        left = arrays[name]
        for other_name in names[pos + 1:]:
            right = arrays[other_name]
            if (left.buffer is right.buffer
                    and left.byte_offset < right.byte_offset + right.nbytes
                    and right.byte_offset < left.byte_offset + left.nbytes):
                raise ValueError(f"restricted Thompson radius resources overlap: {name}, {other_name}")
    flat_arrays = {name: value.view((value.size,))
                   for name, value in arrays.items()}
    return context.submit_kernel(
        "thompson_effective_radius", flat_arrays, {"n": n}, exact_policy=False,
        grid_size=n, threads_per_group=256)


__all__ = ["submit_thompson_effective_radius"]
