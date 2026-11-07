"""Pinned classic aerosol graupel entry-number arithmetic on Metal."""
from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_graupel_number_init(qg, temperature, pressure, qv, graupel_number):
    """Diagnose the private entry ng1d using the driver's raw vapour."""
    values = {"qg": qg, "temperature": temperature, "pressure": pressure,
              "qv": qv, "graupel_number": graupel_number}
    if not isinstance(qg, MetalArray):
        raise TypeError("graupel entry number requires Metal storage")
    context, shape = qg.buffer.context, qg.shape
    for name, value in values.items():
        if (not isinstance(value, MetalArray) or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous or value.shape != shape):
            raise TypeError(f"graupel entry {name} must be matching contiguous float32 storage")
        if value.buffer.context is not context:
            raise ValueError("graupel entry fields must share one Metal context")
    for source in (qg, temperature, pressure, qv):
        if (source.buffer.handle == graupel_number.buffer.handle
                and source.byte_offset < graupel_number.byte_offset + graupel_number.nbytes
                and graupel_number.byte_offset < source.byte_offset + source.nbytes):
            raise ValueError("graupel entry output must not overlap its inputs")
    n = qg.size
    if n > np.iinfo(np.int32).max:
        raise ValueError("graupel entry extent exceeds signed int32 indexing")
    if not n:
        return None
    return context.submit_kernel(
        "thompson_aa_graupel_number_init",
        {name: value.view((n,), np.uint32) for name, value in values.items()},
        {"n": n}, exact_policy=False, grid_size=n, threads_per_group=128)
