"""Metal submission helper for implicit Rayleigh profile damping."""

from __future__ import annotations

import numpy as np

from woof.wrf_exact import ENABLED as WRF_EXACT


def submit_rayleigh_damp(state, field, rdamp, *, slot: str):
    """Scale a contiguous 3-D Metal field by its per-level FP32 factors.

    ``rdamp`` is produced by the existing caller's host profile calculation;
    this helper only uploads that profile and submits the in-place kernel.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal Rayleigh damping supports ordinary float32 arithmetic "
            "only; WRF exact arithmetic is not implemented")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal Rayleigh damping requires a Metal DomainState")

    from .array import MetalArray

    if not isinstance(field, MetalArray):
        raise TypeError("Metal Rayleigh damping field must be a MetalArray")
    if (field.dtype != np.dtype(np.float32)
            or not field.flags.c_contiguous):
        raise TypeError(
            "Metal Rayleigh damping requires contiguous float32 field storage")
    if not isinstance(rdamp, np.ndarray) or rdamp.dtype != np.dtype(np.float32):
        raise TypeError("Rayleigh factors must be a host float32 array")
    if rdamp.ndim != 1 or not rdamp.flags.c_contiguous:
        raise ValueError("Rayleigh factors must be a contiguous 1-D profile")
    if len(field.shape) != 3 or any(dim <= 0 for dim in field.shape):
        raise ValueError("Rayleigh field must have nonempty (nlev, nys, nxs) shape")

    nlev, nys, nxs = field.shape
    if rdamp.shape != (nlev,):
        raise ValueError(
            f"Rayleigh profile shape {rdamp.shape} does not match field levels {nlev}")
    if any(dim > 0x7FFFFFFF for dim in (nlev, nys, nxs)):
        raise ValueError("Rayleigh extents must fit positive int32 parameters")
    count = nlev * nys * nxs
    if count > 0xFFFFFFFF:
        raise ValueError("Rayleigh grid does not fit the uint grid index")

    profile = state.scratch((nlev,), slot)
    if not isinstance(profile, MetalArray):
        raise TypeError("Metal Rayleigh profile workspace is not a MetalArray")
    profile.buffer.write(rdamp.tobytes(), offset=profile.byte_offset)
    return state._array_module.context.submit_kernel(
        "rayleigh_damp",
        {"f": field, "rdamp": profile},
        {"nlev": nlev, "nys": nys, "nxs": nxs},
        exact_policy=False, grid_size=count)
