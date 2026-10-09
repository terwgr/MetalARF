"""Typed Metal launcher for the sun-angle land-albedo update."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_FLOAT_FIELDS = ("albedo", "albbck", "xland", "snow", "xice",
                 "albsol", "albbcksol")


def submit_solar_albedo_update(fields, coszen, *, initialize=False):
    """Apply the RUC ``alb_sol=1`` update to contiguous float32 Metal fields.

    Active MODIS classes are checked on-device so a large resident field does
    not need to be copied to the CPU on each radiation call.
    """
    arrays = {name: fields[name] for name in _FLOAT_FIELDS}
    arrays["ivgtyp"] = fields["ivgtyp"]
    arrays["coszen"] = coszen
    reference = arrays["albsol"]
    if not isinstance(reference, MetalArray) or reference.ndim not in (2, 3):
        raise TypeError("Metal solar albedo requires 2-D or 3-D surface fields")
    shape = reference.shape
    if reference.size > 0x7fffffff:
        raise ValueError("Metal solar albedo field exceeds signed int32 indexing")
    context = reference.buffer.context
    for name, array in arrays.items():
        dtype = np.dtype(np.int32 if name == "ivgtyp" else np.float32)
        if not isinstance(array, MetalArray):
            raise TypeError(f"Metal solar albedo {name} must be a MetalArray")
        if array.dtype != dtype or not array.flags.c_contiguous:
            raise TypeError(f"Metal solar albedo {name} must be contiguous {dtype}")
        if array.buffer.context is not context:
            raise ValueError("Metal solar albedo fields must share one context")
        expected = shape
        if name == "coszen" and array.shape != shape:
            if array.ndim != 1 or array.size != reference.size:
                raise ValueError(
                    f"Metal solar albedo coszen must have shape {shape} or "
                    f"a flat vector of {reference.size}, got {array.shape}")
            array = array.view(shape)
            arrays[name] = array
        elif array.shape != expected:
            raise ValueError(
                f"Metal solar albedo {name} must have shape {shape}, got "
                f"{array.shape}")

    if reference.size == 0:
        return None

    flat = {name: array.view((reference.size,))
            for name, array in arrays.items()}

    # Like CUDA's cp.any preflight, validate before touching the persistent
    # outputs. Each thread writes a distinct status word, avoiding atomics.
    status = MetalArray(context.buffer(reference.size * 4),
                       (reference.size,), np.uint32)
    context.submit_kernel(
        "solar_albedo_validate",
        {"xland": flat["xland"], "snow": flat["snow"],
         "xice": flat["xice"], "ivgtyp": flat["ivgtyp"],
         "coszen": flat["coszen"], "status": status},
        {"count": int(reference.size)}, exact_policy=False,
        grid_size=int(reference.size), threads_per_group=128)
    if bool(np.any(status.to_numpy())):
        raise ValueError("solar albedo needs MODIS IVGTYP classes 1 through 21")

    resources = {name: flat[name] for name in (
        "albedo", "albbck", "xland", "snow", "xice", "ivgtyp",
        "albsol", "albbcksol", "coszen")}
    context.submit_kernel(
        "solar_albedo_update", resources,
        {"count": int(reference.size),
         "initialize": int(bool(initialize))},
        exact_policy=False, grid_size=int(reference.size),
        threads_per_group=128)


__all__ = ["submit_solar_albedo_update"]
