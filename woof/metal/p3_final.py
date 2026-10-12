"""Typed Metal submission for P3 cleanup and ice/rain diagnostics."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from .p3_constants import P3_CONSTANT_NAMES
from woof.wrf_exact import ENABLED as WRF_EXACT


_FIELDS = ("qc", "nc", "qr", "nr", "qi", "qir", "ni", "qib",
           "theta", "qv")
_SCRATCH = ("temperature", "tmparr1", "rho", "inv_rho", "rhofaci")
_DIAGNOSTICS = ("zdbz", "effc", "effi", "vmi", "di", "rhopo")


def submit_p3_final(*, fields: dict, scratch: dict, diagnostics: dict,
                    itab, flags, constants, xxlv, xxls) -> None:
    """Run P3's post-freezing cleanup and diagnostics on Metal.

    The P3 ITAB is a contiguous flat FP32 view with 14,000 entries in source
    ``(densize,rimsize,isize,tabsize)`` order. All prognostic and scratch
    volume fields are contiguous FP32 ``(nz,ny,nx)`` arrays; flags are
    ``(2,ny,nx)``. Caller owns the table and every state/scratch allocation.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "P3 Metal final stage supports ordinary float32 only; source "
            "r_pow behavior is not implemented")
    missing = ([n for n in _FIELDS if n not in fields]
               + [n for n in _SCRATCH if n not in scratch]
               + [n for n in _DIAGNOSTICS if n not in diagnostics])
    if missing:
        raise ValueError("P3 Metal final stage is missing arrays: "
                         + ", ".join(missing))
    theta = fields["theta"]
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("P3 Metal final theta must be a 3-D MetalArray")
    nz, ny, nx = theta.shape
    if min(nz, ny, nx) < 1:
        raise ValueError("P3 Metal final dimensions must be nonempty")
    context = theta.buffer.context
    arrays = {}
    for name in _FIELDS:
        value = fields[name]
        if (not isinstance(value, MetalArray) or value.shape != (nz, ny, nx)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"P3 Metal final {name} must be contiguous FP32 {(nz, ny, nx)}")
        if value.buffer.context is not context:
            raise ValueError(f"P3 Metal final {name} belongs to another context")
        arrays[name] = value
    for name in _SCRATCH:
        value = scratch[name]
        if (not isinstance(value, MetalArray) or value.shape != (nz, ny, nx)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"P3 Metal final {name} must be contiguous FP32 {(nz, ny, nx)}")
        if value.buffer.context is not context:
            raise ValueError(f"P3 Metal final {name} belongs to another context")
        arrays[name] = value
    for name in _DIAGNOSTICS:
        value = diagnostics[name]
        if (not isinstance(value, MetalArray) or value.shape != (nz, ny, nx)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"P3 Metal final {name} must be contiguous FP32 {(nz, ny, nx)}")
        if value.buffer.context is not context:
            raise ValueError(f"P3 Metal final {name} belongs to another context")
        arrays[name] = value
    if (not isinstance(itab, MetalArray) or itab.shape != (14000,)
            or itab.dtype != np.dtype(np.float32)
            or not itab.flags.c_contiguous):
        raise TypeError("P3 Metal final ITAB must be contiguous FP32 (14000,)")
    if itab.buffer.context is not context:
        raise ValueError("P3 Metal final ITAB belongs to another context")
    if (not isinstance(flags, MetalArray) or flags.shape != (2, ny, nx)
            or flags.dtype != np.dtype(np.float32)
            or not flags.flags.c_contiguous):
        raise TypeError(f"P3 Metal final flags must be contiguous FP32 {(2, ny, nx)}")
    if flags.buffer.context is not context:
        raise ValueError("P3 Metal final flags belong to another context")
    arrays["itab"] = itab
    arrays["flags"] = flags
    if (not isinstance(constants, MetalArray)
            or constants.shape != (len(P3_CONSTANT_NAMES),)
            or constants.dtype != np.dtype(np.float32)
            or not constants.flags.c_contiguous):
        raise TypeError(
            "P3 Metal final constants must be contiguous FP32 "
            f"({len(P3_CONSTANT_NAMES)},)")
    if constants.buffer.context is not context:
        raise ValueError("P3 Metal final constants belong to another context")
    arrays["constants"] = constants
    latent = tuple(np.float32(value) for value in (xxlv, xxls))
    if not all(np.isfinite(value) for value in latent):
        raise ValueError("P3 Metal final latent heats must be finite FP32")
    context.submit_kernel(
        "p3_final", arrays,
        {"nz": nz, "ny": ny, "nx": nx,
         "xxlv": float(latent[0]), "xxls": float(latent[1])},
        exact_policy=False, grid_size=ny * nx)


__all__ = ["submit_p3_final"]
