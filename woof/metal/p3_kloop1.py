"""Typed Metal submission for P3's dry-air clipping / k_loop_1 stage."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


_FIELD_RESOURCES = (
    "qc", "nc", "qr", "nr", "qi", "qir", "ni", "qib", "theta",
    "qv", "th_old", "qv_old", "ssat", "pressure",
)
_SCRATCH_RESOURCES = (
    "temperature", "tmparr1", "rho", "inv_rho", "qvs", "qvi",
    "sup", "supi", "rhofacr", "rhofaci", "acn", "qv_cld",
)


def submit_p3_kloop1(*, fields: dict, scratch: dict, flags,
                     log_predict_nc: bool = False) -> None:
    """Run one column-serial P3 k_loop_1 stage on Metal arrays.

    ``fields`` maps P3 field names to level-major FP32 ``(nz,ny,nx)`` arrays;
    ``scratch`` supplies the two entry carriers and ten stage outputs named in
    :data:`_SCRATCH_RESOURCES`. ``flags`` is the entry stage's FP32
    ``(2,ny,nx)`` allocation. The first and second flag planes are accumulated
    in place for possible nucleation and retained hydrometeors.

    MSL's native pow/log10 are ordinary-mode stand-ins for P3's pinned
    ``r_pow`` and FP64 ``log10``. WRF-exact arithmetic is rejected, and this
    stage does not by itself provide a complete P3 backend.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "P3 Metal k_loop_1 supports ordinary float32 only; source-pinned "
            "r_pow and FP64 log10 are not implemented")
    missing_fields = [name for name in _FIELD_RESOURCES if name not in fields]
    missing_scratch = [name for name in _SCRATCH_RESOURCES if name not in scratch]
    if missing_fields or missing_scratch:
        missing = missing_fields + missing_scratch
        raise ValueError("P3 Metal k_loop_1 is missing arrays: "
                         + ", ".join(missing))
    theta = fields["theta"]
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("P3 Metal k_loop_1 theta must be a 3-D MetalArray")
    nz, ny, nx = theta.shape
    if min(nz, ny, nx) < 1:
        raise ValueError("P3 Metal k_loop_1 dimensions must be nonempty")
    shape3 = (nz, ny, nx)
    context = theta.buffer.context
    arrays = {}
    for name in _FIELD_RESOURCES:
        value = fields[name]
        if (not isinstance(value, MetalArray) or value.shape != shape3
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"P3 Metal k_loop_1 {name} must be contiguous FP32 {shape3}")
        if value.buffer.context is not context:
            raise ValueError(f"P3 Metal k_loop_1 {name} belongs to another context")
        arrays[name] = value
    for name in _SCRATCH_RESOURCES:
        value = scratch[name]
        if (not isinstance(value, MetalArray) or value.shape != shape3
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"P3 Metal k_loop_1 {name} must be contiguous FP32 {shape3}")
        if value.buffer.context is not context:
            raise ValueError(f"P3 Metal k_loop_1 {name} belongs to another context")
        arrays[name] = value
    if (not isinstance(flags, MetalArray) or flags.shape != (2, ny, nx)
            or flags.dtype != np.dtype(np.float32)
            or not flags.flags.c_contiguous):
        raise TypeError(
            f"P3 Metal k_loop_1 flags must be contiguous FP32 {(2, ny, nx)}")
    if flags.buffer.context is not context:
        raise ValueError("P3 Metal k_loop_1 flags belong to another context")
    arrays["flags"] = flags

    context.submit_kernel(
        "p3_kloop1", arrays,
        {"nz": nz, "ny": ny, "nx": nx,
         "log_predict_nc": int(bool(log_predict_nc))},
        exact_policy=False, grid_size=ny * nx)


__all__ = ["submit_p3_kloop1"]
