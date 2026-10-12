"""Typed Metal submission for the P3 cloud sedimentation stage."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from .p3_constants import P3_CONSTANT_NAMES
from woof.wrf_exact import ENABLED as WRF_EXACT


_FIELDS = ("qc", "nc", "dz")
_SCRATCH = ("rho", "inv_rho", "acn", "vq", "vn", "fq", "fn")


def submit_p3_sed_cloud(*, fields: dict, scratch: dict, prt_liq, flags,
                        constants, dt: float,
                        log_predict_nc: bool = False) -> None:
    """Run P3's source-ordered cloud sedimentation over flagged columns.

    Field and workspace volumes are contiguous FP32 ``(nz,ny,nx)`` arrays;
    ``prt_liq`` is ``(ny,nx)`` and ``flags`` is ``(2,ny,nx)``. The caller
    supplies the P3-owned sedimentation workspace and sets ``log_predict_nc``
    from the same prognostic-Nc presence rule as the rest of the P3 chain.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "P3 Metal cloud sedimentation supports ordinary float32 only; "
            "source libm operations are not implemented")
    missing = ([n for n in _FIELDS if n not in fields]
               + [n for n in _SCRATCH if n not in scratch])
    if missing:
        raise ValueError("P3 Metal cloud sedimentation is missing arrays: "
                         + ", ".join(missing))
    qc = fields["qc"]
    if not isinstance(qc, MetalArray) or qc.ndim != 3:
        raise TypeError("P3 Metal cloud sedimentation qc must be a 3-D MetalArray")
    nz, ny, nx = qc.shape
    if min(nz, ny, nx) < 1:
        raise ValueError("P3 Metal cloud sedimentation dimensions must be nonempty")
    context = qc.buffer.context
    arrays = {}
    for name in _FIELDS:
        value = fields[name]
        if (not isinstance(value, MetalArray) or value.shape != (nz, ny, nx)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal cloud sedimentation {name} must be contiguous FP32 {(nz, ny, nx)}")
        if value.buffer.context is not context:
            raise ValueError(
                f"P3 Metal cloud sedimentation {name} belongs to another context")
        arrays[name] = value
    for name in _SCRATCH:
        value = scratch[name]
        if (not isinstance(value, MetalArray) or value.shape != (nz, ny, nx)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal cloud sedimentation {name} must be contiguous FP32 {(nz, ny, nx)}")
        if value.buffer.context is not context:
            raise ValueError(
                f"P3 Metal cloud sedimentation {name} belongs to another context")
        arrays[name] = value
    if (not isinstance(prt_liq, MetalArray) or prt_liq.shape != (ny, nx)
            or prt_liq.dtype != np.dtype(np.float32)
            or not prt_liq.flags.c_contiguous):
        raise TypeError(
            f"P3 Metal cloud sedimentation prt_liq must be contiguous FP32 {(ny, nx)}")
    if prt_liq.buffer.context is not context:
        raise ValueError("P3 Metal cloud sedimentation prt_liq uses another context")
    if (not isinstance(flags, MetalArray) or flags.shape != (2, ny, nx)
            or flags.dtype != np.dtype(np.float32)
            or not flags.flags.c_contiguous):
        raise TypeError(
            f"P3 Metal cloud sedimentation flags must be contiguous FP32 {(2, ny, nx)}")
    if flags.buffer.context is not context:
        raise ValueError("P3 Metal cloud sedimentation flags use another context")
    arrays["prt_liq"] = prt_liq
    arrays["flags"] = flags
    if (not isinstance(constants, MetalArray)
            or constants.shape != (len(P3_CONSTANT_NAMES),)
            or constants.dtype != np.dtype(np.float32)
            or not constants.flags.c_contiguous):
        raise TypeError(
            "P3 Metal cloud sedimentation constants must be contiguous FP32 "
            f"({len(P3_CONSTANT_NAMES)},)")
    if constants.buffer.context is not context:
        raise ValueError("P3 Metal cloud sedimentation constants use another context")
    arrays["constants"] = constants
    step = np.float32(dt)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("P3 Metal cloud sedimentation dt must be positive finite FP32")
    context.submit_kernel(
        "p3_sed_cloud", arrays,
        {"nz": nz, "ny": ny, "nx": nx, "dt": float(step),
         "log_predict_nc": int(bool(log_predict_nc))},
        exact_policy=False, grid_size=ny * nx)


__all__ = ["submit_p3_sed_cloud"]
