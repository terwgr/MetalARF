"""Typed Metal submission for P3's main process-rate and prognostic stage."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


_FIELDS = ("qc", "nc", "qr", "nr", "qi", "qir", "ni", "qib",
           "theta", "qv", "th_old", "qv_old", "ssat", "pressure")
_SCRATCH = ("rho", "inv_rho", "qvs", "qvi", "sup", "supi",
            "rhofacr", "rhofaci", "acn", "temperature", "tmparr1",
            "qv_cld")


def submit_p3_kloopmain(*, fields: dict, scratch: dict, flags,
                        itab, icoll, revap, dt: float, it: int,
                        log_predict_nc: bool = False,
                        clbfact_dep: float = 1.0,
                        clbfact_sub: float = 1.0) -> None:
    """Run the source-ordered P3 process-rate update, one invocation per column.

    The prognostic and scratch volumes are contiguous FP32 ``(nz,ny,nx)``
    arrays.  The caller owns the lookup tables, workspace, and column flags;
    all resources must belong to the same Metal context.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "P3 Metal k_loop_main supports ordinary float32 arithmetic only; "
            "source-pinned libm and binary64 excursions are not implemented")
    missing = ([name for name in _FIELDS if name not in fields]
               + [name for name in _SCRATCH if name not in scratch])
    if missing:
        raise ValueError("P3 Metal k_loop_main is missing arrays: "
                         + ", ".join(missing))
    theta = fields["theta"]
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("P3 Metal k_loop_main theta must be a 3-D MetalArray")
    nz, ny, nx = theta.shape
    if min(nz, ny, nx) < 1:
        raise ValueError("P3 Metal k_loop_main dimensions must be nonempty")
    context = theta.buffer.context
    arrays = {}
    for name in _FIELDS:
        value = fields[name]
        if (not isinstance(value, MetalArray) or value.shape != (nz, ny, nx)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal k_loop_main {name} must be contiguous FP32 {(nz, ny, nx)}")
        if value.buffer.context is not context:
            raise ValueError(f"P3 Metal k_loop_main {name} belongs to another context")
        arrays[name] = value
    for name in _SCRATCH:
        value = scratch[name]
        if (not isinstance(value, MetalArray) or value.shape != (nz, ny, nx)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal k_loop_main {name} must be contiguous FP32 {(nz, ny, nx)}")
        if value.buffer.context is not context:
            raise ValueError(f"P3 Metal k_loop_main {name} belongs to another context")
        arrays[name] = value
    if (not isinstance(flags, MetalArray) or flags.shape != (2, ny, nx)
            or flags.dtype != np.dtype(np.float32)
            or not flags.flags.c_contiguous):
        raise TypeError(
            f"P3 Metal k_loop_main flags must be contiguous FP32 {(2, ny, nx)}")
    if flags.buffer.context is not context:
        raise ValueError("P3 Metal k_loop_main flags belong to another context")
    arrays["flags"] = flags
    for name, value, length in (("itab", itab, 14000),
                                ("icoll", icoll, 60000),
                                ("revap", revap, 3000)):
        if (not isinstance(value, MetalArray) or value.shape != (length,)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal k_loop_main {name} must be contiguous FP32 ({length},)")
        if value.buffer.context is not context:
            raise ValueError(f"P3 Metal k_loop_main {name} belongs to another context")
        arrays[name] = value
    step = np.float32(dt)
    dep = np.float32(clbfact_dep)
    sub = np.float32(clbfact_sub)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("P3 Metal k_loop_main dt must be positive finite FP32")
    if not np.isfinite(dep) or not np.isfinite(sub):
        raise ValueError("P3 Metal k_loop_main cloud fractions must be finite FP32")
    if int(it) < 0:
        raise ValueError("P3 Metal k_loop_main timestep index must be nonnegative")
    context.submit_kernel(
        "p3_kloopmain", arrays,
        {"nz": nz, "ny": ny, "nx": nx, "dt": float(step),
         "it": int(it), "log_predict_nc": int(bool(log_predict_nc)),
         "clbfact_dep": float(dep), "clbfact_sub": float(sub)},
        exact_policy=False, grid_size=ny * nx)


__all__ = ["submit_p3_kloopmain"]
