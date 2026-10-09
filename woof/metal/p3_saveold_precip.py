"""Typed Metal submission for P3's cross-step and precipitation finish."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


def submit_p3_saveold_precip(*, theta, qv, th_old, qv_old, prt_liq,
                             prt_sol, rainnc, rainncv, sr, snownc,
                             snowncv, dt: float) -> None:
    """Save P3's old theta/qv and convert precip rates into WRF accumulators.

    All arrays are same-context contiguous FP32 MetalArray objects. The four
    volume arrays use ``(nz,ny,nx)``; the seven surface arrays use ``(ny,nx)``.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "P3 Metal finish supports ordinary float32 arithmetic only")
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("P3 Metal finish theta must be a 3-D MetalArray")
    nz, ny, nx = theta.shape
    if min(nz, ny, nx) < 1:
        raise ValueError("P3 Metal finish dimensions must be nonempty")
    context = theta.buffer.context
    arrays = {
        "theta": theta, "qv": qv, "th_old": th_old, "qv_old": qv_old,
        "prt_liq": prt_liq, "prt_sol": prt_sol, "rainnc": rainnc,
        "rainncv": rainncv, "sr": sr, "snownc": snownc,
        "snowncv": snowncv,
    }
    volume_names = {"theta", "qv", "th_old", "qv_old"}
    for name, value in arrays.items():
        shape = (nz, ny, nx) if name in volume_names else (ny, nx)
        if (not isinstance(value, MetalArray) or value.shape != shape
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"P3 Metal finish {name} must be contiguous FP32 {shape}")
        if value.buffer.context is not context:
            raise ValueError(f"P3 Metal finish {name} belongs to another context")
    context.submit_kernel(
        "p3_saveold_precip", arrays,
        {"nz": nz, "ny": ny, "nx": nx, "dt": float(np.float32(dt))},
        exact_policy=False, grid_size=ny * nx)


__all__ = ["submit_p3_saveold_precip"]
