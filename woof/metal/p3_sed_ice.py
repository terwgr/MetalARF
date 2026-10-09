"""Typed Metal submission for P3 one-category ice sedimentation."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


_FIELDS = ("qi", "qir", "ni", "qib", "dz")
_SCRATCH = ("rho", "inv_rho", "rhofaci", "vq", "vn", "fq", "fn",
            "fqir", "fbir")


def submit_p3_sed_ice(*, fields: dict, scratch: dict, itab, prt_sol,
                      flags, dt: float) -> None:
    """Run P3 ice/rime sedimentation with caller-owned ITAB/workspace.

    Volume arrays are contiguous FP32 ``(nz,ny,nx)``; ITAB is flat FP32
    `(14000,)`, PRTSOL is `(ny,nx)`, and flags are `(2,ny,nx)`.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "P3 Metal ice sedimentation supports ordinary float32 only; "
            "source double-log10 is not implemented")
    missing = ([n for n in _FIELDS if n not in fields]
               + [n for n in _SCRATCH if n not in scratch])
    if missing:
        raise ValueError("P3 Metal ice sedimentation is missing arrays: "
                         + ", ".join(missing))
    qi = fields["qi"]
    if not isinstance(qi, MetalArray) or qi.ndim != 3:
        raise TypeError("P3 Metal ice sedimentation qi must be a 3-D MetalArray")
    nz, ny, nx = qi.shape
    if min(nz, ny, nx) < 1:
        raise ValueError("P3 Metal ice sedimentation dimensions must be nonempty")
    context = qi.buffer.context
    arrays = {}
    for name in _FIELDS:
        value = fields[name]
        if (not isinstance(value, MetalArray) or value.shape != (nz, ny, nx)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal ice sedimentation {name} must be contiguous FP32 {(nz, ny, nx)}")
        if value.buffer.context is not context:
            raise ValueError(
                f"P3 Metal ice sedimentation {name} belongs to another context")
        arrays[name] = value
    for name in _SCRATCH:
        value = scratch[name]
        if (not isinstance(value, MetalArray) or value.shape != (nz, ny, nx)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal ice sedimentation {name} must be contiguous FP32 {(nz, ny, nx)}")
        if value.buffer.context is not context:
            raise ValueError(
                f"P3 Metal ice sedimentation {name} belongs to another context")
        arrays[name] = value
    if (not isinstance(itab, MetalArray) or itab.shape != (14000,)
            or itab.dtype != np.dtype(np.float32)
            or not itab.flags.c_contiguous):
        raise TypeError("P3 Metal ice sedimentation ITAB must be contiguous FP32 (14000,)")
    if itab.buffer.context is not context:
        raise ValueError("P3 Metal ice sedimentation ITAB belongs to another context")
    for name, value, shape in (("prt_sol", prt_sol, (ny, nx)),
                                ("flags", flags, (2, ny, nx))):
        if (not isinstance(value, MetalArray) or value.shape != shape
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal ice sedimentation {name} must be contiguous FP32 {shape}")
        if value.buffer.context is not context:
            raise ValueError(
                f"P3 Metal ice sedimentation {name} belongs to another context")
        arrays[name] = value
    arrays["itab"] = itab
    step = np.float32(dt)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("P3 Metal ice sedimentation dt must be positive finite FP32")
    context.submit_kernel(
        "p3_sed_ice", arrays,
        {"nz": nz, "ny": ny, "nx": nx, "dt": float(step)},
        exact_policy=False, grid_size=ny * nx)


__all__ = ["submit_p3_sed_ice"]
