"""Typed Metal submission for P3 rain sedimentation."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


_FIELDS = ("qr", "nr", "dz")
_SCRATCH = ("rho", "inv_rho", "rhofacr", "vq", "vn", "fq", "fn")


def submit_p3_sed_rain(*, fields: dict, scratch: dict, vm_table, vn_table,
                       prt_liq, flags, dt: float) -> None:
    """Run P3's rain sedimentation using caller-owned P3 velocity tables.

    Volume arrays are contiguous FP32 ``(nz,ny,nx)``; surface arrays are
    ``(ny,nx)``, and ``flags`` is ``(2,ny,nx)``. Each rain table is the flat
    contiguous 3,000-value `(300,10)` source table.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "P3 Metal rain sedimentation supports ordinary float32 only; "
            "source libm operations are not implemented")
    missing = ([n for n in _FIELDS if n not in fields]
               + [n for n in _SCRATCH if n not in scratch])
    if missing:
        raise ValueError("P3 Metal rain sedimentation is missing arrays: "
                         + ", ".join(missing))
    qr = fields["qr"]
    if not isinstance(qr, MetalArray) or qr.ndim != 3:
        raise TypeError("P3 Metal rain sedimentation qr must be a 3-D MetalArray")
    nz, ny, nx = qr.shape
    if min(nz, ny, nx) < 1:
        raise ValueError("P3 Metal rain sedimentation dimensions must be nonempty")
    context = qr.buffer.context
    arrays = {}
    for name in _FIELDS:
        value = fields[name]
        if (not isinstance(value, MetalArray) or value.shape != (nz, ny, nx)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal rain sedimentation {name} must be contiguous FP32 {(nz, ny, nx)}")
        if value.buffer.context is not context:
            raise ValueError(
                f"P3 Metal rain sedimentation {name} belongs to another context")
        arrays[name] = value
    for name in _SCRATCH:
        value = scratch[name]
        if (not isinstance(value, MetalArray) or value.shape != (nz, ny, nx)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal rain sedimentation {name} must be contiguous FP32 {(nz, ny, nx)}")
        if value.buffer.context is not context:
            raise ValueError(
                f"P3 Metal rain sedimentation {name} belongs to another context")
        arrays[name] = value
    for name, table in (("vm_table", vm_table), ("vn_table", vn_table)):
        if (not isinstance(table, MetalArray) or table.shape != (3000,)
                or table.dtype != np.dtype(np.float32)
                or not table.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal rain sedimentation {name} must be contiguous FP32 (3000,)")
        if table.buffer.context is not context:
            raise ValueError(
                f"P3 Metal rain sedimentation {name} belongs to another context")
        arrays[name] = table
    for name, value, shape in (("prt_liq", prt_liq, (ny, nx)),
                                ("flags", flags, (2, ny, nx))):
        if (not isinstance(value, MetalArray) or value.shape != shape
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal rain sedimentation {name} must be contiguous FP32 {shape}")
        if value.buffer.context is not context:
            raise ValueError(
                f"P3 Metal rain sedimentation {name} belongs to another context")
        arrays[name] = value
    step = np.float32(dt)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("P3 Metal rain sedimentation dt must be positive finite FP32")
    context.submit_kernel(
        "p3_sed_rain", arrays,
        {"nz": nz, "ny": ny, "nx": nx, "dt": float(step)},
        exact_policy=False, grid_size=ny * nx)


__all__ = ["submit_p3_sed_rain"]
