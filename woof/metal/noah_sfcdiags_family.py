"""Typed Metal launcher for Noah's SFCDIAGS surface diagnostics."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def _require_column(value, name, shape, context):
    if (not isinstance(value, MetalArray)
            or value.dtype != np.dtype(np.float32)
            or value.shape != tuple(shape)
            or not value.flags.c_contiguous):
        raise TypeError(
            f"Metal Noah SFCDIAGS {name} must be contiguous float32 "
            f"{tuple(shape)}")
    if value.buffer.context is not context:
        raise ValueError("Metal Noah SFCDIAGS inputs must share one context")
    return value


def submit_noah_sfcdiags(fields, qv1, *, rd, cp_air, p0, rcp,
                          active_floor) -> None:
    """Submit Q2/T2/TH2 updates without copying state to the host.

    The arithmetic operators retain the CUDA kernel's explicit association
    and are compiled with contraction disabled. Metal's precise ``pow`` is
    used for the final Exner factor. Its special-function implementation is
    not promised bit-identical to CUDA libdevice ``powf``; ordinary finite
    values should follow the same FP32 operation sequence, with that possible
    last-bit difference in TH2.
    """
    anchor = fields["tsk"]
    if not isinstance(anchor, MetalArray) or anchor.ndim != 2:
        raise TypeError("Metal Noah SFCDIAGS fields must be 2-D MetalArrays")
    shape = anchor.shape
    ny, nx = shape
    context = anchor.buffer.context
    names = ("psfc", "tsk", "cqs2", "chs2", "qsfc", "qfx", "hfx")
    arrays = {f"{name}_a": _require_column(
        fields[name], name, shape, context) for name in names}
    arrays["qv1_a"] = _require_column(qv1, "qv1", shape, context)
    for name in ("q2", "t2", "th2"):
        arrays[f"{name}_a"] = _require_column(
            fields[name], name, shape, context)

    ncol = ny * nx
    if ncol == 0:
        return
    parameters = {
        "rd": float(np.float32(rd)),
        "cp_air": float(np.float32(cp_air)),
        "p0": float(np.float32(p0)),
        "rcp": float(np.float32(rcp)),
        "active_floor": float(np.float32(active_floor)),
        "ny": ny, "nx": nx, "ncol": ncol,
    }
    context.submit_kernel(
        "noah_sfcdiags", arrays, parameters, exact_policy=False,
        grid_size=ncol, threads_per_group=128)
