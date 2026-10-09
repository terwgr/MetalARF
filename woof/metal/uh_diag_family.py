"""Typed Metal launchers for the WRF updraft-helicity diagnostic."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def _validate(arrays, context):
    for name, value in arrays.items():
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"Metal UH {name} must be contiguous float32")
        if value.buffer.context is not context:
            raise ValueError("Metal UH arrays must share one context")


def submit_uh_step(u, v, w, ph, phb, msfu, msfv, ht, dn, dnw, fnm, fnp,
                   cf1, cf2, cf3, rdx, rdy, uh, use, up_heli_max):
    """Run the column kernel, then smooth/fold and apply WRF boundary copies."""
    arrays = {"u": u, "v": v, "w": w, "ph": ph, "phb": phb,
              "msfu": msfu, "msfv": msfv, "ht": ht, "dn": dn,
              "dnw": dnw, "fnm": fnm, "fnp": fnp, "uh": uh,
              "use": use, "up_heli_max": up_heli_max}
    context = w.buffer.context if isinstance(w, MetalArray) else None
    _validate(arrays, context)
    nzf, ny, nx = w.shape
    nz = nzf - 1
    if (u.shape != (nz, ny, nx + 1) or v.shape != (nz, ny + 1, nx)
            or ph.shape != w.shape or uh.shape != (ny, nx)
            or use.shape != (ny, nx) or up_heli_max.shape != (ny, nx)
            or msfu.shape != (ny, nx + 1) or msfv.shape != (ny + 1, nx)
            or ht.shape != (ny, nx) or dn.size != nz or dnw.size != nz
            or fnm.size != nz or fnp.size != nz
            or phb.shape not in (w.shape, (nzf,))):
        raise ValueError("Metal UH input shapes do not match the staggered grid")
    ncol = ny * nx
    if not ncol:
        return
    context.submit_kernel("uh_columns", arrays={
        "u": u, "v": v, "w": w, "ph": ph, "phb": phb,
        "msfu": msfu, "msfv": msfv, "ht": ht, "dn": dn, "dnw": dnw,
        "fnm": fnm, "fnp": fnp, "uh": uh, "usecol": use,
    }, parameters={"cf1": float(np.float32(cf1)),
                   "cf2": float(np.float32(cf2)),
                   "cf3": float(np.float32(cf3)),
                   "rdx": float(np.float32(rdx)),
                   "rdy": float(np.float32(rdy)),
                   "phb3d": int(phb.ndim == 3), "nz": nz,
                   "ny": ny, "nx": nx, "ncol": ncol},
        exact_policy=True, grid_size=ncol, threads_per_group=128)
    submit_uh_fold(uh, use, up_heli_max)


def submit_uh_fold(uh, use, target):
    if (not isinstance(uh, MetalArray) or uh.ndim != 2
            or target.shape != uh.shape or use.shape != uh.shape):
        raise TypeError("Metal UH fold requires matching 2-D arrays")
    context = uh.buffer.context
    _validate({"uh": uh, "use": use, "target": target}, context)
    ny, nx = uh.shape
    ncol = ny * nx
    if not ncol:
        return
    context.submit_kernel("uh_smooth_max", {
        "uh": uh, "usecol": use, "up_heli_max": target,
    }, {"ny": ny, "nx": nx}, exact_policy=True,
       grid_size=ncol, threads_per_group=128)
    context.submit_kernel("uh_boundary_copy", {
        "field": target,
    }, {"ny": ny, "nx": nx, "ncol": ncol}, exact_policy=True,
       grid_size=ncol, threads_per_group=128)
