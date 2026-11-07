"""Typed Metal dispatch for the water-budget corrections in correct mode."""
from __future__ import annotations

import numpy as np
from woof.metal.array import MetalArray, MetalArrayModule


def cloud_column_water(qc, temperature, pressure, qv, dz):
    if not isinstance(qc, MetalArray) or qc.ndim != 3:
        raise TypeError("Metal cloud water requires a three-dimensional MetalArray")
    context = qc.buffer.context
    arrays = dict(qc=qc, temperature=temperature, pressure=pressure, qv=qv, dz=dz)
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray) or array.shape != qc.shape
                or array.dtype != np.dtype(np.float32)
                or array.buffer.context is not context):
            raise TypeError(f"Metal cloud water {name} must share float32 shape and context")
    nz, ny, nx = qc.shape
    water = MetalArrayModule(context).empty((ny, nx), np.float64)
    context.submit_kernel("thompson_cloud_column_water", {**arrays, "water": water},
                          dict(nz=nz, ny=ny, nx=nx), exact_policy=False,
                          grid_size=ny * nx, threads_per_group=128)
    return water


def book_cloud(state, cfg, before, after, *, dt=1.0):
    from woof.core.thompson_cloud_booking import CLOUDNC_SLOT, CLOUDNCV_SLOT
    if (not isinstance(before, MetalArray) or not isinstance(after, MetalArray)
            or before.shape != after.shape or before.ndim != 2
            or before.dtype != np.dtype(np.float64)
            or after.dtype != before.dtype
            or before.buffer.context is not after.buffer.context):
        raise TypeError("Metal cloud booking needs matching binary64 column totals")
    context = before.buffer.context
    ny, nx = before.shape
    cloudnc = state.scratch((ny, nx), CLOUDNC_SLOT)
    cloudncv = state.scratch((ny, nx), CLOUDNCV_SLOT)
    bits = int(np.asarray(float(dt), np.float64).view(np.uint64))
    zone = max(0, int(cfg.spec_zone)) if (getattr(cfg, "specified", False)
            or getattr(cfg, "nested", False)) else 0
    context.submit_kernel("thompson_cloud_book",
                          dict(before=before, after=after, cloudnc=cloudnc, cloudncv=cloudncv),
                          dict(ny=ny, nx=nx, spec_zone=zone,
                               factor_lo=int(np.asarray(bits & 0xFFFFFFFF, np.uint32).view(np.int32)),
                               factor_hi=int(np.asarray(bits >> 32, np.uint32).view(np.int32))),
                          exact_policy=False, grid_size=ny * nx, threads_per_group=128)
    return cloudnc, cloudncv


def correct_ntiedtke(pipeline, *, forcing):
    names = (("qvf", "pqte") if forcing else
             ("qv3d", "pqte", "zqq", "rqvcuten", "rqccuten", "rqicuten"))
    arrays = {}
    for name in names:
        array = pipeline.w.bind(name, 0)
        arrays[name] = array.view((array.size,))
    pipeline.module.context.submit_kernel(
        "ntiedtke_correct_forcing" if forcing else "ntiedtke_correct_water", arrays,
        dict(nz=pipeline.nz, ncol=pipeline.width), exact_policy=False,
        grid_size=pipeline.nz * pipeline.width, threads_per_group=128)
