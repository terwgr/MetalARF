"""Typed Metal submitters for NSSL's state gather/scatter transport kernels."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray


_TPB = 256
_STATE_FIELDS = 16


def _validate(array, name, context, shape):
    if not isinstance(array, MetalArray):
        raise TypeError(f"{name} must be a MetalArray")
    if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous FP32 storage")
    if array.buffer.context is not context:
        raise ValueError("NSSL resources must share the submitting MetalContext")
    if array.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {array.shape}")


def _dimensions(shape):
    if len(shape) != 3:
        raise ValueError("NSSL driver fields must have shape (nz, ny, nx)")
    nz, ny, nx = shape
    if any(isinstance(v, (bool, np.bool_))
           or not isinstance(v, (int, np.integer)) or v < 1
           for v in (nz, ny, nx)) or nz < 2 or nz > 256:
        raise ValueError("NSSL driver requires 2 <= nz <= 256 and positive horizontal extents")
    return tuple(int(v) for v in (nz, ny, nx))


def submit_nssl2_driver_gather_initialize(
        context, state, *, air_density, qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
        qrcuten, qscuten, qicuten, qccuten, dt_s, first_step=False,
        cu_used=False, predicted_ccn=True):
    """Gather Registry moments into the driver's 16-plane internal slab."""
    if type(first_step) is not bool or type(cu_used) is not bool or type(predicted_ccn) is not bool:
        raise TypeError("NSSL driver switches must be bool")
    if not isinstance(state, MetalArray) or len(state.shape) != 4 or state.shape[0] != 16:
        raise ValueError("state must have shape (16, nz, ny, nx)")
    nz, ny, nx = _dimensions(state.shape[1:])
    shape = (nz, ny, nx)
    fields = {name: value for name, value in locals().copy().items()
              if name in {"air_density", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
                          "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
                          "qvolg", "qvolh", "qrcuten", "qscuten", "qicuten", "qccuten"}}
    for name, value in fields.items():
        _validate(value, name, context, shape)
    _validate(state, "state", context, (16, *shape))
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("NSSL dt_s must be positive and finite") from exc
    step32 = np.float32(step)
    if not np.isfinite(step) or step <= 0 or not np.isfinite(step32) or step32 <= 0:
        raise ValueError("NSSL dt_s must be representable as positive float32")
    n = nz * ny * nx
    resources = {**fields, "state": state}
    return context.submit_kernel(
        "nssl2_driver_gather_initialize", resources,
        {"dt": step32, "first_step": int(first_step), "cu_used": int(cu_used),
         "n": n, "predicted_ccn": int(predicted_ccn),
         "nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=n, threads_per_group=_TPB)


def submit_nssl2_driver_scatter(
        context, state, air_density, *, qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
        predicted_ccn=True):
    """Scatter the internal moment slab back to Registry mixing ratios."""
    if type(predicted_ccn) is not bool:
        raise TypeError("predicted_ccn must be bool")
    if not isinstance(state, MetalArray) or len(state.shape) != 4 or state.shape[0] != 16:
        raise ValueError("state must have shape (16, nz, ny, nx)")
    nz, ny, nx = _dimensions(state.shape[1:])
    shape = (nz, ny, nx)
    _validate(state, "state", context, (16, *shape))
    fields = {name: value for name, value in locals().copy().items()
              if name in {"air_density", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
                          "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh"}}
    for name, value in fields.items():
        _validate(value, name, context, shape)
    n = nz * ny * nx
    return context.submit_kernel(
        "nssl2_driver_scatter", {**fields, "state": state},
        {"n": n, "predicted_ccn": int(predicted_ccn),
         "nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=n, threads_per_group=_TPB)


def submit_nssl2_driver_reduce_precipitation(
        context, category_export, *, rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, hailnc, hailncv, sr):
    """Apply WRF's standard four-category surface precipitation reduction."""
    if not isinstance(category_export, MetalArray) or len(category_export.shape) != 3:
        raise ValueError("category_export must have shape (5, ny, nx)")
    if category_export.shape[0] != 5:
        raise ValueError("category_export must have five category planes")
    ny, nx = category_export.shape[1:]
    if ny < 1 or nx < 1:
        raise ValueError("precipitation export must have positive horizontal extents")
    _validate(category_export, "category_export", context, (5, ny, nx))
    fields = {name: value for name, value in locals().copy().items()
              if name in {"rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
                          "graupelncv", "hailnc", "hailncv", "sr"}}
    for name, value in fields.items():
        _validate(value, name, context, (ny, nx))
    ncol = ny * nx
    return context.submit_kernel(
        "nssl2_driver_reduce_precipitation",
        {"category_export": category_export, **fields},
        {"ncol": ncol, "ny": ny, "nx": nx}, exact_policy=False, grid_size=ncol,
        threads_per_group=_TPB)


__all__ = [
    "submit_nssl2_driver_gather_initialize",
    "submit_nssl2_driver_reduce_precipitation",
    "submit_nssl2_driver_scatter",
]
