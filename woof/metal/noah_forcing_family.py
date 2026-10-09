"""Typed Metal launchers for the Noah forcing prologue and rain epilogue."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def _require_column(value, name, shape, context):
    if (not isinstance(value, MetalArray)
            or value.dtype != np.dtype(np.float32)
            or value.shape != tuple(shape)
            or not value.flags.c_contiguous):
        raise TypeError(
            f"Metal Noah forcing {name} must be contiguous float32 {tuple(shape)}")
    if value.buffer.context is not context:
        raise ValueError("Metal Noah forcing inputs must share one context")
    return value


def submit_noah_forcing(fields, *, p_if0, p_if1, temperature0, qv0, dz0,
                        sr_scheme, pending, freezing) -> None:
    """Submit the fused Noah forcing prologue for one 2-D domain."""
    anchor = fields["sfcprs"]
    if not isinstance(anchor, MetalArray) or anchor.ndim != 2:
        raise TypeError("Metal Noah forcing fields must be 2-D MetalArrays")
    ny, nx = anchor.shape
    shape = (ny, nx)
    context = anchor.buffer.context
    inputs = {
        "p_if0": p_if0,
        "p_if1": p_if1,
        "temperature0": temperature0,
        "qv0": qv0,
        "dz0": dz0,
        "br": fields["br"],
        # The unused resource is still bound when the scheme SR is absent;
        # temperature0 is a same-context placeholder and the shader does not
        # read it in that branch.
        "sr_scheme": temperature0 if sr_scheme is None else sr_scheme,
        "pending": pending,
        "sfcprs": fields["sfcprs"],
        "sfctmp": fields["sfctmp"],
        "qv1": fields["qv1"],
        "dz8w1": fields["dz8w1"],
        "rib": fields["rib"],
        "sr": fields["sr"],
        "rainbl": fields["rainbl"],
    }
    for name, value in tuple(inputs.items()):
        inputs[name] = _require_column(value, name, shape, context)

    ncol = ny * nx
    if ncol == 0:
        return
    context.submit_kernel(
        "noah_forcing",
        {f"{name}_a": value for name, value in inputs.items()},
        {"freezing": float(np.float32(freezing)),
         "use_scheme_sr": int(sr_scheme is not None),
         "ny": ny, "nx": nx, "ncol": ncol},
        exact_policy=False, grid_size=ncol, threads_per_group=128)


def submit_noah_rain_clear(rainbl, pending) -> None:
    """Submit the fused clear of the Noah rain carrier arrays."""
    if not isinstance(rainbl, MetalArray) or rainbl.ndim != 2:
        raise TypeError("Metal Noah rain arrays must be 2-D MetalArrays")
    shape = rainbl.shape
    context = rainbl.buffer.context
    _require_column(rainbl, "rainbl", shape, context)
    _require_column(pending, "pending", shape, context)
    ncol = int(rainbl.size)
    if ncol == 0:
        return
    context.submit_kernel(
        "noah_rain_clear",
        {"rainbl_a": rainbl, "pending_a": pending},
        {"ny": shape[0], "nx": shape[1], "ncol": ncol},
        exact_policy=False, grid_size=ncol, threads_per_group=128)


def submit_noah_rain_add(rainbl, pending) -> None:
    """Accumulate pending precipitation into Noah-MP's RAINBL carrier."""
    if not isinstance(rainbl, MetalArray) or rainbl.ndim != 2:
        raise TypeError("Metal Noah rain arrays must be 2-D MetalArrays")
    shape = rainbl.shape
    context = rainbl.buffer.context
    _require_column(rainbl, "rainbl", shape, context)
    _require_column(pending, "pending", shape, context)
    ncol = int(rainbl.size)
    if ncol == 0:
        return
    context.submit_kernel(
        "noah_rain_add",
        {"rainbl_a": rainbl, "pending_a": pending},
        {"ny": shape[0], "nx": shape[1], "ncol": ncol},
        exact_policy=False, grid_size=ncol, threads_per_group=128)


def submit_noah_sr_from_temperature(temperature, sr, *, freezing) -> None:
    """Fill Noah-MP's fallback snow ratio from the lowest temperature level."""
    if (not isinstance(temperature, MetalArray) or temperature.ndim != 3
            or temperature.dtype != np.dtype(np.float32)
            or not temperature.flags.c_contiguous):
        raise TypeError("Metal Noah SR temperature must be contiguous float32 (nz, ny, nx)")
    if (not isinstance(sr, MetalArray) or sr.ndim != 2
            or sr.dtype != np.dtype(np.float32)
            or not sr.flags.c_contiguous):
        raise TypeError("Metal Noah SR output must be contiguous float32 (ny, nx)")
    nz, ny, nx = temperature.shape
    if nz < 1 or sr.shape != (ny, nx):
        raise ValueError("Metal Noah SR temperature/output shapes do not match")
    context = temperature.buffer.context
    if sr.buffer.context is not context:
        raise ValueError("Metal Noah SR arrays must share one context")
    ncol = ny * nx
    if ncol == 0:
        return
    context.submit_kernel(
        "noah_sr_from_temperature",
        {"temperature_a": temperature.view((ncol,)),
         "sr_a": sr.view((ncol,))},
        {"freezing": float(np.float32(freezing)), "ncol": ncol},
        exact_policy=False, grid_size=ncol, threads_per_group=128)
