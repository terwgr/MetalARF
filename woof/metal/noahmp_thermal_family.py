"""Typed resident launchers for independently supportable Noah-MP leaves.

THERMOPROP and PHASECHANGE use native Metal math calls; FRH2O remains refused.
"""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_LAYOUTS = {
    "thermoprop": (44, 4, 30),
    "hrt": (39, 1, 29),
    "hstep": (36, 1, 35),
    "tsnosoi": (42, 5, 8),
    "phasechange": (57, 4, 32),
}
_UNSUPPORTED = {
    "frh2o": "requires the shared glibc-compatible r_pow and r_log helpers",
}
_THREADS = 64


def submit_thermal_leaf(context, leaf: str, xs: MetalArray,
                        ixs: MetalArray, ys: MetalArray):
    """Submit one flat-batch thermal leaf without host conversion or packing."""
    if leaf in _UNSUPPORTED:
        raise NotImplementedError(
            f"Metal Noah-MP {leaf} is unavailable: {_UNSUPPORTED[leaf]}")
    if leaf not in _LAYOUTS:
        raise KeyError(f"unsupported Metal Noah-MP thermal leaf {leaf!r}")
    n_in, n_int, n_out = _LAYOUTS[leaf]
    if not isinstance(xs, MetalArray) or xs.dtype != np.dtype(np.float32):
        raise TypeError("thermal xs must be a float32 MetalArray")
    if not isinstance(ixs, MetalArray) or ixs.dtype != np.dtype(np.int32):
        raise TypeError("thermal ixs must be an int32 MetalArray")
    if not isinstance(ys, MetalArray) or ys.dtype != np.dtype(np.float32):
        raise TypeError("thermal ys must be a float32 MetalArray")
    if xs.ndim != 2:
        raise ValueError("thermal xs must have shape (ncase, n_in)")
    ncase = int(xs.shape[0])
    if tuple(xs.shape) != (ncase, n_in):
        raise ValueError(f"{leaf}: expected xs shape (ncase, {n_in})")
    if tuple(ixs.shape) != (ncase, n_int):
        raise ValueError(f"{leaf}: expected ixs shape (ncase, {n_int})")
    if tuple(ys.shape) != (ncase, n_out):
        raise ValueError(f"{leaf}: expected ys shape (ncase, {n_out})")
    if not ncase:
        return None
    entrypoint = f"noahmp_thermal_{leaf}"
    return context.submit_kernel(
        entrypoint,
        {"xs": xs, "ixs": ixs, "ys": ys},
        {"ncase": ncase, "n_in": n_in, "n_int": n_int, "n_out": n_out},
        exact_policy=False,
        grid_size=ncase,
        threads_per_group=_THREADS)


def submit_thermal_thermoprop(context, xs: MetalArray,
                              ixs: MetalArray, ys: MetalArray):
    return submit_thermal_leaf(context, "thermoprop", xs, ixs, ys)


def submit_thermal_hrt(context, xs: MetalArray,
                       ixs: MetalArray, ys: MetalArray):
    return submit_thermal_leaf(context, "hrt", xs, ixs, ys)



def submit_thermal_hstep(context, xs: MetalArray,
                         ixs: MetalArray, ys: MetalArray):
    return submit_thermal_leaf(context, "hstep", xs, ixs, ys)


def submit_thermal_tsnosoi(context, xs: MetalArray,
                           ixs: MetalArray, ys: MetalArray):
    return submit_thermal_leaf(context, "tsnosoi", xs, ixs, ys)


def submit_thermal_phasechange(context, xs: MetalArray,
                               ixs: MetalArray, ys: MetalArray):
    return submit_thermal_leaf(context, "phasechange", xs, ixs, ys)


__all__ = [
    "submit_thermal_leaf",
    "submit_thermal_thermoprop",
    "submit_thermal_hrt",
    "submit_thermal_hstep",
    "submit_thermal_tsnosoi",
    "submit_thermal_phasechange",
]
