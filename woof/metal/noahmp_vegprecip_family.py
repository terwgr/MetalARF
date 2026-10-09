"""Typed launchers for the Noah-MP vegetation and precipitation leaves."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_THREADS = 64
_I32_MAX = np.iinfo(np.int32).max


def _validate_rows(leaf, *, x, ix, y, ncase, nx, nix, ny):
    if not isinstance(ncase, (int, np.integer)):
        raise TypeError(f"Noah-MP {leaf} ncase must be an integer")
    ncase = int(ncase)
    if not 0 <= ncase <= _I32_MAX:
        raise ValueError(f"Noah-MP {leaf} ncase must fit int32")
    expected = {"x": ((ncase, nx), np.dtype(np.float32)),
                "ix": ((ncase, nix), np.dtype(np.int32)),
                "y": ((ncase, ny), np.dtype(np.float32))}
    arrays = {"x": x, "ix": ix, "y": y}
    context = None
    for name, array in arrays.items():
        if not isinstance(array, MetalArray):
            raise TypeError(f"Noah-MP {leaf} {name} must be a MetalArray")
        shape, dtype = expected[name]
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"Noah-MP {leaf} {name} must be {dtype} with shape {shape}")
        if not array.flags.c_contiguous:
            raise ValueError(f"Noah-MP {leaf} {name} must be contiguous")
        if context is None:
            context = array.buffer.context
        elif array.buffer.context is not context:
            raise ValueError(f"Noah-MP {leaf} arrays must share one context")
    return ncase, context, arrays


def submit_phenology(*, x, ix, y, ncase):
    """Submit PHENOLOGY packed rows (31 float, 6 int, 6 float slots).

    The snow-cover exponential uses native MSL ``exp``. This enables the
    WRF caller path without claiming parity with glibc ``expf``.
    """
    ncase, context, arrays = _validate_rows(
        "PHENOLOGY", x=x, ix=ix, y=y, ncase=ncase,
        nx=31, nix=6, ny=6)
    if not ncase:
        return None
    return context.submit_kernel(
        "noahmp_vegprecip_phenology", arrays, {"ncase": ncase},
        exact_policy=False,
        grid_size=((ncase + _THREADS - 1) // _THREADS) * _THREADS,
        threads_per_group=_THREADS)


def submit_precip_heat(*, x, ix, y, ncase):
    """Submit PRECIP_HEAT packed rows (16 float, 1 int, 16 float slots).

    The source's exponential and fractional power use native MSL ``exp`` and
    ``pow``. This is the default Metal arithmetic policy.
    """
    ncase, context, arrays = _validate_rows(
        "PRECIP_HEAT", x=x, ix=ix, y=y, ncase=ncase,
        nx=16, nix=1, ny=16)
    if not ncase:
        return None
    return context.submit_kernel(
        "noahmp_vegprecip_precip_heat", arrays, {"ncase": ncase},
        exact_policy=False,
        grid_size=((ncase + _THREADS - 1) // _THREADS) * _THREADS,
        threads_per_group=_THREADS)


__all__ = ["submit_phenology", "submit_precip_heat"]
