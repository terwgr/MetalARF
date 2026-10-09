"""Typed Metal launcher for the self-contained Noah-MP snow initializer."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_THREADS = 64
_SI_IN_STRIDE = 33
_SI_IX_STRIDE = 2
_SI_OUT_STRIDE = 22
_NI_IN_STRIDE = 58
_NI_IX_STRIDE = 21
_NI_OUT_STRIDE = 93
_I32_MAX = np.iinfo(np.int32).max


def submit_snow_init(*, x, ix, y, ncase):
    """Submit the pinned SNOW_INIT export over packed columns.

    ``x``/``y`` are the float32 slot-major column records and ``ix`` contains
    the int32 ``nsnow,nsoil`` dimensions, matching noahmp_driver_gpu.py.
    The caller owns all buffers and preserves the preinitialized input slots
    that SNOW_INIT intentionally leaves unchanged in its packed result.
    """
    if not isinstance(ncase, (int, np.integer)):
        raise TypeError("Metal Noah-MP driver ncase must be an integer")
    ncase = int(ncase)
    if not 0 <= ncase <= _I32_MAX:
        raise ValueError("Metal Noah-MP driver ncase must fit int32")
    arrays = {"x": x, "ix": ix, "y": y}
    shapes = {"x": (ncase, _SI_IN_STRIDE),
              "ix": (ncase, _SI_IX_STRIDE),
              "y": (ncase, _SI_OUT_STRIDE)}
    context = None
    for name, array in arrays.items():
        if not isinstance(array, MetalArray):
            raise TypeError(f"Metal Noah-MP driver {name} must be a MetalArray")
        expected_dtype = np.dtype(np.int32) if name == "ix" else np.dtype(np.float32)
        if array.dtype != expected_dtype or not array.flags.c_contiguous:
            raise TypeError(
                f"Metal Noah-MP driver {name} requires contiguous "
                f"{expected_dtype.name} storage")
        if array.shape != shapes[name]:
            raise ValueError(
                f"Metal Noah-MP driver {name} shape {array.shape} != {shapes[name]}")
        if context is None:
            context = array.buffer.context
        elif array.buffer.context is not context:
            raise ValueError("Metal Noah-MP driver arrays must share one context")
    if ncase == 0:
        return None
    if ncase > _I32_MAX - (_THREADS - 1):
        raise ValueError("Metal Noah-MP driver dispatch extent exceeds int32")
    # The CUDA batch wrapper allocates y with cp.zeros; mirror its untouched
    # tail slots when a legal column uses fewer than the fixed maxima.
    y.fill(np.float32(0.0))
    grid_size = ((ncase + _THREADS - 1) // _THREADS) * _THREADS
    return context.submit_kernel(
        "noahmp_driver_snow_init", arrays, {"ncase": ncase},
        exact_policy=True, grid_size=grid_size,
        threads_per_group=_THREADS)


def submit_noahmp_init(*, x, ix, y, ncase):
    """Submit NOAHMP_INIT for packed columns.

    The supercooled-liquid branch uses MSL's native ``pow`` at the pinned
    ``r_pow`` call site. Numerical parity is therefore not claimed here.
    The launcher clears ``y`` before dispatch, matching the CUDA wrapper's
    ``cp.zeros`` for output padding below live layer counts.
    """
    if not isinstance(ncase, (int, np.integer)):
        raise TypeError("Metal Noah-MP driver ncase must be an integer")
    ncase = int(ncase)
    if not 0 <= ncase <= _I32_MAX:
        raise ValueError("Metal Noah-MP driver ncase must fit int32")
    arrays = {"x": x, "ix": ix, "y": y}
    shapes = {"x": (ncase, _NI_IN_STRIDE),
              "ix": (ncase, _NI_IX_STRIDE),
              "y": (ncase, _NI_OUT_STRIDE)}
    context = None
    for name, array in arrays.items():
        if not isinstance(array, MetalArray):
            raise TypeError(f"Metal Noah-MP driver {name} must be a MetalArray")
        expected_dtype = np.dtype(np.int32) if name == "ix" else np.dtype(np.float32)
        if array.dtype != expected_dtype or not array.flags.c_contiguous:
            raise TypeError(
                f"Metal Noah-MP driver {name} requires contiguous "
                f"{expected_dtype.name} storage")
        if array.shape != shapes[name]:
            raise ValueError(
                f"Metal Noah-MP driver {name} shape {array.shape} != {shapes[name]}")
        if context is None:
            context = array.buffer.context
        elif array.buffer.context is not context:
            raise ValueError("Metal Noah-MP driver arrays must share one context")
    if ncase == 0:
        return None
    if ncase > _I32_MAX - (_THREADS - 1):
        raise ValueError("Metal Noah-MP driver dispatch extent exceeds int32")
    y.fill(np.float32(0.0))
    grid_size = ((ncase + _THREADS - 1) // _THREADS) * _THREADS
    return context.submit_kernel(
        "noahmp_driver_noahmp_init", arrays, {"ncase": ncase},
        exact_policy=False, grid_size=grid_size,
        threads_per_group=_THREADS)
