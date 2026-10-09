"""Typed Metal launcher for classic Thompson REFL_10CM columns."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from ..wrf_exact import ENABLED as WRF_EXACT


_TABLE_WORDS = 8 + 3 * 50


def _radar_table_words() -> np.ndarray:
    """Pack the canonical radar_init products used by Thompson's soak."""
    from woof.core.refl import radar_init

    rc = radar_init()
    constants = np.asarray((
        rc.k_w, rc.m_w_0.real, rc.m_w_0.imag,
        rc.m_i_0.real, rc.m_i_0.imag,
        3.1415926535897932384626434, rc.pi5, rc.lamda4,
    ), dtype=np.float64)
    values = np.concatenate((constants, rc.xxds, rc.xdts, rc.simpson))
    if values.shape != (_TABLE_WORDS,) or not np.isfinite(values).all():
        raise ValueError("Thompson radar_init table has an invalid layout")
    return np.ascontiguousarray(values).view(np.uint64)


def launch_refl10cm_thompson(
        qv, qr, nr, qs, qg, graupel_number_shadow, t, p, refl,
        *, tables: MetalArray | None = None):
    """Submit classic Thompson reflectivity for one output-due call.

    Arguments match ``woof.core.refl.launch_refl10cm_thompson``. The
    graupel-number scratch is supplied by the same Thompson call; it is not
    transported model state. Fields are contiguous FP32 MetalArrays with
    common ``(nz, ny, nx)`` shape. Radar tables/constants are sourced from
    the canonical ``woof.core.refl.radar_init()`` owner and uploaded here.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal Thompson reflectivity supports ordinary arithmetic only")
    arrays = {
        "qv": qv, "qr": qr, "nr": nr, "qs": qs, "qg": qg,
        "graupel_number_shadow": graupel_number_shadow,
        "t": t, "p": p, "refl": refl,
    }
    shape = tuple(refl.shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError(f"Thompson reflectivity needs a nonempty 3-D field, got {shape}")
    if shape[0] > 256:
        raise ValueError("Thompson reflectivity supports at most 256 levels")
    if int(np.prod(shape, dtype=np.int64)) > 0x7fffffff:
        raise ValueError("Thompson reflectivity size exceeds the int32 ABI")
    if any(not isinstance(value, MetalArray)
           or value.dtype != np.dtype(np.float32)
           or value.shape != shape
           or not value.flags.c_contiguous for value in arrays.values()):
        raise TypeError(
            "Thompson reflectivity fields must be matching contiguous FP32 MetalArrays")
    context = refl.buffer.context
    if any(value.buffer.context is not context for value in arrays.values()):
        raise ValueError("Thompson reflectivity fields must share one Metal context")

    table_words = _radar_table_words()
    if tables is None:
        tables = MetalArray(context.buffer(table_words.nbytes),
                            table_words.shape, np.uint64)
    elif (not isinstance(tables, MetalArray)
          or tables.dtype != np.dtype(np.uint64)
          or tables.shape != (_TABLE_WORDS,)
          or tables.buffer.context is not context):
        raise TypeError("Thompson radar tables must be a 158-word uint64 MetalArray")
    tables.buffer.write(table_words.tobytes())
    nz, ny, nx = shape
    return context.submit_kernel(
        "refl10cm_thompson_column", {**arrays, "tables": tables},
        {"nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=ny * nx, threads_per_group=32)
