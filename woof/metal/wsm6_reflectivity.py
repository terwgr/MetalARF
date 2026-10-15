"""Typed Metal launcher for WSM6 REFL_10CM columns."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from ..wrf_exact import ENABLED as WRF_EXACT


_TABLE_CONSTANTS = 45
_TABLE_WORDS = _TABLE_CONSTANTS + 5 * 50


def _radar_table_words(hail_opt: int) -> np.ndarray:
    from woof.core.refl import radar_init_wsm6, wsm6_rimed

    rc = radar_init_wsm6(hail_opt)
    rimed = wsm6_rimed(hail_opt)
    if rc.xcre != (4.0, 1.0, 4.0, 7.0):
        raise ValueError(
            "WSM6 radar table requires rain PSD exponents (4, 1, 4, 7)")
    constants = np.asarray((
        rc.k_w, rc.m_w_0.real, rc.m_w_0.imag,
        rc.m_i_0.real, rc.m_i_0.imag,
        rc.xam_r, rc.xam_s, rc.xam_g,
        3.1415926535897932384626434, rc.pi5, rc.lamda4,
        287.0, 0.622, 1.0e-10, 1.0e-9, 1.0e-10, 0.9,
        rc.xcsg[2], rc.xcsg[3], 0.176 / 0.93, 900.0, 1000.0,
        1.0e18, 10.0, 273.15, rc.xcrg[2], rc.xcrg[3], 8.0e6,
        rimed.n0g, rc.xocms, rc.xocmg, rc.xobms,
        1.0e-12, 1.0e-12, 1.0e-22, -0.001, -0.12,
        2.0e6, 1.0e11, 0.005, 0.99, -35.0, np.log(10.0),
        6.0, 0.25,
    ), dtype=np.float64)
    values = np.concatenate((constants, rc.xxds, rc.xdts, rc.xxdg,
                             rc.xdtg, rc.simpson))
    if values.shape != (_TABLE_WORDS,) or not np.isfinite(values).all():
        raise ValueError("WSM6 radar_init table has an invalid layout")
    return np.ascontiguousarray(values).view(np.uint64)


def launch_refl10cm_wsm6(qv, qr, qs, qg, t, p, refl, *, hail_opt=0):
    """Submit WSM6's source-faithful reflectivity diagnostic on Metal."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal WSM6 reflectivity supports ordinary arithmetic only")
    arrays = {
        "qv": qv, "qr": qr, "qs": qs, "qg": qg,
        "t": t, "p": p, "refl": refl,
    }
    shape = tuple(refl.shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError(f"WSM6 reflectivity needs a nonempty 3-D field, got {shape}")
    if shape[0] > 256:
        raise ValueError("WSM6 reflectivity supports at most 256 levels")
    if int(np.prod(shape, dtype=np.int64)) > 0x7fffffff:
        raise ValueError("WSM6 reflectivity size exceeds the int32 ABI")
    if any(not isinstance(value, MetalArray)
           or value.dtype != np.dtype(np.float32)
           or value.shape != shape
           or not value.flags.c_contiguous for value in arrays.values()):
        raise TypeError(
            "WSM6 reflectivity fields must be matching contiguous FP32 MetalArrays")
    context = refl.buffer.context
    if any(value.buffer.context is not context for value in arrays.values()):
        raise ValueError("WSM6 reflectivity fields must share one Metal context")

    table_words = _radar_table_words(hail_opt)
    tables = MetalArray(context.buffer(table_words.nbytes),
                        table_words.shape, np.uint64)
    tables.buffer.write(table_words.tobytes())
    nz, ny, nx = shape
    return context.submit_kernel(
        "refl10cm_wsm6_column", {**arrays, "tables": tables},
        {"nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=ny * nx, threads_per_group=32)
