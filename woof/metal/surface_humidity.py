"""Metal launchers for WRF's final surface-humidity diagnostic bounds."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from .runtime import MetalContext


def submit_cap_land_q2(q2: MetalArray, qv: MetalArray,
                       xland: MetalArray) -> None:
    """Apply ``xland < 1.5 ? min(q2, 1.05*qv1) : q2`` in place.

    ``qv`` is the contiguous mass-grid 3-D physics field. Its level-zero
    plane is an offset view over the same Metal allocation; no staging or
    host arithmetic occurs.
    """
    if not isinstance(q2, MetalArray) or q2.ndim != 2:
        raise TypeError("Metal land-Q2 capping requires a 2-D Metal q2")
    ny, nx = q2.shape
    if (q2.dtype != np.dtype(np.float32)
            or q2.strides != (nx * 4, 4)):
        raise TypeError("Metal land-Q2 q2 must be contiguous FP32")
    if (not isinstance(qv, MetalArray) or qv.dtype != np.dtype(np.float32)
            or qv.shape[1:] != (ny, nx)
            or qv.ndim != 3
            or qv.strides != (ny * nx * 4, nx * 4, 4)):
        raise TypeError(
            "Metal land-Q2 qv must be contiguous FP32 (nz, ny, nx)")
    if (not isinstance(xland, MetalArray)
            or xland.dtype != np.dtype(np.float32)
            or xland.shape != (ny, nx)
            or xland.strides != (nx * 4, 4)):
        raise TypeError("Metal land-Q2 xland must be contiguous FP32 (ny, nx)")
    context = q2.buffer.context
    if (qv.buffer.context is not context
            or xland.buffer.context is not context):
        raise ValueError("Metal land-Q2 arrays must share one context")
    n = ny * nx
    if n > 0xFFFFFFFF:
        raise ValueError("Metal land-Q2 grid index does not fit uint32")
    qv1 = qv.view((ny, nx), byte_offset=qv.byte_offset)
    context.submit_kernel(
        "surface_cap_land_q2",
        {"q2": q2, "qv1": qv1, "xland": xland},
        {"nx": nx, "ny": ny, "n": n},
        exact_policy=False, grid_size=n)


_ALL_LAND = {}


def submit_cap_surface_q2(q2: MetalArray, qv: MetalArray) -> None:
    """Apply ``min(q2, 1.05*qv1)`` to every column in place.

    The operational fork's bound (module_surface_driver.F:4117-4127) is the
    land kernel over an all-land mask, so it shares one launcher and ABI.
    """
    if not isinstance(q2, MetalArray) or q2.ndim != 2:
        raise TypeError("Metal surface-Q2 capping requires a 2-D Metal q2")
    context = q2.buffer.context
    key = (id(context), tuple(q2.shape))
    land = _ALL_LAND.get(key)
    if land is None or land[0] is not context:
        mask = MetalArray(context.buffer(max(4, q2.size * 4)), q2.shape, np.float32)
        mask[...] = np.ones(q2.shape, dtype=np.float32)
        land = _ALL_LAND[key] = (context, mask)
    submit_cap_land_q2(q2, qv, land[1])

