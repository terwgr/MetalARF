"""Typed Metal launcher boundary for WRF's double-moment WDM6 column."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


_VOLUME = ("theta", "qv", "qc", "qi", "qr", "qs", "qg", "nn", "nc",
          "nr", "den", "p", "pii", "dz", "effc", "effi", "effs")
_SURFACE = ("xland", "rainnc", "rainncv", "snownc", "snowncv",
            "graupelnc", "graupelncv", "sr")


def submit_wdm6_column(*, theta, qv, qc, qi, qr, qs, qg, nn, nc, nr, den,
                       p, pii, dz, xland, rainnc, rainncv, snownc, snowncv,
                       graupelnc, graupelncv, sr, effc, effi, effs, dt,
                       hail_opt=0, count_status=None):
    """Submit one source-equivalent thread per column and report CUDA status.

    ``count_status`` is a contiguous int32 view of a one-word buffer. The
    shader reinterprets its storage as ``atomic_uint`` and ORs the same low
    status bits as CUDA. The runtime's generated signature explicitly declares
    int32 storage, so the contract remains typed and the word is read after
    dispatch by :func:`woof.core.wdm6.launch_wdm6`.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal WDM6 does not implement WRF exact-D arithmetic")
    arrays = {
        "theta": theta, "qv": qv, "qc": qc, "qi": qi, "qr": qr,
        "qs": qs, "qg": qg, "nn": nn, "nc": nc, "nr": nr,
        "den": den, "p": p, "pii": pii, "dz": dz, "effc": effc,
        "effi": effi, "effs": effs, "xland": xland,
        "rainnc": rainnc, "rainncv": rainncv, "snownc": snownc,
        "snowncv": snowncv, "graupelnc": graupelnc,
        "graupelncv": graupelncv, "sr": sr,
    }
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("Metal WDM6 theta must be a three-dimensional MetalArray")
    nz, ny, nx = map(int, theta.shape)
    from woof.core.wdm6_constants import wdm6_level_tier
    tier = wdm6_level_tier(nz)
    if min(ny, nx) <= 0 or max(nz, ny, nx, ny * nx) > 0x7FFFFFFF:
        raise ValueError("Metal WDM6 dimensions are empty or exceed int32 indexing")
    volume_shape, surface_shape = (nz, ny, nx), (ny, nx)
    context = theta.buffer.context
    for name, array in arrays.items():
        if not isinstance(array, MetalArray):
            raise TypeError(f"Metal WDM6 {name} must be a MetalArray")
        expected = surface_shape if name in _SURFACE else volume_shape
        if array.shape != expected:
            raise ValueError(f"Metal WDM6 {name} shape {array.shape} != {expected}")
        if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
            raise TypeError(f"Metal WDM6 {name} requires contiguous float32 storage")
        if array.buffer.context is not context:
            raise ValueError("Metal WDM6 resources must share one context")
    if not isinstance(hail_opt, (int, np.integer)) or int(hail_opt) not in (0, 1):
        raise ValueError(f"WDM6 hail_opt must be 0 or 1, got {hail_opt!r}")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0:
        raise ValueError("Metal WDM6 dt must be finite and positive FP32")
    if (not isinstance(count_status, MetalArray)
            or count_status.dtype != np.dtype(np.int32)
            or count_status.shape != (1,)
            or not count_status.flags.c_contiguous
            or count_status.buffer.context is not context):
        raise TypeError(
            "Metal WDM6 needs a one-word int32 MetalArray view of its uint32 "
            "column-status storage")
    params = {
        "dt": float(dt32), "hail_opt": int(hail_opt),
        "nz": nz, "ny": ny, "nx": nx, "status_words": 1,
    }
    return context.submit_kernel(
        "wdm6_column", {**arrays, "count_status": count_status}, params,
        exact_policy=False, grid_size=ny * nx, threads_per_group=32)


__all__ = ["submit_wdm6_column"]
