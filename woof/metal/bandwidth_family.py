"""Metal launchers for the bandwidth glue array operations."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


def _supported(arrays):
    return all(isinstance(a, MetalArray)
               and a.dtype == np.dtype(np.float32)
               and a.flags.c_contiguous for a in arrays)


def _overlap(a, b):
    return (a.buffer is b.buffer
            and a.byte_offset < b.byte_offset + b.nbytes
            and b.byte_offset < a.byte_offset + a.nbytes)


def _submit_add(src, dst):
    n = dst.size
    if n:
        src.buffer.context.submit_kernel(
            "glue_add", {"src": src.view((n,)), "dst": dst.view((n,))}, {"n": n},
            exact_policy=False, grid_size=n)


def add_array(src, dst):
    """Metal counterpart to the CUDA fast path; return False for fallback."""
    if (not _supported((src, dst)) or src.shape != dst.shape
            or dst.size > 2_147_483_647
            or (_overlap(src, dst) and
                (src.buffer is not dst.buffer
                 or src.byte_offset != dst.byte_offset))):
        return False
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal bandwidth glue addition supports ordinary FP32 only")
    _submit_add(src, dst)
    return True


def prepare_add_arrays(state, pairs):
    """Return an ordered Metal batch launcher, or None for wrapper fallback."""
    if not pairs or len(pairs) > 64:
        return None
    arrays = tuple(a for src, dst in pairs for a in (src, dst))
    if (not _supported(arrays)
            or any(src.shape != dst.shape or dst.size > 2_147_483_647
                   for src, dst in pairs)):
        return None
    key = tuple((id(a.buffer), a.byte_offset, a.size) for a in arrays)
    cached = getattr(state, "_glue_add_launch", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    for row, (src, dst) in enumerate(pairs):
        if (_overlap(src, dst) and
                (src.buffer is not dst.buffer
                 or src.byte_offset != dst.byte_offset)):
            return None
        if any(_overlap(dst, other)
               for j, pair in enumerate(pairs) if j != row
               for other in pair):
            return None
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal bandwidth glue addition supports ordinary FP32 only")

    def launch():
        # Preserve the wrapper's ordered fallback semantics if rows alias in
        # a way admitted by its destination-vs-other-row checks.
        for src, dst in pairs:
            _submit_add(src, dst)

    setattr(state, "_glue_add_launch", (key, launch))
    return launch


def capture_theta_forcing(state):
    """Fuse the FP32 theta-rate conversion into the destination array."""
    dst = state.rthften
    arrays = (state.rth_t, state.mub2d, state.mup, state.c1h, state.c2h,
              state.msft, dst)
    if not _supported(arrays) or state.rth_t.ndim != 3:
        return False
    nz, ny, nx = state.rth_t.shape
    if (dst.shape != (nz, ny, nx) or not dst.size
            or dst.size > 2_147_483_647 or ny * nx > 2_147_483_647
            or any(a.shape != (ny, nx) for a in arrays[1:3] + arrays[5:6])
            or any(a.shape != (nz,) for a in arrays[3:5])
            or any(_overlap(dst, a) for a in arrays[:-1])):
        return False
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal theta forcing capture supports ordinary FP32 only")
    state._array_module.context.submit_kernel(
        "glue_capture_theta_forcing",
        {"rth": state.rth_t, "mub": state.mub2d, "mup": state.mup,
         "c1": state.c1h, "c2": state.c2h, "msft": state.msft,
         "dst": dst},
        {"n": dst.size, "ncol": ny * nx,
         "mapped": int(bool(state.has_msf)), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=dst.size)
    return True
