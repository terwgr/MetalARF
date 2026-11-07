"""Metal launchers for the bandwidth glue array operations."""

from __future__ import annotations

import numpy as np

from .array import MetalArray, overlaps
from woof.wrf_exact import ENABLED as WRF_EXACT


def _supported(arrays):
    return all(isinstance(a, MetalArray)
               and a.dtype == np.dtype(np.float32)
               and a.flags.c_contiguous for a in arrays)


def _submit_add(src, dst):
    n = dst.size
    if n:
        src.buffer.context.submit_kernel(
            "wrf_exact_glue_add" if WRF_EXACT else "glue_add",
            {"src": src.view((n,)), "dst": dst.view((n,))}, {"n": n},
            exact_policy=WRF_EXACT, grid_size=n)


def add_array(src, dst):
    """Metal counterpart to the CUDA fast path; return False for fallback."""
    if (not _supported((src, dst)) or src.shape != dst.shape
            or dst.size > 2_147_483_647
            or (overlaps(src, dst) and
                (src.buffer is not dst.buffer
                 or src.byte_offset != dst.byte_offset))):
        return False
    _submit_add(src, dst)
    return True


def add_positive_array(src, dst):
    """``dst += max(src, 0)`` on Metal; return False for an unsupported layout."""
    if (not _supported((src, dst)) or src.shape != dst.shape
            or dst.size > 2_147_483_647 or overlaps(src, dst)):
        return False
    n = dst.size
    if n:
        src.buffer.context.submit_kernel(
            "wrf_exact_glue_add_positive" if WRF_EXACT else "glue_add_positive",
            {"src": src.view((n,)), "dst": dst.view((n,))},
            {"n": n}, exact_policy=WRF_EXACT, grid_size=n)
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
        if (overlaps(src, dst) and
                (src.buffer is not dst.buffer
                 or src.byte_offset != dst.byte_offset)):
            return None
        if any(overlaps(dst, other)
               for j, pair in enumerate(pairs) if j != row
               for other in pair):
            return None

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
            or any(overlaps(dst, a) for a in arrays[:-1])):
        return False
    state._array_module.context.submit_kernel(
        "wrf_exact_capture_theta_forcing" if WRF_EXACT else "glue_capture_theta_forcing",
        {"rth": state.rth_t, "mub": state.mub2d, "mup": state.mup,
         "c1": state.c1h, "c2": state.c2h, "msft": state.msft,
         "dst": dst},
        {"n": dst.size, "ncol": ny * nx,
         "mapped": int(bool(state.has_msf)), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=WRF_EXACT, grid_size=dst.size)
    return True


def map_tendency(src, factor, dst, *, add):
    """Apply a 2-D map factor to a 3-D tendency with two FP32 rounds."""
    arrays = (src, factor, dst)
    if (not _supported(arrays) or src.ndim != 3 or src.shape != dst.shape
            or factor.shape != src.shape[1:] or src.size > np.iinfo(np.int32).max
            or any(a.buffer.context is not src.buffer.context for a in arrays)
            or overlaps(dst, factor) or (overlaps(src, dst) and
                (src.buffer is not dst.buffer or src.byte_offset != dst.byte_offset))):
        raise TypeError("Metal mapped tendency needs matching contiguous FP32 arrays")
    nz, ny, nx = src.shape
    if src.size:
        src.buffer.context.submit_kernel(
            "wrf_exact_map_add" if add else "wrf_exact_map_copy",
            {"src": src, "map": factor, "dst": dst},
            {"n": src.size, "ncol": ny * nx, "nz": nz, "ny": ny, "nx": nx},
            exact_policy=True, grid_size=src.size)


def accumulate_scaled_array(src, dst, scale):
    """Two binary32 CuPy ufunc rounds, including their CUDA FTZ behavior."""
    if (not _supported((src, dst)) or src.shape != dst.shape
            or src.buffer.context is not dst.buffer.context
            or src.size > np.iinfo(np.int32).max
            or (overlaps(src, dst) and src.byte_offset != dst.byte_offset)):
        raise TypeError("Metal flux accumulation requires matching contiguous FP32 rows")
    factor = np.float32(scale)
    if not np.isfinite(factor):
        raise ValueError("Metal flux accumulation scale must be finite")
    if src.size:
        src.buffer.context.submit_kernel(
            "wrf_exact_accumulate_scaled" if WRF_EXACT else "glue_accumulate_scaled",
            {"src": src.view((src.size,)), "dst": dst.view((dst.size,))},
            {"n": src.size, "scale": float(factor)},
            exact_policy=WRF_EXACT, grid_size=src.size, threads_per_group=128)


def direct_normal_flux(direct, cosine, normal):
    """Native beam divided by its own zenith cosine, zero at night."""
    arrays = (direct, cosine, normal)
    if (not _supported(arrays) or any(a.shape != normal.shape for a in arrays)
            or any(a.buffer.context is not normal.buffer.context for a in arrays)
            or normal.size > np.iinfo(np.int32).max
            or overlaps(normal, direct) or overlaps(normal, cosine)):
        raise TypeError("Metal direct-normal flux needs separate matching FP32 rows")
    if normal.size:
        if WRF_EXACT:
            if not np.isfinite(direct.to_numpy()).all() or not np.isfinite(cosine.to_numpy()).all():
                raise ValueError("strict direct-normal flux requires finite inputs")
            from .array import MetalArrayModule
            result = MetalArrayModule(normal.buffer.context).empty(normal.shape, dtype=np.float32)
            normal.buffer.context.submit_kernel(
                "wrf_exact_direct_normal_flux",
                {"direct": direct.view((normal.size,)),
                 "cosine": cosine.view((normal.size,)), "normal": result.view((normal.size,))},
                {"n": normal.size}, exact_policy=True,
                grid_size=normal.size, threads_per_group=128)
            if not np.isfinite(result.to_numpy()).all():
                raise ValueError("strict direct-normal flux produced a nonfinite result")
            normal[...] = result
            return
        normal.buffer.context.submit_kernel(
            "glue_direct_normal_flux",
            {"direct": direct.view((normal.size,)),
             "cosine": cosine.view((normal.size,)), "normal": normal.view((normal.size,))},
            {"n": normal.size}, exact_policy=False,
            grid_size=normal.size, threads_per_group=128)


def add_product_rn(a, b, c, out):
    """a + RN(b*c), rounded twice with gradual binary32 underflow."""
    arrays = (a, b, c, out)
    if (not _supported(arrays) or any(v.shape != out.shape for v in arrays)
            or any(v.buffer.context is not out.buffer.context for v in arrays)
            or out.size > np.iinfo(np.int32).max
            or any(overlaps(out, v) for v in arrays[:-1])):
        raise TypeError("Metal add-product needs distinct matching contiguous FP32 rows")
    if out.size:
        out.buffer.context.submit_kernel(
            "glue_add_product_rn",
            {name: value.view((out.size,)) for name, value in zip(("a", "b", "c", "out"), arrays)},
            {"n": out.size}, exact_policy=False,
            grid_size=out.size, threads_per_group=128)
