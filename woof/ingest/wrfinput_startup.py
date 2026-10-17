"""Native WRF cold-start vertical velocity initialization."""
from __future__ import annotations

import numpy as np


def initialize_wrfinput_vertical_velocity(
        state, restored, cfg, *, use_input_w=False,
        periodic_x=False, periodic_y=False):
    """Apply WRF v4.7.1 start_em.F:1475-1532 to a cold imported state.

    WRF diagnoses a near-zero input surface W regardless of use_input_w.
    Otherwise use_input_w preserves the file when true and diagnoses when
    false. The caller supplies the producing namelist's periodic controls.
    Both current and previous time levels receive the initialized column.
    """
    for name, value in (("use_input_w", use_input_w),
                        ("periodic_x", periodic_x), ("periodic_y", periodic_y)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be the producing WRF logical value")
    nz, ny, nx = state.u.shape[0], state.w.shape[-2], state.w.shape[-1]
    if nz < 3 or state.w.shape != (nz + 1, ny, nx):
        raise ValueError("WRF cold-start W requires three wind levels and one extra W interface")
    if state.u.shape != (nz, ny, nx + 1) or state.v.shape != (nz, ny + 1, nx):
        raise ValueError("WRF cold-start W requires native staggered U and V dimensions")
    for name in ("u", "v", "w", "ht", "znw"):
        value = getattr(state, name)
        if value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous:
            raise ValueError(f"WRF cold-start W needs contiguous stored FP32 {name} for the kernel layout")
    metal = hasattr(state.w, "to_numpy")
    surface = state.w[0].to_numpy() if metal else state.w[0]
    threshold = float(np.float32(1.e-6))
    zero_surface = (abs(float(surface.max().item())) < threshold
                    and abs(float(surface.min().item())) < threshold)
    if use_input_w and not zero_surface:
        return False

    raw = restored.raw
    # Directional factors retain their producing WRF values; they are not
    # combined into the runtime's single isotropic map-factor carrier.
    mx_source = raw["MAPFAC_MX"] if "MAPFAC_MX" in raw else raw["MAPFAC_M"]
    my_source = raw["MAPFAC_MY"] if "MAPFAC_MY" in raw else raw["MAPFAC_M"]
    mx = np.ascontiguousarray(mx_source, dtype=np.float32)
    my = np.ascontiguousarray(my_source, dtype=np.float32)
    if mx.shape != (ny, nx) or my.shape != (ny, nx):
        raise ValueError("WRF cold-start W map factors must match the native mass grid")
    rdx = np.float32(np.float32(1.0) / np.float32(cfg.dx))
    rdy = np.float32(np.float32(1.0) / np.float32(cfg.dy))
    if metal:
        host = [np.asarray(a.to_numpy() if hasattr(a, "to_numpy") else a,
                           np.float32)
                for a in (state.u, state.v, state.ht, state.znw, state.w)]
        state.w[...] = _cold_start_w_host(
            *host[:3], mx, my, *host[3:], np.float32(state.cf1), np.float32(state.cf2),
            np.float32(state.cf3), rdx, rdy, bool(periodic_x), bool(periodic_y))
        if getattr(state, "w0", None) is not None:
            state.w0[...] = state.w
        return True
    import cupy as cp
    from woof.core.kernels import get_kernel

    mx, my = cp.asarray(mx), cp.asarray(my)
    kernel = get_kernel("wrf_cold_start_w", "wrf_cold_start_w")
    kernel(((ny * nx + 127) // 128,), (128,),
           (state.u, state.v, state.ht, mx, my, state.znw, state.w,
            np.float32(state.cf1), np.float32(state.cf2), np.float32(state.cf3),
            rdx, rdy, np.int32(periodic_x), np.int32(periodic_y),
            np.int32(nz), np.int32(ny), np.int32(nx)))
    if getattr(state, "w0", None) is not None:
        state.w0[...] = state.w
    return True


def _cold_start_w_host(u, v, ht, mx, my, znw, w, cf1, cf2, cf3, rdx, rdy,
                       periodic_x, periodic_y):
    """``wrf_cold_start_w`` on the host: each FP32 operation rounds once,
    in the kernel's order, with no contraction."""
    nz = u.shape[0]
    ny, nx = ht.shape
    f = np.float32
    i = np.arange(nx)
    j = np.arange(ny)
    im1 = np.where(i == 0, nx - 1 if periodic_x else 0, i - 1)
    ip1 = np.where(i == nx - 1, 0 if periodic_x else nx - 1, i + 1)
    jm1 = np.where(j == 0, ny - 1 if periodic_y else 0, j - 1)
    jp1 = np.where(j == ny - 1, 0 if periodic_y else ny - 1, j + 1)

    def level_mix(a):
        return (cf1 * a[0] + cf2 * a[1]) + cf3 * a[2]

    uc = level_mix(u)                  # (ny, nx + 1)
    vc = level_mix(v)                  # (ny + 1, nx)
    uc0, uc1 = uc[:, :-1], uc[:, 1:]
    vc0, vc1 = vc[:-1, :], vc[1:, :]
    dyn = ht[jp1, :] - ht
    dys = ht - ht[jm1, :]
    dxe = ht[:, ip1] - ht
    dxw = ht - ht[:, im1]
    yscale = (my * f(0.5)) * rdy
    xscale = (mx * f(0.5)) * rdx
    y = yscale * (dyn * vc1 + dys * vc0)
    x = xscale * (dxe * uc1 + dxw * uc0)
    surface = y + x
    out = np.array(w, dtype=np.float32, copy=True)
    out[0] = surface
    for k in range(1, nz + 1):
        out[k] = (surface * znw[k]) * znw[k]
    return out
