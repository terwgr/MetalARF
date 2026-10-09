"""Typed Metal launchers for WRF topographic shortwave kernels.

The shadow mask is an int32 Metal resource; the shared submitter validates
it against the registered storage type like any other resource.
"""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def _domain(state, *arrays):
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("topographic radiation requires a Metal DomainState")
    if any(not isinstance(a, MetalArray) for a in arrays):
        raise TypeError("topographic radiation resources must be MetalArray")
    if any(a.dtype != np.dtype(np.float32) for a in arrays):
        raise TypeError("topographic radiation float resources must be FP32")
    if not arrays:
        raise ValueError("topographic radiation requires at least one resource")
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("topographic radiation resources must have matching shapes")
    if len(shape) != 2 or not all(shape):
        raise ValueError("topographic radiation expects nonempty (ny, nx) arrays")
    return shape


def submit_slope_geometry(state, ht, msftx, msfty, sina, cosa, *, dx, dy,
                          slope, slp_azi):
    ny, nx = _domain(state, ht, msftx, msfty, sina, cosa, slope, slp_azi)
    return state._array_module.context.submit_kernel(
        "topo_slope_geometry",
        {"ht": ht, "msftx": msftx, "msfty": msfty, "sina": sina,
         "cosa": cosa, "slope": slope, "slp_azi": slp_azi},
        {"rdx": float(np.float32(1.0) / np.float32(dx)),
         "rdy": float(np.float32(1.0) / np.float32(dy)), "nx": nx, "ny": ny},
        exact_policy=False, grid_size=nx * ny)


def submit_shadow_init(state, ht_loc, ht_shad, shadowmask, *, nested):
    if not isinstance(shadowmask, MetalArray) or shadowmask.dtype != np.dtype(np.int32):
        raise TypeError("shadowmask must be an int32 MetalArray")
    ny, nx = _domain(state, ht_shad)
    if ht_loc.shape != (ny + 4, nx + 4) or shadowmask.shape != (ny, nx):
        raise ValueError("shadow-init resources have incompatible halo/domain shapes")
    return state._array_module.context.submit_kernel(
        "topo_shadow_init",
        {"ht_loc": ht_loc, "ht_shad": ht_shad, "shadowmask": shadowmask},
        {"nested": int(bool(nested)), "nx": nx, "ny": ny},
        exact_policy=False, grid_size=nx * ny)


def submit_shadow_scan(state, ht_loc, xlat, xlong, sina, cosa, shadowmask,
                       ht_shad, *, xtime, gmt, radfrq, declin, dx, dy,
                       shadlen):
    if not isinstance(shadowmask, MetalArray) or shadowmask.dtype != np.dtype(np.int32):
        raise TypeError("shadowmask must be an int32 MetalArray")
    ny, nx = _domain(state, xlat, xlong, sina, cosa, ht_shad)
    if ht_loc.shape != (ny + 4, nx + 4) or shadowmask.shape != (ny, nx):
        raise ValueError("shadow-scan resources have incompatible halo/domain shapes")
    return state._array_module.context.submit_kernel(
        "topo_shadow_scan",
        {"ht_loc": ht_loc, "xlat": xlat, "xlong": xlong, "sina": sina,
         "cosa": cosa, "shadowmask": shadowmask, "ht_shad": ht_shad},
        {"xtime": float(np.float32(xtime)), "gmt": float(np.float32(gmt)),
         "radfrq": float(np.float32(radfrq)),
         "declin": float(np.float32(declin)), "dx": float(np.float32(dx)),
         "dy": float(np.float32(dy)), "shadlen": float(np.float32(shadlen)),
         "nx": nx, "ny": ny},
        exact_policy=False, grid_size=nx * ny)


def submit_diffuse_fraction(state, coszen, swdown, ht, swddif,
                            diffuse_frac, *, solcon, scheme_splits_direct_beam):
    ny, nx = _domain(state, coszen, swdown, ht, swddif, diffuse_frac)
    return state._array_module.context.submit_kernel(
        "topo_diffuse_frac",
        {"coszen": coszen, "swdown": swdown, "ht": ht,
         "swddif": swddif, "diffuse_frac": diffuse_frac},
        {"solcon": float(np.float32(solcon)),
         "ruiz": int(not scheme_splits_direct_beam), "n": nx * ny,
         "nx": nx, "ny": ny},
        exact_policy=False, grid_size=nx * ny)


def submit_adjust(state, xlat, coszen, shadowmask, diffuse_frac, hrang,
                  slope, slp_azi, *, declin, swdown, gsw, swnorm, gswsave):
    if not isinstance(shadowmask, MetalArray) or shadowmask.dtype != np.dtype(np.int32):
        raise TypeError("shadowmask must be an int32 MetalArray")
    ny, nx = _domain(state, xlat, coszen, diffuse_frac, hrang, slope, slp_azi,
                     swdown, gsw, swnorm, gswsave)
    if shadowmask.shape != (ny, nx):
        raise ValueError("shadowmask shape differs from surface radiation fields")
    return state._array_module.context.submit_kernel(
        "topo_rad_adjust",
        {"xlat": xlat, "coszen": coszen, "shadowmask": shadowmask,
         "diffuse_frac": diffuse_frac, "hrang": hrang, "slope": slope,
         "slp_azi": slp_azi, "swdown": swdown, "gsw": gsw,
         "swnorm": swnorm, "gswsave": gswsave},
        {"declin": float(np.float32(declin)), "n": nx * ny, "nx": nx,
         "ny": ny},
        exact_policy=False, grid_size=nx * ny)


def submit_restore(state, swdown, gsw, swnorm, gswsave):
    ny, nx = _domain(state, swdown, gsw, swnorm, gswsave)
    return state._array_module.context.submit_kernel(
        "topo_rad_restore",
        {"swdown": swdown, "gsw": gsw, "swnorm": swnorm,
         "gswsave": gswsave},
        {"n": nx * ny, "nx": nx, "ny": ny},
        exact_policy=False, grid_size=nx * ny)
