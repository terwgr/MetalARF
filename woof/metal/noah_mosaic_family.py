"""Typed caller for the non-urban Noah mosaic column on Metal."""

from __future__ import annotations

import numpy as np

from .array import MetalArray, MetalArrayModule


_F2D = tuple("psfc sfcprs sfctmp qv1 qgh dz8w1 glw swdown rainbl sr chs "
             "cqs2 chs2 rib vegfra shdmin shdmax tmn xland xice snoalb "
             "embck tsk hfx qfx lh grdflx qsfc canwat snow snowc snowh "
             "albedo albbck emiss znt z0 snotime lai smstav smstot sfcrunoff "
             "udrunoff acsnow acsnom snopcx potevp noahres chklowq".split())
_F3D = ("smois", "tslb", "sh2o", "smcrel")
_MOSAIC_TILE = tuple(n + "_mosaic" for n in
    "tsk qsfc canwat snow snowh snowc albedo albbck emiss embck znt z0 "
    "hfx qfx lh grdflx snotime".split())
_MOSAIC_SOIL = ("tslb_mosaic", "smois_mosaic", "sh2o_mosaic")
_PACKED_STATE = {}


def _require(value, name, shape, dtype, context):
    if (not isinstance(value, MetalArray) or value.shape != tuple(shape)
            or value.dtype != np.dtype(dtype) or not value.flags.c_contiguous):
        raise ValueError(f"Metal Noah mosaic {name} must be contiguous "
                         f"{np.dtype(dtype)} with shape {tuple(shape)}")
    if value.buffer.context is not context:
        raise ValueError("Metal Noah mosaic arrays must share one context")
    return value


def submit_noah_mosaic_column(dev, params, dt, dzs, *, mosaic_cat,
                              categories, xice_threshold, frpcpn,
                              usemonalb, rdlai2d, opt_thcnd, itimestep,
                              urban=None):
    """Run one thread per cell while retaining CUDA's descending tile order.

    The shader owns the tile-local Noah state and the weighted cell reduction;
    the caller passes the full tile planes without materializing host copies.
    """
    if "tsk" not in dev or not isinstance(dev["tsk"], MetalArray):
        raise TypeError("Metal Noah mosaic requires MetalArray tile state")
    if int(mosaic_cat) < 1:
        raise ValueError("mosaic_cat < 1 leaves no dominant tile")
    if len(dzs) != 4:
        raise ValueError("Metal Noah mosaic requires four soil layers")
    if int(itimestep) < 1:
        raise ValueError("Noah mosaic itimestep must be positive")
    if urban is not None:
        raise NotImplementedError("Metal Noah mosaic UCM is not registered")
    ny, nx = dev["tsk"].shape
    context = dev["tsk"].buffer.context
    from .noah_family import _device_tables
    vegtbl, soiltbl, genp, dzs_dev = _device_tables(params, dzs, context)
    grid_state = {n: dev[n] for n in _F2D}
    soil_state = {n: dev[n] for n in _F3D}
    for name, value in grid_state.items():
        _require(value, name, (ny, nx), np.float32, context)
    for name, value in soil_state.items():
        _require(value, name, (4, ny, nx), np.float32, context)
    key = (id(context), ny, nx)
    cached = _PACKED_STATE.get(key)
    if cached is None or cached[0] is not context:
        cached = (context, MetalArrayModule(context).empty(
            (65, ny, nx), dtype=np.float32))
        _PACKED_STATE[key] = cached
    packed = cached[1]
    pack_arrays = {f"{n}_a": value for n, value in
                   (*grid_state.items(), *soil_state.items())}
    pack_arrays["state_fields"] = packed
    ncol = ny * nx
    context.submit_kernel(
        "noah_mosaic_pack", pack_arrays,
        {"nstate": 65, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=ncol, threads_per_group=128)
    arrays = {"ivgtyp": dev["ivgtyp"], "isltyp": dev["isltyp"],
              "state_fields": packed}
    arrays.update({n: dev[n] for n in _MOSAIC_TILE + _MOSAIC_SOIL})
    arrays.update({n: dev[n] for n in ("mosaic_cat_index", "landusef2")})
    lcz = MetalArrayModule(context).asarray(
        np.asarray(categories.lcz or (0,), dtype=np.int32))
    arrays.update({"lcz": lcz, "vegtbl": vegtbl, "soiltbl": soiltbl,
                   "genp": genp, "dzs": dzs_dev})
    for name, value in arrays.items():
        if name in ("ivgtyp", "isltyp"):
            shape, dtype = (ny, nx), np.int32
        elif name == "state_fields":
            shape, dtype = (65, ny, nx), np.float32
        elif name in _F3D:
            shape, dtype = (4, ny, nx), np.float32
        elif name in _MOSAIC_SOIL:
            shape, dtype = (4 * mosaic_cat, ny, nx), np.float32
        elif name in _MOSAIC_TILE or name == "landusef2":
            shape, dtype = (mosaic_cat, ny, nx), np.float32
        elif name == "mosaic_cat_index":
            shape, dtype = (mosaic_cat, ny, nx), np.int32
        elif name == "lcz":
            shape, dtype = (max(1, len(categories.lcz)),), np.int32
        elif name == "vegtbl":
            shape, dtype = (params.lucats, 15), np.float32
        elif name == "soiltbl":
            shape, dtype = (params.slcats, 10), np.float32
        elif name == "genp":
            shape, dtype = (16,), np.float32
        elif name == "dzs":
            shape, dtype = (4,), np.float32
        else:
            shape, dtype = (ny, nx), np.float32
        _require(value, name, shape, dtype, context)
    context.submit_kernel(
        "noah_mosaic_column", arrays,
        {"dt": float(np.float32(dt)), "lucats": int(params.lucats),
         "slcats": int(params.slcats), "isurban": int(categories.isurban),
         "isice": int(categories.isice),
         "xice_threshold": float(np.float32(xice_threshold)),
         "itimestep": int(itimestep), "frpcpn": int(bool(frpcpn)),
         "usemonalb": int(bool(usemonalb)), "rdlai2d": int(bool(rdlai2d)),
         "opt_thcnd": int(opt_thcnd), "mosaic_cat": int(mosaic_cat),
         "nlcz": len(categories.lcz), "ntile_soil": 4 * int(mosaic_cat),
         "nstate": 65, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ncol, threads_per_group=64)
    unpack_arrays = {f"{n}_a": value for n, value in
                     (*grid_state.items(), *soil_state.items())}
    unpack_arrays["state_fields"] = packed
    context.submit_kernel(
        "noah_mosaic_unpack", unpack_arrays,
        {"nstate": 65, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=ncol, threads_per_group=128)


def require_noah_mosaic_metal_supported(cuda_symbol: str, *, urban=False) -> None:
    """Keep only the separately unported urban pointer-table ABI gated."""
    if cuda_symbol != "noah_mosaic_column":
        raise NotImplementedError(
            f"Metal {cuda_symbol} has no typed entry point in this caller")
    if urban:
        raise NotImplementedError(
            "Metal Noah mosaic UCM still needs the UCM state pointer-table ABI")
