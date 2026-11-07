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
_UCM_BUNDLES = {}
_UCM_STATUS = {}
_URBAN_TILE = tuple(n + "_urb2d_mosaic" for n in
                    "tr tb tg tc qc uc ts sh lh g rn".split())
_URBAN_TILE = _URBAN_TILE[:7] + ("ts_rul2d_mosaic",) + _URBAN_TILE[7:]
_URBAN_SOIL = tuple(n + "_urb3d_mosaic" for n in ("trl", "tbl", "tgl"))


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
                              urban=None, atmosphere=None, solar=None,
                              use_wudapt_lcz=0, urban_water_balance=None):
    """Run one thread per cell while retaining CUDA's descending tile order.

    The shader owns the tile-local Noah state and the weighted cell reduction;
    the caller passes the full tile planes without materializing host copies.
    """
    from woof.wrf_exact import ENABLED as WRF_EXACT
    if WRF_EXACT:
        raise NotImplementedError("strict Metal Noah mosaic solver is not supported")
    if "tsk" not in dev or not isinstance(dev["tsk"], MetalArray):
        raise TypeError("Metal Noah mosaic requires MetalArray tile state")
    if int(mosaic_cat) < 1:
        raise ValueError("mosaic_cat < 1 leaves no dominant tile")
    if len(dzs) != 4:
        raise ValueError("Metal Noah mosaic requires four soil layers")
    if int(itimestep) < 1:
        raise ValueError("Noah mosaic itimestep must be positive")
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
    urban_in = urban_out = ()
    entry = "noah_mosaic_column"
    extra = {}
    if urban is not None:
        entry = "noah_mosaic_ucm_column"
        urban_in, urban_out, extra, ucm_arrays, check = _urban_inputs(
            dev, urban, atmosphere, solar, use_wudapt_lcz,
            urban_water_balance, categories, mosaic_cat, ny, nx, context)
        arrays.update(ucm_arrays)
        context.submit_bookkeeping(urban_in)
    context.submit_kernel(
        entry, arrays,
        {**extra, "dt": float(np.float32(dt)), "lucats": int(params.lucats),
         "slcats": int(params.slcats), "isurban": int(categories.isurban),
         "isice": int(categories.isice),
         "xice_threshold": float(np.float32(xice_threshold)),
         "itimestep": int(itimestep), "frpcpn": int(bool(frpcpn)),
         "usemonalb": int(bool(usemonalb)), "rdlai2d": int(bool(rdlai2d)),
         "opt_thcnd": int(opt_thcnd), "mosaic_cat": int(mosaic_cat),
         "nlcz": len(categories.lcz), "ntile_soil": 4 * int(mosaic_cat),
         "nstate": 65, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ncol, threads_per_group=64)
    if urban is not None:
        context.submit_bookkeeping(urban_out)
        check()
    unpack_arrays = {f"{n}_a": value for n, value in
                     (*grid_state.items(), *soil_state.items())}
    unpack_arrays["state_fields"] = packed
    context.submit_kernel(
        "noah_mosaic_unpack", unpack_arrays,
        {"nstate": 65, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=ncol, threads_per_group=128)


def _urban_inputs(dev, urban, atmosphere, solar, use_wudapt_lcz,
                  urban_water_balance, categories, mosaic_cat, ny, nx,
                  context):
    """The UCM state bundle, tile planes and parameters of the urban column.

    CUDA hands the grid UCM state as a pointer table; Metal copies the planes
    into one bundle (device to device) in the same order and copies them back
    after the launch.
    """
    from woof.core.urban_ucm import (
        UCM_STATE_PLANES, _get, _jmonth, _describe_status,
        water_balance_params)
    module = MetalArrayModule(context)
    for name in _URBAN_TILE:
        _require(dev[name], name, (mosaic_cat, ny, nx), np.float32, context)
    for name in _URBAN_SOIL:
        _require(dev[name], name, (4 * mosaic_cat, ny, nx), np.float32,
                 context)
    nplanes = sum(1 if layers == 0 else layers for _, layers in UCM_STATE_PLANES)
    key = (id(context), ny, nx)
    held = _UCM_BUNDLES.get(key)
    if held is None or held[0] is not context:
        held = (context, module.empty((nplanes, ny, nx), dtype=np.float32))
        _UCM_BUNDLES[key] = held
    bundle = held[1]
    pairs_in, pairs_out, offset = [], [], 0
    for name, layers in UCM_STATE_PLANES:
        shape = (ny, nx) if layers == 0 else (layers, ny, nx)
        target = _require(_get(urban, name), name, shape, np.float32, context)
        view = bundle.view(shape, byte_offset=offset)
        pairs_in.append((target, view))
        pairs_out.append((view, target))
        offset += target.nbytes
    params = water_balance_params(urban.params, urban_water_balance)
    pkey = (id(context), id(params))
    cached = _UCM_BUNDLES.get(pkey)
    if cached is None or cached[0] is not params:
        table = np.asarray(params.table, dtype=np.float32).reshape(-1)
        glob = np.asarray(params.globals_, dtype=np.float32).reshape(-1)
        cached = (params, module.asarray(np.concatenate((table, glob))),
                  module.asarray(np.asarray(params.switches, dtype=np.int32)))
        _UCM_BUNDLES[pkey] = cached

    def plane0(name):
        value = atmosphere[name]
        return _require(value.view((ny, nx), byte_offset=value.byte_offset),
                        name, (ny, nx), np.float32, context)

    status = _UCM_STATUS.get(id(urban))
    if status is None or status[0] is not urban:
        status = (urban, module.zeros((1,), dtype=np.uint32))
        _UCM_STATUS[id(urban)] = status
    status[1].fill(0)
    arrays = {n: dev[n] for n in (*_URBAN_TILE, *_URBAN_SOIL)}
    arrays.update({
        "urban_state": bundle, "float_bundle": cached[1], "isw": cached[2],
        "frc": _require(urban.fields["frc_urb2d"], "frc_urb2d", (ny, nx),
                        np.float32, context),
        "u1": plane0("u"), "v1": plane0("v"),
        "hrang": _require(solar.hrang, "hrang", (ny, nx), np.float32, context),
        "ust": _require(dev["ust"], "ust", (ny, nx), np.float32, context),
        "err": status[1]})
    extra = {"nurban_state": nplanes, "n_utype": int(params.table.shape[0]),
             "float_bundle_size": int(cached[1].size),
             "natural": int(categories.natural),
             "use_lcz": int(use_wudapt_lcz), "jmonth": int(_jmonth(solar))}

    def check():
        flags = int(status[1].to_numpy()[0])
        if flags:
            _describe_status(flags)
    return tuple(pairs_in), tuple(pairs_out), extra, arrays, check
