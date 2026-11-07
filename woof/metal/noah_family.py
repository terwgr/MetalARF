"""Typed Metal launchers for the ordinary Noah land-surface column."""

from __future__ import annotations

import numpy as np

from .array import MetalArray, MetalArrayModule


_DEVICE_TABLES: dict[tuple, tuple] = {}
_URBAN_TABLES: dict[tuple, tuple] = {}
_F2D = (
    "psfc", "sfcprs", "sfctmp", "qv1", "qgh", "dz8w1", "glw",
    "swdown", "rainbl", "sr", "chs", "cqs2", "chs2", "rib",
    "vegfra", "shdmin", "shdmax", "tmn", "xland", "xice", "snoalb",
    "embck", "tsk", "hfx", "qfx", "lh", "grdflx", "qsfc", "canwat",
    "snow", "snowc", "snowh", "albedo", "albbck", "emiss", "znt",
    "z0", "snotime", "lai", "smstav", "smstot", "sfcrunoff",
    "udrunoff", "acsnow", "acsnom", "snopcx", "potevp", "noahres",
    "reslin", "chklowq",
)
_F3D = ("smois", "tslb", "sh2o", "smcrel")
_F2D_RW = frozenset((
    "chs2", "tsk", "hfx", "qfx", "lh", "grdflx", "qsfc", "canwat",
    "snow", "snowc", "snowh", "albedo", "albbck", "emiss", "znt",
    "z0", "snotime", "lai", "smstav", "smstot", "sfcrunoff",
    "udrunoff", "acsnow", "acsnom", "snopcx", "potevp", "noahres",
    "reslin", "chklowq",
))


def _device_tables(params, dzs, context):
    dzs32 = np.asarray(dzs, dtype=np.float32)
    if dzs32.shape != (4,):
        raise ValueError("Metal Noah requires four soil-layer depths")
    key = (id(context), id(params), tuple(float(x) for x in dzs32))
    cached = _DEVICE_TABLES.get(key)
    if cached is not None and cached[0] is params and cached[1] is context:
        return cached[2]
    xp = MetalArrayModule(context)
    tables = (
        xp.asarray(np.asarray(params.veg, dtype=np.float32).reshape(
            params.lucats, 15)),
        xp.asarray(np.asarray(params.soil, dtype=np.float32).reshape(
            params.slcats, 10)),
        xp.asarray(np.asarray(params.gen, dtype=np.float32)),
        xp.asarray(dzs32),
    )
    _DEVICE_TABLES[key] = (params, context, tables)
    return tables


def _require_array(value, name, shape, dtype, context):
    if (not isinstance(value, MetalArray) or value.shape != tuple(shape)
            or value.dtype != np.dtype(dtype)
            or value.strides != _contiguous_strides(shape, np.dtype(dtype).itemsize)):
        raise ValueError(
            f"Metal Noah {name} must be contiguous {np.dtype(dtype)} with "
            f"shape {tuple(shape)}")
    if value.buffer.context is not context:
        raise ValueError("Metal Noah inputs must share one context")
    return value


def _contiguous_strides(shape, itemsize):
    stride = itemsize
    result = []
    for extent in reversed(tuple(shape)):
        result.append(stride)
        stride *= max(int(extent), 1)
    return tuple(reversed(result))


def _validate_tables(params, tables, context):
    veg, soil, gen, dzs = tables
    for value, name, shape in (
            (veg, "vegetation table", (params.lucats, 15)),
            (soil, "soil table", (params.slcats, 10)),
            (gen, "general table", (16,)), (dzs, "layer depths", (4,))):
        _require_array(value, name, shape, np.float32, context)


def _validate_column_inputs(dev, params):
    from woof.core.noah import NUM_SOIL_LAYERS

    if len(dev["smois"].shape) != 3:
        raise ValueError("Metal Noah requires (4, ny, nx) soil arrays")
    layers, ny, nx = dev["smois"].shape
    if layers != NUM_SOIL_LAYERS or ny <= 0 or nx <= 0:
        raise ValueError("Metal Noah requires four non-empty soil layers")
    context = dev["smois"].buffer.context
    _require_array(dev["ivgtyp"], "ivgtyp", (ny, nx), np.int32, context)
    _require_array(dev["isltyp"], "isltyp", (ny, nx), np.int32, context)
    for name in _F2D:
        _require_array(dev[name], name, (ny, nx), np.float32, context)
    for name in _F3D:
        _require_array(dev[name], name, (layers, ny, nx), np.float32, context)
    _require_array(dev["ebal"], "ebal", (ny, nx), np.int32, context)
    return ny, nx, context


def _column_arrays(dev, tables):
    vegtbl, soiltbl, genp, dzs = tables
    arrays = {"ivgtyp": dev["ivgtyp"], "isltyp": dev["isltyp"]}
    arrays.update({name: dev[name] for name in _F2D})
    arrays.update({name: dev[name] for name in _F3D})
    arrays.update({"ebal": dev["ebal"], "vegtbl": vegtbl,
                   "soiltbl": soiltbl, "genp": genp, "dzs": dzs})
    return arrays


def submit_noah_column(dev: dict, params, dt: float, dzs, *,
                       isurban: int = 13, isice: int = 15,
                       xice_threshold: float = 0.5, frpcpn: bool = False,
                       usemonalb: bool = False, rdlai2d: bool = False,
                       opt_thcnd: int = 1, itimestep: int = 2, urban=None) -> None:
    """Run the ordinary or urban-handoff Noah instantiation."""
    from woof.wrf_exact import ENABLED as WRF_EXACT
    if WRF_EXACT and urban is not None and int(urban["option"]) != 1:
        raise NotImplementedError("strict Metal Noah supports only single-layer urban handoff")
    if WRF_EXACT and urban is not None and not 1 <= int(urban["natural"]) <= params.lucats:
        raise ValueError("strict Metal Noah urban natural category must be within the vegetation table")
    ny, nx, context = _validate_column_inputs(dev, params)
    if (isinstance(itimestep, bool) or not isinstance(itimestep, int)
            or itimestep < 1):
        raise ValueError("Noah itimestep must be a positive integer")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in (isurban, isice, opt_thcnd)):
        raise TypeError("Noah category and thermal selectors must be integers")
    tables = _device_tables(params, dzs, context)
    _validate_tables(params, tables, context)
    arrays = _column_arrays(dev, tables)
    parameters = {"dt": float(np.float32(dt)),
         "lucats": int(params.lucats), "slcats": int(params.slcats),
         "isurban": isurban, "isice": isice,
         "xice_threshold": float(np.float32(xice_threshold)),
         "itimestep": itimestep, "frpcpn": int(bool(frpcpn)),
         "usemonalb": int(bool(usemonalb)), "rdlai2d": int(bool(rdlai2d)),
         "opt_thcnd": opt_thcnd, "ny": ny, "nx": nx}
    entry = "wrf_exact_noah_column" if WRF_EXACT else "noah_column"
    if urban is not None:
        option = int(urban["option"])
        if option not in (1, 2, 3):
            raise ValueError("Noah urban option must be 1, 2, or 3")
        mask = urban["category_mask"]
        if not isinstance(mask, MetalArray) or mask.ndim != 1:
            raise ValueError("Noah urban category mask must be a resident vector")
        arrays["urban_cat"] = _require_array(mask, "urban_cat", mask.shape,
                                              np.int32, context)
        for name in ("frc_urb2d", "ts_urb2d", "tsk_rural_bep", "rural_q1",
                     "rural_q2k", "rural_zlvl"):
            value = urban.get(name)
            if value is None and name == "tsk_rural_bep" and option == 1:
                value = urban["ts_urb2d"]
            arrays[name] = _require_array(value, name, (ny, nx), np.float32, context)
        parameters.update(urban_opt=option, urban_cat_size=mask.size,
                          natural=int(urban["natural"]))
        key = (id(context), id(params), ny, nx, tuple(float(v) for v in dzs))
        cached = _URBAN_TABLES.get(key)
        if cached is None or cached[0] is not params or cached[1] is not context:
            packed = np.concatenate((
                np.asarray(params.veg, dtype=np.float32).reshape(-1),
                np.asarray(params.soil, dtype=np.float32).reshape(-1),
                np.asarray(params.gen, dtype=np.float32).reshape(-1),
                np.asarray(dzs, dtype=np.float32).reshape(-1)))
            # The two time-varying urban planes follow immutable tables.
            module = MetalArrayModule(context)
            device = module.empty((packed.size + 2 * ny * nx,), dtype=np.float32)
            device.view((packed.size,))[...] = packed
            cached = (params, context, device)
            _URBAN_TABLES[key] = cached
        for name in ("vegtbl", "soiltbl", "genp", "dzs"):
            arrays.pop(name)
        offset = (params.lucats * 15 + params.slcats * 10 + 20) * 4
        for name in ("frc_urb2d", "ts_urb2d"):
            destination = cached[2].view((ny, nx), byte_offset=offset)
            context.submit_bookkeeping(((arrays.pop(name), destination),))
            offset += ny * nx * 4
        arrays["tables"] = cached[2]
        parameters["table_size"] = cached[2].size
        entry = "wrf_exact_noah_column_urban" if WRF_EXACT else "noah_column_urban"
    context.submit_kernel(entry, arrays, parameters,
        exact_policy=WRF_EXACT, grid_size=ny * nx, threads_per_group=64)


def initialize_noah_liquid_water(dev: dict, params, dzs) -> None:
    """Match LSMINIT: validate soil categories, then initialize SH2O on Metal."""
    from woof.core.noah import NUM_SOIL_LAYERS

    moisture = dev["smois"]
    if (not isinstance(moisture, MetalArray) or moisture.ndim != 3
            or moisture.shape[0] != NUM_SOIL_LAYERS):
        raise ValueError("Noah initialization requires four soil layers")
    _, ny, nx = moisture.shape
    context = moisture.buffer.context
    _require_array(moisture, "smois", moisture.shape, np.float32, context)
    isltyp = _require_array(dev["isltyp"], "isltyp", (ny, nx),
                            np.int32, context)
    tslb = _require_array(dev["tslb"], "tslb", moisture.shape,
                          np.float32, context)
    sh2o = _require_array(dev["sh2o"], "sh2o", moisture.shape,
                          np.float32, context)
    tables = _device_tables(params, dzs, context)
    _validate_tables(params, tables, context)
    _, soiltbl, _, _ = tables
    xp = MetalArrayModule(context)
    status = xp.zeros((1,), dtype=np.uint32)
    from woof.wrf_exact import ENABLED as WRF_EXACT
    if WRF_EXACT:
        scalars = {"slcats": int(params.slcats), "ny": ny, "nx": nx}
        context.submit_kernel(
            "wrf_exact_noah_init_check",
            {"isltyp": isltyp, "tslb": tslb, "soiltbl": soiltbl, "status": status},
            scalars, exact_policy=True, grid_size=ny * nx, threads_per_group=64)
        flags = int(status.to_numpy()[0])
        if flags & 1:
            raise ValueError("isltyp category is outside the Noah soil table")
        if flags & 2:
            raise NotImplementedError(
                "strict Metal Noah frozen-soil initialization requires CUDA powf")
        context.submit_kernel(
            "wrf_exact_noah_init_copy", {"smois": moisture, "sh2o": sh2o},
            scalars, exact_policy=True, grid_size=ny * nx, threads_per_group=64)
        return
    context.submit_kernel(
        "noah_check_soil_categories",
        {"isltyp": isltyp, "status": status},
        {"slcats": int(params.slcats), "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=64)
    if int(status.to_numpy()[0]) != 0:
        raise ValueError("isltyp category is outside the Noah soil table")
    context.submit_kernel(
        "noah_initialize_sh2o",
        {"isltyp": isltyp, "smois": moisture, "tslb": tslb,
         "sh2o": sh2o, "soiltbl": soiltbl},
        {"slcats": int(params.slcats), "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=64)


def submit_noah_seaice(dev: dict, dt: float, *, frpcpn: bool,
                       xice_threshold: float, albedo_default: float,
                       check_snow_density: bool) -> None:
    """WRF's seaice_noah on Metal arrays (in place); see launch_noah_seaice."""
    from woof.core.noah import (
        NUM_SOIL_LAYERS, SEAICE_F2D, SEAICE_SNOWDEPTH_MAX_M,
        SEAICE_SNOWDEPTH_MIN_M, SEAICE_THICKNESS_DEFAULT_M)

    ny, nx = dev["tsk"].shape
    context = dev["tsk"].buffer.context
    arrays = {name: _require_array(dev[name], f"sea ice {name}", (ny, nx),
                                   np.float32, context)
              for name in SEAICE_F2D}
    arrays["tslb"] = _require_array(dev["tslb"], "sea ice tslb",
                                    (NUM_SOIL_LAYERS, ny, nx), np.float32, context)
    arrays["fatal"] = fatal = MetalArrayModule(context).zeros((1,), dtype=np.int32)
    from woof.wrf_exact import ENABLED as WRF_EXACT
    context.submit_kernel(
        "wrf_exact_noah_seaice" if WRF_EXACT else "noah_seaice_column", arrays,
        {"dt": float(np.float32(dt)),
         "xice_threshold": float(np.float32(xice_threshold)),
         "frpcpn": int(bool(frpcpn)),
         "albedo_default": float(np.float32(albedo_default)),
         "sithick": float(np.float32(SEAICE_THICKNESS_DEFAULT_M)),
         "snowdepth_min": float(np.float32(SEAICE_SNOWDEPTH_MIN_M)),
         "snowdepth_max": float(np.float32(SEAICE_SNOWDEPTH_MAX_M)),
         "ny": ny, "nx": nx},
        exact_policy=WRF_EXACT, grid_size=ny * nx, threads_per_group=64)
    if check_snow_density and int(fatal.to_numpy()[0]) != 0:
        raise ValueError(
            "Noah sea ice: Physical snow depth is less than snow water equiv. "
            "(WRF SFLX_SEAICE's fatal): a sea-ice column has SNOW/1000 > SNOWH "
            "after the 0.01 m floor and the depth clamp, so its snow density "
            "exceeds 1 and the ice-column conduction would be computed from "
            "an impossible snowpack.  Fix the initial SNOW/SNOWH pair.")
