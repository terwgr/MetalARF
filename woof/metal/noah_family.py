"""Typed Metal launchers for the ordinary Noah land-surface column."""

from __future__ import annotations

import numpy as np

from .array import MetalArray, MetalArrayModule


_DEVICE_TABLES: dict[tuple, tuple] = {}
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
                       opt_thcnd: int = 1, itimestep: int = 2) -> None:
    """Run ordinary non-urban Noah, preserving the CUDA launch contract."""
    ny, nx, context = _validate_column_inputs(dev, params)
    if (isinstance(itimestep, bool) or not isinstance(itimestep, int)
            or itimestep < 1):
        raise ValueError("Noah itimestep must be a positive integer")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in (isurban, isice, opt_thcnd)):
        raise TypeError("Noah category and thermal selectors must be integers")
    tables = _device_tables(params, dzs, context)
    _validate_tables(params, tables, context)
    context.submit_kernel(
        "noah_column", _column_arrays(dev, tables),
        {"dt": float(np.float32(dt)),
         "lucats": int(params.lucats), "slcats": int(params.slcats),
         "isurban": isurban, "isice": isice,
         "xice_threshold": float(np.float32(xice_threshold)),
         "itimestep": itimestep, "frpcpn": int(bool(frpcpn)),
         "usemonalb": int(bool(usemonalb)), "rdlai2d": int(bool(rdlai2d)),
         "opt_thcnd": opt_thcnd, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=64)


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
