"""Typed Metal submitters for WRF's complete CLM lake column.

The WRF transcription keeps its 131 restart and 71 static words in the
public per-cell arrays.  Each grid cell owns one thread; non-lake columns
return before touching state or surface outputs.  This avoids host gathers
and preserves all inactive-cell bits.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from woof.core.lake_schema import (
    LAKE_DEFAULT_DEPTH, LAKE_DEFAULT_USE_DEPTH, LAKE_OUTPUT_LAYOUT,
    LAKE_STATE_WORDS, LAKE_STATIC_WORDS,
)
from .array import MetalArray, MetalArrayModule
from .runtime import MetalContext


def _require_array(name: str, value: Any, *, context: MetalContext,
                   dtype, shape) -> MetalArray:
    if (not isinstance(value, MetalArray) or value.dtype != np.dtype(dtype)
            or value.shape != tuple(shape) or not value.flags.c_contiguous):
        raise TypeError(
            f"Metal lake {name} must be contiguous {np.dtype(dtype)} "
            f"with shape {tuple(shape)}")
    if value.buffer.context is not context:
        raise ValueError(f"Metal lake {name} belongs to a different Metal context")
    return value


def _surface_fields(fields, names, *, context, shape, dtypes=None):
    dtypes = dtypes or {}
    return {name: _require_array(
        name, fields[name], context=context, dtype=dtypes.get(name, np.float32),
        shape=shape) for name in names}


@dataclass
class MetalLakeModel:
    context: MetalContext
    shape: tuple[int, int]
    errors: MetalArray
    lakemask: MetalArray
    zero_pending: MetalArray
    xice_threshold: float
    iswater: int
    depth_missing: bool = False

    @property
    def column_count(self) -> int:
        # This is a diagnostic property, never used in the step path.  Read
        # only the mask when a caller explicitly asks for the census.
        return int(np.count_nonzero(self.lakemask.to_numpy() == np.float32(1)))

    def _check_errors(self) -> None:
        status = self.errors.to_numpy()
        bad = status[status != 0]
        if not bad.size:
            return
        code = int(bad[0])
        if code == 61001:
            raise ValueError(
                "sf_lake_physics=1 with use_lakedepth=1 requires LAKE_DEPTH; "
                "prepare lake-depth geography or set use_lakedepth=0 for "
                "the WRF default depth")
        if code == 61002:
            raise ValueError(
                "Lake columns require finite latitude for the "
                "latitude-dependent eddy diffusivity")
        if code == 61003:
            raise ValueError("Lake soil categories must be in WRF's 1..19 table")
        if code == 61004:
            raise ValueError("Lake initial soil, depth, temperature, snow and ice must be finite")
        raise RuntimeError(
            f"WRF CLM lake rejected a column at module_sf_lake.F:{code}; "
            "its state and fluxes were not committed")

    def check_errors(self) -> None:
        self._check_errors()

    def step(self, fields, atmosphere, *, dt: float, precipitation,
             pending_rainbl=None) -> None:
        if dt <= 0:
            raise ValueError("Lake time step must be positive to convert accumulated precipitation to a rate")
        ny, nx = self.shape
        n = ny * nx
        ctx = self.context
        p_interface = atmosphere["p_interface"][:2]
        resources = {
            "lakemask": fields["lakemask"],
            "temperature": atmosphere["temperature"][0],
            "p_interface": p_interface,
            "dz": atmosphere["dz"][0],
            "qv": atmosphere["qv"][0],
            "u": atmosphere["u"][0],
            "v": atmosphere["v"][0],
            "glw": fields["glw"], "emiss": fields["emiss"],
            "precipitation": precipitation,
            "pending_rainbl": (self.zero_pending if pending_rainbl is None
                                else pending_rainbl),
            "swdown": fields["swdown"], "albedo": fields["albedo"],
            "latitude": fields["lake_latitude"],
            "lake_columns": fields["lake_columns"],
            "lake_static": fields["lake_static"], "errors": self.errors,
        }
        resources.update({name: fields[name] for name, _ in LAKE_OUTPUT_LAYOUT})
        floats = set(resources) - {
            "errors", "lake_columns", "lake_static", "p_interface"}
        for name in floats:
            _require_array(name, resources[name], context=ctx, dtype=np.float32,
                           shape=(ny, nx))
        _require_array("p_interface", p_interface, context=ctx,
                       dtype=np.float32, shape=(2, ny, nx))
        _require_array("lake_columns", resources["lake_columns"], context=ctx,
                       dtype=np.float32, shape=(LAKE_STATE_WORDS, ny, nx))
        _require_array("lake_static", resources["lake_static"], context=ctx,
                       dtype=np.float32, shape=(LAKE_STATIC_WORDS, ny, nx))
        _require_array("errors", self.errors, context=ctx, dtype=np.int32,
                       shape=(n,))
        _require_array("pending_rainbl", resources["pending_rainbl"],
                       context=ctx, dtype=np.float32, shape=(ny, nx))
        ctx.submit_kernel(
            "lake_step_columns", resources,
            {"n": n, "ny": ny, "nx": nx, "dt": float(dt),
             "xice_threshold": self.xice_threshold,
             "iswater": self.iswater, "lake_min_elev": 5.0,
             "state_words": LAKE_STATE_WORDS,
             "static_words": LAKE_STATIC_WORDS, "two": 2},
            exact_policy=False, grid_size=n, threads_per_group=32)
        self._check_errors()


def initialize_lake_metal(
        fields, *, latitude, lake_depth=None, lake_depth_flag=None,
        use_lakedepth=LAKE_DEFAULT_USE_DEPTH,
        lakedepth_default=LAKE_DEFAULT_DEPTH, iswater=17,
        xice_threshold: float = 0.5) -> MetalLakeModel:
    """Initialize complete WRF lake state on the active Metal context."""
    if isinstance(use_lakedepth, (bool, np.bool_)) or use_lakedepth not in (0, 1):
        raise ValueError("use_lakedepth must be 0 or 1")
    use_lakedepth = int(use_lakedepth)
    if isinstance(iswater, bool) or int(iswater) <= 0:
        raise ValueError("Lake initialization needs the land-use table's positive ISWATER category")
    threshold = float(xice_threshold)
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"lake xice_threshold must be in (0, 1], got {xice_threshold!r}")
    if not math.isfinite(float(lakedepth_default)):
        raise ValueError("lakedepth_default must be finite")

    tsk = fields["tsk"]
    if not isinstance(tsk, MetalArray) or tsk.ndim != 2:
        raise TypeError("Metal lake tsk must be a two-dimensional MetalArray")
    shape = tuple(tsk.shape)
    context = tsk.buffer.context
    module = MetalArrayModule(context)
    n = math.prod(shape)
    if "lakemask" not in fields:
        fields["lakemask"] = module.zeros(shape, np.float32)
    _require_array("lakemask", fields["lakemask"], context=context,
                   dtype=np.float32, shape=shape)

    if latitude is None:
        lat = module.zeros(shape, np.float32)
        latitude_present = 0
    elif isinstance(latitude, MetalArray):
        lat = module.asarray(latitude, dtype=np.float32)
        latitude_present = 1
    else:
        host_lat = np.asarray(latitude, dtype=np.float32)
        if host_lat.ndim == 0:
            host_lat = np.full(shape, host_lat, np.float32)
        if host_lat.shape != shape:
            raise ValueError("Lake latitude must cover the surface grid")
        lat = module.asarray(host_lat, dtype=np.float32)
        latitude_present = 1
    _require_array("latitude", lat, context=context, dtype=np.float32,
                   shape=shape)

    present = [name in fields for name in ("lake_columns", "lake_static", "lake_latitude")]
    if any(present) and not all(present):
        raise ValueError("Incomplete lake restart would discard water, ice or sediment heat storage")
    had_restart = all(present)
    if had_restart:
        for name, dims in (("lake_columns", (LAKE_STATE_WORDS, *shape)),
                           ("lake_static", (LAKE_STATIC_WORDS, *shape)),
                           ("lake_latitude", shape)):
            _require_array(name, fields[name], context=context,
                           dtype=np.float32, shape=dims)
    else:
        fields["lake_columns"] = module.zeros((LAKE_STATE_WORDS, *shape), np.float32)
        fields["lake_static"] = module.zeros((LAKE_STATIC_WORDS, *shape), np.float32)
        fields["lake_latitude"] = lat

    has_depth = lake_depth is not None or "lake_depth" in fields
    if lake_depth is None:
        lake_depth = fields.get("lake_depth")
    depth_flag = int(has_depth) if lake_depth_flag is None else int(lake_depth_flag)
    depth_missing = use_lakedepth == 1 and (not has_depth or depth_flag == 0)
    if lake_depth is None:
        depth = module.zeros(shape, np.float32)
    else:
        depth = module.asarray(lake_depth, dtype=np.float32)
    _require_array("lake_depth", depth, context=context, dtype=np.float32,
                   shape=shape)

    errors = module.zeros((n,), np.int32)
    owner = MetalLakeModel(context, shape, errors, fields["lakemask"],
                           module.zeros(shape, np.float32),
                           threshold, int(iswater),
                           depth_missing)
    # Keep the CUDA initialization contract even for restarts: the surface
    # configuration still requires the named bathymetry when it requests it.
    active_count = owner.column_count if depth_missing or latitude is None else 0
    if active_count and depth_missing:
        raise ValueError(
            "sf_lake_physics=1 with use_lakedepth=1 requires LAKE_DEPTH; "
            "prepare lake-depth geography or set use_lakedepth=0 for "
            "the WRF default depth")
    if active_count and latitude is None:
        raise ValueError(
            "Lake columns require latitude for the latitude-dependent "
            "eddy diffusivity")
    if not had_restart:
        if "isltyp" not in fields:
            raise ValueError("Lake initialization requires isltyp for sediment properties")
        _require_array("isltyp", fields["isltyp"], context=context,
                       dtype=np.int32, shape=shape)
        init_resources = {
            "lakemask": fields["lakemask"], "isltyp": fields["isltyp"],
            "lake_depth": depth, "tsk": fields["tsk"],
            "snow": fields["snow"] if "snow" in fields else module.zeros(shape, np.float32),
            "xice": fields["xice"] if "xice" in fields else module.zeros(shape, np.float32),
            "lake_columns": fields["lake_columns"],
            "lake_static": fields["lake_static"],
            "lake_latitude": fields["lake_latitude"],
            "ivgtyp": fields["ivgtyp"] if "ivgtyp" in fields else module.zeros(shape, np.int32),
            "xland": fields["xland"] if "xland" in fields else module.zeros(shape, np.float32),
            "errors": errors,
        }
        for name in ("snow", "xice", "xland"):
            _require_array(name, init_resources[name], context=context,
                           dtype=np.float32, shape=shape)
        _require_array("ivgtyp", init_resources["ivgtyp"], context=context,
                       dtype=np.int32, shape=shape)
        ny, nx = shape
        params = {"n": n, "ny": ny, "nx": nx,
                  "use_depth": use_lakedepth,
                  "depth_flag": depth_flag,
                  "default_depth": float(lakedepth_default),
                  "xice_threshold": threshold, "iswater": int(iswater),
                  "lake_min_elev": 5.0,
                  "latitude_present": latitude_present,
                  "state_words": LAKE_STATE_WORDS,
                  "static_words": LAKE_STATIC_WORDS, "two": 2}
        context.submit_kernel(
            "lake_init_columns", init_resources, params,
            exact_policy=False, grid_size=n, threads_per_group=32)
        owner._check_errors()
    return owner
