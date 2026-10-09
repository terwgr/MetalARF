"""Typed launch surface for New Tiedtke's Metal stage family."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from woof.metal.array import MetalArray


_SIGNATURE_PATH = (Path(__file__).resolve().parent
                   / "signature_fragments" / "ntiedtke_family.json")
with _SIGNATURE_PATH.open(encoding="utf-8") as _stream:
    _SIGNATURES = {entry["name"]: entry
                   for entry in json.load(_stream)["kernels"]}
_DTYPES = {"float": np.dtype(np.float32), "int": np.dtype(np.int32),
           "float32": np.dtype(np.float32), "int32": np.dtype(np.int32)}


def submit_ntiedtke_kernel(stage: str, resources: dict, parameters: dict,
                            *, grid_size: int):
    """Submit a source-signature kernel over an explicit one-dimensional grid.

    Resource values are flat contiguous views retaining the level-major byte
    order of the source arrays. Scalar values are packed by the Metal runtime
    in the field order published by the signature fragment.
    """
    if stage not in _SIGNATURES:
        raise ValueError(f"New Tiedtke Metal stage is not implemented: {stage}")
    if isinstance(grid_size, bool) or not isinstance(
            grid_size, (int, np.integer)):
        raise TypeError("New Tiedtke grid size must be an integer")
    grid_size = int(grid_size)
    if grid_size <= 0 or grid_size > 0x7FFFFFFF:
        raise ValueError("New Tiedtke grid size must be in positive int32 range")
    spec = _SIGNATURES[stage]
    resources_spec = {item[0]: item[1] for item in spec["resources"]}
    params_spec = {item[0]: item[1] for item in spec["parameters"]}
    if set(resources) != set(resources_spec):
        missing = sorted(set(resources_spec) - set(resources))
        extra = sorted(set(resources) - set(resources_spec))
        raise ValueError(
            f"{stage} resource names mismatch; missing={missing}, extra={extra}")
    if set(parameters) != set(params_spec):
        missing = sorted(set(params_spec) - set(parameters))
        extra = sorted(set(parameters) - set(params_spec))
        raise ValueError(
            f"{stage} parameter names mismatch; missing={missing}, extra={extra}")

    arrays = {}
    context = None
    for name, type_name in resources_spec.items():
        array = resources[name]
        if not isinstance(array, MetalArray):
            raise TypeError(f"{stage} {name} must be a MetalArray")
        if (array.ndim != 1 or array.dtype != _DTYPES[type_name]
                or not array.flags.c_contiguous):
            raise TypeError(
                f"{stage} {name} requires a flat contiguous {_DTYPES[type_name]}")
        if context is not None and array.buffer.context is not context:
            raise ValueError(f"{stage} resources must share one Metal context")
        context = array.buffer.context
        arrays[name] = array
    if context is None:
        raise ValueError(f"{stage} has no Metal resources")

    scalar_values = {}
    for name, type_name in params_spec.items():
        value = np.asarray(parameters[name], dtype=_DTYPES[type_name])
        if value.ndim != 0:
            raise TypeError(f"{stage} {name} must be scalar")
        scalar_values[name] = value.item()
    if "ncol" in scalar_values and stage.startswith("ntiedtke_") \
            and not stage.startswith("ntiedtke_gather_") \
            and not stage.startswith("ntiedtke_scatter_") \
            and not stage.startswith("ntiedtke_force_fold") \
            and not stage.startswith("ntiedtke_llo3_"):
        if int(scalar_values["ncol"]) != grid_size:
            raise ValueError(f"{stage} ncol parameter does not match launch width")
    vertical_extent = scalar_values.get("nz", scalar_values.get("klev", 1))
    if int(vertical_extent) <= 0:
        raise ValueError(f"{stage} vertical extent must be positive")

    return context.submit_kernel(
        stage, arrays, scalar_values, exact_policy=False,
        grid_size=grid_size,
        threads_per_group=int(spec["threads_per_group"]))


def submit_ntiedtke_stage(stage: str, resources: dict, parameters: dict,
                           *, ncol: int):
    """Submit one column stage using its ``ncol`` scalar as grid width."""
    if "ncol" not in parameters or int(parameters["ncol"]) != int(ncol):
        raise ValueError(f"{stage} ncol parameter does not match launch width")
    return submit_ntiedtke_kernel(stage, resources, parameters,
                                   grid_size=int(ncol))


class MetalNtWorkspace:
    """The level-major New Tiedtke state, owned by one Metal context."""

    def __init__(self, module, *, ncol: int, nz: int):
        from woof.core.ntiedtke import (
            NT_CUTYPEN_SCR_I_SLOTS, NT_CUTYPEN_SCR_SLOTS,
            NT_LEVEL_F, NT_LEVEL_I, NT_SURFACE_F, NT_SURFACE_I,
            nt_aliased_names,
        )

        if ncol <= 0 or nz <= 0:
            raise ValueError("New Tiedtke workspace extents must be positive")
        self.module = module
        self.ncol = int(ncol)
        self.nz = int(nz)
        self.rows = self.nz + 2
        self._level = {}
        self._surface = {}
        self._bindings = {}
        self._aliased = nt_aliased_names()
        for names, dtype in ((NT_LEVEL_F, np.float32), (NT_LEVEL_I, np.int32)):
            for name in names:
                if name not in self._aliased:
                    self._level[name] = module.zeros(
                        (self.rows, self.ncol), dtype=dtype)
        for names, dtype in ((NT_SURFACE_F, np.float32),
                             (NT_SURFACE_I, np.int32)):
            for name in names:
                if name not in self._aliased:
                    self._surface[name] = module.zeros(self.ncol, dtype=dtype)
        self.scr = module.zeros(
            (NT_CUTYPEN_SCR_SLOTS, self.rows, self.ncol), np.float32)
        self.scr_i = module.zeros(
            (NT_CUTYPEN_SCR_I_SLOTS, self.rows, self.ncol), np.int32)

    def resolve(self, name: str) -> str:
        from woof.core.ntiedtke import nt_resolve
        return nt_resolve(name, self._level.keys() | self._surface.keys())

    def bind(self, name: str, base: int = 1) -> MetalArray:
        key = (name, base)
        if key in self._bindings:
            return self._bindings[key]
        target = self.resolve(name)
        if target in self._surface:
            result = self._surface[target]
        elif target in self._level:
            source = self._level[target]
            if base == 1:
                result = source
            elif base == 0:
                result = source.view(
                    (self.rows - 1, self.ncol),
                    byte_offset=source.byte_offset + self.ncol * source.dtype.itemsize)
            else:
                raise ValueError(f"level base must be 0 or 1, got {base!r}")
        else:
            raise KeyError(f"New Tiedtke workspace has no storage for {name!r}")
        self._bindings[key] = result
        return result

    def bytes_allocated(self) -> int:
        arrays = (*self._level.values(), *self._surface.values(),
                  self.scr, self.scr_i)
        return sum(array.nbytes for array in arrays)


class _MetalNtPipeline:
    """One reusable New Tiedtke workspace at the chosen column tile width."""

    def __init__(self, module, *, ncol: int, nz: int, width: int, dt: float,
                 tiedtke_closure: bool):
        self.module = module
        self.ncol = int(ncol)
        self.nz = int(nz)
        self.width = int(width)
        self.w = MetalNtWorkspace(module, ncol=width, nz=nz)
        self.llo3_mask = module.zeros((1,), np.int32)
        self.llo3_partials = module.zeros((2 * width,), np.int32)
        self.llo3_flags = None
        self.scalars = {
            "ncol": np.int32(width), "nz": np.int32(nz),
            "klev": np.int32(nz), "stepcu": np.int32(1),
            "itimestep": np.int32(2), "ktopm2": np.int32(2),
            "llo3": np.int32(-1), "dt": np.float32(dt),
            "tiedtke_closure": np.int32(bool(tiedtke_closure)),
            "delt": np.float32(dt), "ztmst": np.float32(dt),
            "cp": np.float32(1004.5), "rd": np.float32(287.0),
            "rv": np.float32(461.6), "xlv": np.float32(2.5e6),
            "xlf": np.float32(3.5e5), "grav": np.float32(9.81),
        }

    def retime(self, dt: float) -> None:
        value = np.float32(dt)
        self.scalars["dt"] = value
        self.scalars["delt"] = value
        self.scalars["ztmst"] = value

    def _run_stage(self, stage: str) -> None:
        from woof.core.ntiedtke import (
            NT_LEVEL_BASE, NT_STAGE_ALIASES, NT_STAGE_SIGNATURE,
        )
        base = NT_LEVEL_BASE.get(stage, 1)
        entry = _SIGNATURES[stage]
        resources = {}
        for name, _dtype, _access in entry["resources"]:
            if name == "llo3_mask":
                resources[name] = self.llo3_mask.view((1,))
            elif name == "scr":
                resources[name] = self.w.scr.view((self.w.scr.size,))
            elif name == "scr_i":
                resources[name] = self.w.scr_i.view((self.w.scr_i.size,))
            else:
                source = NT_STAGE_ALIASES.get((stage, name), name)
                resources[name] = self.w.bind(source, base).view(
                    (self.w.bind(source, base).size,))
        parameters = {name: self.scalars[name]
                      for name, _dtype in entry["parameters"]}
        submit_ntiedtke_stage(stage, resources, parameters,
                              ncol=self.width)

    def zero_run_head(self) -> None:
        for name in ("pcte", "ptenu", "ptenv"):
            self.w.bind(name, 1).fill(np.float32(0.0))

    def snapshot_forcing(self) -> None:
        self.module.context.submit_bookkeeping((
            (self.w.bind("pqte", 1), self.w.bind("zqq", 1)),
            (self.w.bind("ptte", 1), self.w.bind("ztt", 1)),
        ))

    def snapshot_momentum(self) -> None:
        self.module.context.submit_bookkeeping((
            (self.w.bind("pvom", 1), self.w.bind("ztenu", 1)),
            (self.w.bind("pvol", 1), self.w.bind("ztenv", 1)),
        ))

    def reduce_llo3(self, chunk_index: int) -> None:
        submit_ntiedtke_kernel(
            "ntiedtke_llo3_reduce_columns",
            {"klab": self.w.bind("klab", 1).view((self.w.bind("klab", 1).size,)),
             "partials": self.llo3_partials.view((self.llo3_partials.size,))},
            {"ncol": self.width, "nz": self.nz}, grid_size=self.width)
        submit_ntiedtke_kernel(
            "ntiedtke_llo3_reduce_finish",
            {"partials": self.llo3_partials.view((self.llo3_partials.size,)),
             "llo3_mask": self.llo3_mask.view((1,)),
             "flags": self.llo3_flags.view((self.llo3_flags.size,))},
            {"ncol": self.width, "chunk_index": chunk_index}, grid_size=1)

    def run_chunk(self, chunk_index: int) -> None:
        from woof.core.ntiedtke import NT_CALL_ORDER
        self.zero_run_head()
        self._run_stage("ntiedtke_prep")
        self._run_stage("ntiedtke_convert")
        self.snapshot_forcing()
        for stage in NT_CALL_ORDER[2:]:
            if stage == "ntiedtke_cuascn":
                self.reduce_llo3(chunk_index)
            elif stage == "ntiedtke_cududvn":
                self.snapshot_momentum()
            self._run_stage(stage)


def _metal_flat(array, *, shape, name, context):
    if (not isinstance(array, MetalArray) or array.shape != tuple(shape)
            or array.dtype != np.dtype(np.float32)
            or not array.flags.c_contiguous
            or array.buffer.context is not context):
        raise TypeError(
            f"New Tiedtke {name} must be a same-context contiguous FP32 "
            f"MetalArray with shape {tuple(shape)}")
    return array.view((array.size,))


def run_new_tiedtke(owner, *, atmosphere, fields, state, cfg):
    """Run New Tiedtke using only Metal storage and native stage launches."""
    from woof.core.ntiedtke import _nt_model_dt, nt_tile_columns
    from woof.core.preflight import MEASURED_LOCAL_MEMORY_PROFILE
    from woof.core.physics import _NativeNtCumulusResult

    module = state._array_module
    context = module.context
    nz, ny, nx = state.p.shape
    ncol = ny * nx
    volume_shape = (nz, ny, nx)
    interface_shape = (nz + 1, ny, nx)
    surface_shape = (ny, nx)

    def input_array(value, shape, name):
        return _metal_flat(value, shape=shape, name=name, context=context)

    exner = input_array(atmosphere["exner"], volume_shape, "exner")
    force_names = ("gf_rthdynten", "rthratenlw", "rthratensw",
                   "gf_rthblten", "gf_rqvdynten", "gf_rqvblten")
    force_inputs = {}
    for name in force_names:
        lane = None if owner._driver is None else getattr(owner._driver, name, None)
        if lane is None:
            lane = module.zeros(volume_shape, dtype=np.float32)
        force_inputs[name] = input_array(lane, volume_shape, name)
    thften = module.empty(volume_shape, dtype=np.float32)
    qvften = module.empty(volume_shape, dtype=np.float32)
    submit_ntiedtke_kernel(
        "ntiedtke_force_fold",
        {"exner": exner, **force_inputs,
         "thften": thften.view((thften.size,)),
         "qvften": qvften.view((qvften.size,))},
        {"ncol": ncol, "nz": nz}, grid_size=ncol * nz)

    driver_lev = {
        "t3d": atmosphere["temperature"], "qv3d": atmosphere["qv"],
        "qc3d": atmosphere["qc"], "qi3d": atmosphere["qi"],
        "u3d": atmosphere["u"], "v3d": atmosphere["v"],
        "pcps": atmosphere["pressure"], "dz8w": atmosphere["dz"],
        "rho3d": atmosphere["rho"], "exner": atmosphere["exner"],
        "qvften": qvften, "thften": thften,
    }
    driver_iface = {"p8w": atmosphere["p_interface"], "w": state.w}
    surface = {"xland": fields["xland"], "hfx": fields["hfx"],
               "qfx": fields["qfx"]}
    flat_lev = {name: input_array(value, volume_shape, name)
                for name, value in driver_lev.items()}
    flat_iface = {name: input_array(value, interface_shape, name)
                  for name, value in driver_iface.items()}
    flat_sfc = {name: input_array(value, surface_shape, name)
                for name, value in surface.items()}

    width = nt_tile_columns(
        ncol, int(MEASURED_LOCAL_MEMORY_PROFILE.multiprocessor_count))
    key = (id(context), width, nz)
    if owner._metal_pipeline is None or owner._metal_pipeline[0] != key:
        owner._metal_pipeline = None
        pipeline = _MetalNtPipeline(
            module, ncol=ncol, nz=nz, width=width,
            dt=float(_nt_model_dt(cfg)),
            tiedtke_closure=bool(
                getattr(cfg, "ntiedtke_tiedtke_closure", False)))
        owner._metal_pipeline = (key, pipeline)
    else:
        pipeline = owner._metal_pipeline[1]
        pipeline.retime(float(_nt_model_dt(cfg)))
    chunks = (ncol + width - 1) // width
    pipeline.llo3_flags = module.zeros((chunks,), np.int32)
    pipeline.w.bind("dx", 1).fill(np.float32(cfg.dx))

    out_lev = {name: module.empty((nz, ncol), dtype=np.float32) for name in
               ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                "rucuten", "rvcuten")}
    out_sfc = {name: module.empty((ncol,), dtype=np.float32)
               for name in ("raincv", "pratec")}
    for chunk_index, lo in enumerate(range(0, ncol, width)):
        count = min(width, ncol - lo)
        for name, source in flat_lev.items():
            _pack_chunk(source, pipeline.w.bind(name, 0),
                        stage="ntiedtke_gather_level", full_ncol=ncol,
                        start=lo, count=count, width=width, levels=nz)
        for name, source in flat_iface.items():
            _pack_chunk(source, pipeline.w.bind(name, 0),
                        stage="ntiedtke_gather_interface", full_ncol=ncol,
                        start=lo, count=count, width=width, levels=nz + 1)
        for name, source in flat_sfc.items():
            _pack_chunk(source, pipeline.w.bind(name, 1),
                        stage="ntiedtke_gather_surface", full_ncol=ncol,
                        start=lo, count=count, width=width, levels=0)
        pipeline.llo3_mask.fill(np.int32(0))
        pipeline.run_chunk(chunk_index)
        for name, target in out_lev.items():
            _scatter_chunk(pipeline.w.bind(name, 0), target,
                           stage="ntiedtke_scatter_level", full_ncol=ncol,
                           start=lo, count=count, width=width, levels=nz)
        for name, target in out_sfc.items():
            _scatter_chunk(pipeline.w.bind(name, 1), target,
                           stage="ntiedtke_scatter_surface", full_ncol=ncol,
                           start=lo, count=count, width=width)

    masks = pipeline.llo3_flags.to_numpy()
    for mask in masks:
        first, ever = bool(int(mask) & 1), bool(int(mask) & 2)
        if first != ever:
            raise ValueError(
                "llo3's hoist is unsound for this chunk: klab is zero at "
                "klev on every column but non-zero at some level below")
        pipeline.scalars["llo3"] = np.int32(first)

    def grid(array):
        return array.view(volume_shape)

    return _NativeNtCumulusResult(
        owner=owner,
        rthcuten=grid(out_lev["rthcuten"]),
        rqvcuten=grid(out_lev["rqvcuten"]),
        rqccuten=grid(out_lev["rqccuten"]),
        rqicuten=(grid(out_lev["rqicuten"])
                  if getattr(state, "qi", None) is not None else None),
        rucuten=grid(out_lev["rucuten"]),
        rvcuten=grid(out_lev["rvcuten"]),
        rainc=out_sfc["raincv"].view(surface_shape),
    )


def merge_ntiedtke_rain(*, result, driver, state):
    """Accumulate New Tiedtke's per-call rain into Metal driver state."""
    nz, ny, nx = state.p.shape
    shape = (ny, nx)
    context = state._array_module.context
    rainc = _metal_flat(driver.rainc, shape=shape, name="driver RAINC",
                        context=context)
    pending = _metal_flat(
        driver._pending_rainbl, shape=shape,
        name="pending RAINBL", context=context)
    increment = _metal_flat(
        result.rainc, shape=shape, name="RAINC increment", context=context)
    surface = driver.fields.get("surface_raincv")
    if surface is None:
        surface_array = rainc
        has_surface = 0
    else:
        surface_array = _metal_flat(
            surface, shape=shape, name="surface RAINCV", context=context)
        has_surface = 1
    submit_ntiedtke_kernel(
        "ntiedtke_driver_rain_merge",
        {"increment": increment, "rainc": rainc,
         "pending_rainbl": pending, "surface_raincv": surface_array},
        {"ncol": ny * nx, "has_surface_raincv": has_surface},
        grid_size=ny * nx)


def _flat_metal(array: MetalArray, *, dtype, name: str) -> MetalArray:
    if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
        raise TypeError(f"New Tiedtke {name} must be a MetalArray {dtype}")
    return array.view((array.size,))


def _pack_chunk(array, target, *, stage, full_ncol, start, count, width, levels):
    source = _flat_metal(array, dtype=np.float32, name=stage)
    resource = {"source": source, "target": target.view((target.size,))}
    if stage == "ntiedtke_gather_level":
        params = {"full_ncol": full_ncol, "start": start, "count": count,
                  "width": width, "nz": levels}
        grid = levels * width
    elif stage == "ntiedtke_gather_interface":
        params = {"full_ncol": full_ncol, "start": start, "count": count,
                  "width": width, "nzi": levels}
        grid = levels * width
    else:
        params = {"full_ncol": full_ncol, "start": start, "count": count,
                  "width": width}
        grid = width
    submit_ntiedtke_kernel(stage, resource, params, grid_size=grid)


def _scatter_chunk(source, target, *, stage, full_ncol, start, count, width,
                   levels=0):
    resource = {"source": source.view((source.size,)),
                "target": target.view((target.size,))}
    params = {"full_ncol": full_ncol, "start": start, "count": count,
              "width": width}
    if stage == "ntiedtke_scatter_level":
        params["nz"] = levels
        grid = levels * count
    else:
        grid = count
    submit_ntiedtke_kernel(stage, resource, params, grid_size=grid)


__all__ = ["submit_ntiedtke_kernel", "submit_ntiedtke_stage",
           "run_new_tiedtke", "merge_ntiedtke_rain"]
