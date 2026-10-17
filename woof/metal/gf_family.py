"""Typed launch surface for the Grell–Freitas Metal column kernel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from woof.metal.array import MetalArray, MetalArrayModule


_SIGNATURE_PATH = (Path(__file__).resolve().parent
                   / "signature_fragments" / "gf_family.json")
with _SIGNATURE_PATH.open(encoding="utf-8") as _stream:
    _ENTRY = json.load(_stream)["kernels"][0]
#: Columns per command buffer.  A full mesh in one launch outlives the
#: GPU watchdog on a display GPU (kIOGPUCommandBufferCallbackErrorImpacting-
#: Interactivity); each batch is its own submission.
GF_COLUMNS_PER_SUBMIT = 2048

_OUTPUTS = {"rthcuten", "rqvcuten", "rqccuten", "rqicuten",
            "rucuten", "rvcuten", "rainc"}
_SURFACE_FIELDS = {"ht", "hfx", "qfx", "xland", "kpbl", "dx_column", "rainc"}


def _planes(name, nz, spp_enabled):
    if name in _SURFACE_FIELDS:
        return 1
    if name in {"w", "p_interface"}:
        return int(nz) + 1
    if name == "spp_conv":
        return 4 if spp_enabled else 1
    return int(nz)

_DTYPES = {"float": np.dtype(np.float32),
           "float32": np.dtype(np.float32),
           "int32": np.dtype(np.int32)}


def submit_gf_gfdrv_stage(resources: dict, parameters: dict, *, ncol: int):
    """Launch GFDRV over columns using direct contiguous Metal resources."""
    if isinstance(ncol, bool) or not isinstance(ncol, (int, np.integer)):
        raise TypeError("GF column count must be an integer")
    ncol = int(ncol)
    if ncol <= 0 or ncol > 0x7fffffff:
        raise ValueError("GF column count must fit positive int32")
    resource_specs = {item[0]: item[1] for item in _ENTRY["resources"]}
    parameter_specs = {item[0]: item[1] for item in _ENTRY["parameters"]
                       if item[0] != "col0"}
    if set(resources) != set(resource_specs):
        missing = sorted(set(resource_specs) - set(resources))
        extra = sorted(set(resources) - set(resource_specs))
        raise ValueError(
            f"GF resource names mismatch; missing={missing}, extra={extra}")
    if set(parameters) != set(parameter_specs):
        missing = sorted(set(parameter_specs) - set(parameters))
        extra = sorted(set(parameters) - set(parameter_specs))
        raise ValueError(
            f"GF parameter names mismatch; missing={missing}, extra={extra}")
    arrays = {}
    context = None
    for name, type_name in resource_specs.items():
        array = resources[name]
        if (not isinstance(array, MetalArray) or array.ndim != 1
                or array.dtype != _DTYPES[type_name]
                or not array.flags.c_contiguous):
            raise TypeError(
                f"GF {name} requires a flat contiguous {_DTYPES[type_name]} MetalArray")
        if context is not None and array.buffer.context is not context:
            raise ValueError("GF resources must share one Metal context")
        context = array.buffer.context
        arrays[name] = array
    if context is None:
        raise ValueError("GF requires Metal resources")
    scalar_values = {}
    for name, type_name in parameter_specs.items():
        value = np.asarray(parameters[name], dtype=_DTYPES[type_name])
        if value.ndim != 0:
            raise TypeError(f"GF {name} must be scalar")
        scalar_values[name] = value.item()
    if int(scalar_values["ncol"]) != ncol:
        raise ValueError("GF ncol parameter does not match launch width")
    if int(scalar_values["nz"]) <= 0:
        raise ValueError("GF nz must be positive")
    result = None
    for col0 in range(0, ncol, GF_COLUMNS_PER_SUBMIT):
        result = context.submit_kernel(
            "gf_gfdrv_stage", arrays, {**scalar_values, "col0": col0},
            exact_policy=False,
            grid_size=min(GF_COLUMNS_PER_SUBMIT, ncol - col0),
            threads_per_group=64)
    return result


def gf_metal_scratch_bytes(nz, columns, *, spp_enabled=False):
    """Peak bounded workspace and resident packing buffers for one call."""
    from woof.core.gf import GF_METAL_KMAX, gf_workspace_floats

    tile = min(int(columns), GF_COLUMNS_PER_SUBMIT)
    workspace = gf_workspace_floats(nz, tile, capacity=GF_METAL_KMAX) * 4
    packed = sum(_planes(name, nz, spp_enabled)
                 for name, _, _ in _ENTRY["resources"] if name != "workspace") * tile * 4
    return workspace + (packed if columns > tile else 0)


def submit_gf_tiled(resources, parameters, *, ncol, column_limit=GF_COLUMNS_PER_SUBMIT):
    """Pack independent columns into one bounded, reused per-call arena."""
    from woof.core.gf import GF_METAL_KMAX, gf_workspace_floats

    if type(column_limit) is not int or not 1 <= column_limit <= GF_COLUMNS_PER_SUBMIT:
        raise ValueError("GF tile must lie within the column submission limit")
    if isinstance(ncol, bool) or not isinstance(ncol, (int, np.integer)) or not 0 < ncol <= 0x7fffffff:
        raise ValueError("GF column count must fit positive int32")
    ncol = int(ncol)
    if int(parameters.get("ncol", -1)) != ncol:
        raise ValueError("GF ncol parameter does not match launch width")
    specifications = {name: dtype for name, dtype, _ in _ENTRY["resources"] if name != "workspace"}
    if set(resources) != set(specifications):
        raise ValueError("the tiled GF resources must match the ABI without workspace")
    context = resources["rainc"].buffer.context
    nz = int(parameters["nz"])
    if not 0 < nz <= GF_METAL_KMAX:
        raise ValueError("GF level count exceeds its compiled workspace capacity")
    rows = {name: _planes(name, nz, bool(parameters["spp_enabled"])) for name in resources}
    for name, array in resources.items():
        if (not isinstance(array, MetalArray) or array.ndim != 1
                or not array.flags.c_contiguous or array.buffer.context is not context
                or array.dtype != _DTYPES[specifications[name]]
                or array.size != rows[name] * ncol):
            raise ValueError(f"GF {name} must contain {rows[name]} contiguous column planes")
    module = MetalArrayModule(context)
    tile = min(int(ncol), column_limit)
    workspace = module.empty(
        (gf_workspace_floats(nz, tile, capacity=GF_METAL_KMAX),),
        dtype=np.float32)
    if tile == ncol:
        workspace.fill(0)
        return submit_gf_gfdrv_stage(
            {**resources, "workspace": workspace}, parameters, ncol=ncol)

    buffers = {}
    for name, array in resources.items():
        buffers[name] = module.empty((rows[name] * tile,), array.dtype)

    def transfer(pairs):
        submission = None
        for offset in range(0, len(pairs), 64):
            submission = context.submit_bookkeeping(tuple(pairs[offset:offset + 64]))
        return submission

    submission = None
    for start in range(0, ncol, tile):
        count = min(tile, ncol - start)
        packed = {name: array.view((rows[name] * count,))
                  for name, array in buffers.items()}
        gather, scatter = [], []
        for name, array in resources.items():
            itemsize = array.dtype.itemsize
            for level in range(rows[name]):
                full = array.view((count,), byte_offset=array.byte_offset
                                  + (level * ncol + start) * itemsize)
                part = packed[name].view((count,), byte_offset=packed[name].byte_offset
                                         + level * count * itemsize)
                (scatter if name in _OUTPUTS else gather).append(
                    (part, full) if name in _OUTPUTS else (full, part))
        transfer(gather)
        workspace.fill(0)
        for name in _OUTPUTS:
            packed[name].fill(0)
        submission = submit_gf_gfdrv_stage(
            {**packed, "workspace": workspace}, {**parameters, "ncol": count}, ncol=count)
        submission = transfer(scatter)
    return submission


__all__ = ["GF_COLUMNS_PER_SUBMIT", "gf_metal_scratch_bytes",
           "submit_gf_gfdrv_stage", "submit_gf_tiled"]
