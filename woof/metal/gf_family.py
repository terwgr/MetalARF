"""Typed launch surface for the Grell–Freitas Metal column kernel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from woof.metal.array import MetalArray


_SIGNATURE_PATH = (Path(__file__).resolve().parent
                   / "signature_fragments" / "gf_family.json")
with _SIGNATURE_PATH.open(encoding="utf-8") as _stream:
    _ENTRY = json.load(_stream)["kernels"][0]
#: Columns per command buffer.  A full mesh in one launch outlives the
#: GPU watchdog on a display GPU (kIOGPUCommandBufferCallbackErrorImpacting-
#: Interactivity); each batch is its own submission.
GF_COLUMNS_PER_SUBMIT = 2048

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


__all__ = ["GF_COLUMNS_PER_SUBMIT", "submit_gf_gfdrv_stage"]
