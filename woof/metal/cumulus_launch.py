"""Launch surface shared by the translated cumulus column kernels."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from woof.metal.array import MetalArray

_REGISTRY = Path(__file__).with_name("kernel_signatures.json")
_DTYPES = {"float": np.dtype(np.float32), "float32": np.dtype(np.float32),
           "int32": np.dtype(np.int32), "uint32": np.dtype(np.uint32)}
#: Columns per command buffer: a full mesh in one launch outlives the GPU
#: watchdog on a display GPU, so every submission is bounded.
COLUMNS_PER_SUBMIT = 2048


@lru_cache(maxsize=None)
def _entry(family: str, name: str) -> dict:
    with _REGISTRY.open(encoding="utf-8") as stream:
        return next(k for k in json.load(stream)["kernels"] if k["name"] == name)


def submit_column_kernel(family: str, name: str, resources: dict,
                         parameters: dict, *, grid_size: int):
    """Submit one registered column kernel over ``grid_size`` threads."""
    spec = _entry(family, name)
    resource_types = {r[0]: r[1] for r in spec["resources"]}
    parameter_types = {p[0]: p[1] for p in spec["parameters"]}
    if set(resources) != set(resource_types):
        raise ValueError(f"{name} resources do not match its signature")
    if set(parameters) != set(parameter_types):
        raise ValueError(f"{name} parameters do not match its signature")
    context = None
    arrays = {}
    for field, type_name in resource_types.items():
        array = resources[field]
        if (not isinstance(array, MetalArray) or array.ndim != 1
                or array.dtype != _DTYPES[type_name]
                or not array.flags.c_contiguous):
            raise TypeError(f"{name} {field} requires a flat contiguous "
                            f"{_DTYPES[type_name]} MetalArray")
        if context is not None and array.buffer.context is not context:
            raise ValueError(f"{name} resources must share one Metal context")
        context = array.buffer.context
        arrays[field] = array
    scalars = {field: np.asarray(parameters[field], dtype=_DTYPES[type_name]).item()
               for field, type_name in parameter_types.items()}
    if not 0 < int(grid_size) <= 0x7FFFFFFF:
        raise ValueError(f"{name} launch size must fit positive int32")
    return context.submit_kernel(
        name, arrays, scalars, exact_policy=False, grid_size=int(grid_size),
        threads_per_group=int(spec["threads_per_group"]))
