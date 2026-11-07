"""Metal SFIRE kernels (woof.core.sfire_*): CUDA argument order, Metal storage."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .array import MetalArray

_REGISTRY = Path(__file__).with_name("kernel_signatures.json")
_UNITS = ("sfire_core", "sfire_phys", "sfire_moisture", "sfire_wind", "sfire_atm",
          "sfire_coupling", "sfire_spotting", "sfire_ideal", "sfire_ideal_atmos")


@lru_cache(maxsize=1)
def _signatures():
    table = {}
    for k in json.loads(_REGISTRY.read_text())["kernels"]:
        if Path(k["source"]).stem in _UNITS:
            table[k["cuda_symbol"]] = (k["name"], tuple(r[0] for r in k["resources"]),
                                       tuple(k["parameters"]), k["threads_per_group"])
    return table


def is_metal(value) -> bool:
    return isinstance(value, MetalArray)


class Kernel:
    """``get_function(name)`` result: called like a CuPy RawKernel."""

    def __init__(self, cuda_name):
        self.name = cuda_name

    def __call__(self, grid, block, args, **_ignored):
        name, resources, parameters, _ = _signatures()[self.name]
        arrays = [a for a in args if isinstance(a, MetalArray)]
        scalars = [a for a in args if not isinstance(a, MetalArray)]
        declared = [p for p in parameters if p[0] != "unused"]
        if len(arrays) != len(resources) or len(scalars) != len(declared):
            raise ValueError(f"SFIRE {self.name} takes {len(resources)} arrays and {len(declared)} scalars")
        context = arrays[0].buffer.context
        if any(a.buffer.context is not context or not a.flags.c_contiguous for a in arrays):
            raise TypeError("SFIRE resources must be contiguous MetalArrays on one context")
        values = {key: (int(v) if kind == "int32" else float(np.float32(v)))
                  for (key, kind), v in zip(declared, scalars)}
        if len(declared) != len(parameters):
            values["unused"] = 0
        threads = int(np.prod(grid)) * int(np.prod(block))
        if threads:
            context.submit_kernel(
                name, {k: a.view((a.size,)) for k, a in zip(resources, arrays)},
                values, exact_policy=False, grid_size=threads,
                threads_per_group=min(int(np.prod(block)), 128))


class Module:
    """Stand-in for a CuPy module of one SFIRE unit."""

    def get_function(self, name):
        return Kernel(name)


module = Module()
