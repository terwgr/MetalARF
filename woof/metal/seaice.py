"""Metal sea-ice thermodynamics (woof.ocean.seaice)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .array import MetalArray

_REGISTRY = Path(__file__).with_name("kernel_signatures.json")


@lru_cache(maxsize=1)
def _signatures():
    kernels = [k for k in json.loads(_REGISTRY.read_text())["kernels"]
               if k["source"] == "woof/metal/shaders/seaice_family.metal"]
    return {k["cuda_symbol"]: (k["name"], tuple(r[0] for r in k["resources"]),
                               tuple(p[0] for p in k["parameters"][:-1])) for k in kernels}


def launch(cuda_name, arrays, scalars=()):
    """Run ``cuda_name`` with the CUDA argument order on Metal storage."""
    name, resources, parameters = _signatures()[cuda_name]
    if len(arrays) != len(resources) or len(scalars) != len(parameters):
        raise ValueError(f"sea ice {cuda_name} takes {len(resources)} arrays and {len(parameters)} scalars")
    context = arrays[0].buffer.context
    if any(not isinstance(a, MetalArray) or a.buffer.context is not context
           or not a.flags.c_contiguous for a in arrays):
        raise TypeError("sea ice resources must be contiguous MetalArrays on one context")
    n = arrays[0].size
    if n:
        context.submit_kernel(
            name, {k: a.view((a.size,)) for k, a in zip(resources, arrays)},
            {**{k: float(np.float32(v)) for k, v in zip(parameters, scalars)}, "n": n},
            exact_policy=False, grid_size=n, threads_per_group=128)


def mask_xice(mask, xice, out):
    launch("PhysicsDriver._run_noah_seaice", (mask, xice, out))


def launch_pwp(cuda_name, arrays, scalars, levels, n):
    name, resources, parameters = _signatures()[cuda_name]
    if len(arrays) != len(resources) or len(scalars) + 1 != len(parameters):
        raise ValueError(f"invalid sea ice PWP arguments for {cuda_name}")
    context = arrays[0].buffer.context
    if any(not isinstance(a, MetalArray) or a.buffer.context is not context
           or not a.flags.c_contiguous for a in arrays):
        raise TypeError("sea ice resources must be contiguous MetalArrays on one context")
    if n:
        context.submit_kernel(name, {k: a.view((a.size,)) for k, a in zip(resources, arrays)},
            {**{k: float(np.float32(v)) for k, v in zip(parameters[:-1], scalars)},
             "levels": int(levels), "n": int(n)}, exact_policy=False,
            grid_size=n, threads_per_group=128)
