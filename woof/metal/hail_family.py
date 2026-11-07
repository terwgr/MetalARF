"""Metal launchers for the hail diagnostics (woof/core/hail.py).

Each launcher takes the CUDA kernel's positional arguments, so the caller's
argument list is shared between the two backends.
"""
from __future__ import annotations

import numpy as np

from .array import MetalArray
from .generated import kernel_specs

_THREADS = {"hail_prepare": 128, "hail_duration": 128, "hail_psd": 128,
            "hailcast_columns": 32, "hailcast_columns_wrf39": 32}
_CUDA_ORDER = {
    "hail_prepare": ("theta", "p", "alt", "qv", "phb", "php", "w", "t", "z", "rho",
                     "mask", "nx", "ny", "nz", "base_profile"),
    "hail_duration": ("mask", "oldmask", "olddur", "dur", "nx", "ny", "dt",
                      "periodic_x", "periodic_y"),
    "hail_psd": ("rho", "qg", "ng", "vg", "qh", "nh", "vh", "lowest", "column",
                 "runlowest", "runcolumn", "nx", "ny", "nz", "number_g", "volume_g",
                 "number_h", "volume_h", "thompson", "density_g", "density_h",
                 "alpha_g", "alpha_h", "intercept", "intercept_h"),
    "hailcast_columns": ("t", "z", "p", "rho", "qv", "qi", "qc", "qr", "qs", "qg",
                         "w", "ht", "dur", "e1", "e2", "e3", "e4", "e5", "maximum",
                         "mean", "sd", "runmax", "nx", "ny", "nz"),
}
_CUDA_ORDER["hailcast_columns_wrf39"] = _CUDA_ORDER["hailcast_columns"]


def submit_hail(name: str, args) -> None:
    """Run one hail kernel with the CUDA argument order."""
    values = dict(zip(_CUDA_ORDER[name], args, strict=True))
    prefix = name.upper()
    resources = {field: values[field]
                 for field, _, _ in getattr(kernel_specs, f"{prefix}_RESOURCES")}
    context = next(iter(resources.values())).buffer.context
    for field, array in resources.items():
        if (not isinstance(array, MetalArray) or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous or array.buffer.context is not context):
            raise TypeError(f"Metal hail {field} must be contiguous float32 "
                            "storage in the state's context")
    if name == "hail_psd" and {float(values["alpha_g"]), float(values["alpha_h"])} - {0.0, 1.0}:
        raise NotImplementedError("Metal hail PSD supports alpha 0 and 1 only")
    parameters = {field: float(values[field]) if "float" in kind else int(values[field])
                  for field, kind in getattr(kernel_specs, f"{prefix}_PARAM_FIELDS")}
    context.submit_kernel(
        name, {field: array.view((array.size,)) for field, array in resources.items()},
        parameters, exact_policy=False,
        grid_size=int(values["nx"]) * int(values["ny"]), threads_per_group=_THREADS[name])
