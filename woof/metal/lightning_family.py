"""Typed Metal launchers for the WRF lightning diagnostics."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray

PATCH_FIELDS = ("refl", "w", "t", "z", "ht", "land")
COLUMN_FIELDS = ("refl", "w", "t", "z", "dz", "theta", "pi", "qc", "qr", "qi",
                 "qs", "qg", "ht", "land", "lat", "lon", "top", "inputnum",
                 "inputden")
OUTPUT_FIELDS = ("ic_flashrate", "cg_flashrate", "ic_flashcount",
                 "cg_flashcount", "lpi")


def _check(arrays, context):
    for name, value in arrays.items():
        dtype = np.dtype(np.int32 if name in ("top", "invalid") else np.float32)
        if (not isinstance(value, MetalArray) or value.dtype != dtype
                or not value.flags.c_contiguous or value.buffer.context is not context):
            raise TypeError(f"Metal lightning {name} must be contiguous {dtype} "
                            "on one context")


def submit_lightning_patch(fields, summary, *, option, tile):
    context = summary.buffer.context
    arrays = {name: fields[name] for name in PATCH_FIELDS}
    _check({**arrays, "summary": summary}, context)
    nz, ny, nx = map(int, fields["t"].shape)
    i0, i1, j0, j1 = (int(v) for v in tile)
    return context.submit_kernel(
        "lightning_patch", {**arrays, "summary": summary},
        {"nz": nz, "ny": ny, "nx": nx, "option": int(option),
         "i0": i0, "i1": i1, "j0": j0, "j1": j1},
        exact_policy=False, grid_size=1, threads_per_group=128)


def submit_lightning_columns(fields, summary, output, invalid, *, option, method,
                             month, dx, dy, adjust, num, den, factor, interval, tile):
    context = summary.buffer.context
    arrays = {name: fields[name] for name in COLUMN_FIELDS}
    arrays["summary"] = summary
    arrays.update({"ic": output["ic_flashrate"], "cg": output["cg_flashrate"],
                   "icc": output["ic_flashcount"], "cgc": output["cg_flashcount"],
                   "lpi": output["lpi"], "invalid": invalid})
    _check(arrays, context)
    nz, ny, nx = map(int, fields["t"].shape)
    i0, i1, j0, j1 = (int(v) for v in tile)
    flt = lambda v: float(np.float32(v))
    return context.submit_kernel(
        "lightning_columns", arrays,
        {"nz": nz, "ny": ny, "nx": nx, "option": int(option), "method": int(method),
         "month": int(month), "dx": flt(dx), "dy": flt(dy), "adjust": flt(adjust),
         "num": flt(num), "den": flt(den), "factor": flt(factor),
         "interval": flt(interval), "i0": i0, "i1": i1, "j0": j0, "j1": j1},
        exact_policy=False, grid_size=nx * ny, threads_per_group=128)
