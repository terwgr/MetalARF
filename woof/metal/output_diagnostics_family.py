"""Typed Metal launcher for the interval precipitation buckets."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray


def submit_precipitation_buckets(rainc, rain, snow, graup, t2, pc, pn, sn, gn,
                                 *, reset: bool):
    arrays = {"rainc": rainc, "rain": rain, "snow": snow, "graup": graup,
              "t2": t2, "pc": pc, "pn": pn, "sn": sn, "gn": gn}
    context = pc.buffer.context
    for name, value in arrays.items():
        if (not isinstance(value, MetalArray) or value.dtype != np.dtype(np.float32)
                or value.ndim != 2 or value.shape != pc.shape
                or not value.flags.c_contiguous or value.buffer.context is not context):
            raise TypeError(f"Metal precipitation bucket {name} must be a "
                            "contiguous float32 plane on one context")
    ny, nx = map(int, pc.shape)
    count = ny * nx
    if not count:
        return None
    from woof.wrf_exact import ENABLED
    return context.submit_kernel(
        "precipitation_buckets", arrays,
        {"count": count, "ny": ny, "nx": nx, "reset": int(bool(reset))},
        exact_policy=ENABLED, grid_size=count, threads_per_group=128)
