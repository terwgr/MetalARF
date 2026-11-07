"""Strict Metal submission for the climate output accumulator."""
from __future__ import annotations

import numpy as np
from woof.core.climate_diagnostics import PERSISTENT
from woof.metal.array import MetalArray
from woof.wrf_exact import ENABLED


def clock_refusal(dt, diag_int, wind_int):
    """Return a reason if source FP32 periods cannot be converted to int32."""
    if any(type(value) is not int or not 0 < value < 2**31
           for value in (diag_int, wind_int)):
        return "climate diagnostic intervals must be positive int32 minutes"
    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            step = np.float32(dt)
            if not np.isfinite(step) or step <= 0:
                return "climate diagnostic timestep must be positive finite FP32"
            periods = [np.float32(np.float32(np.float32(value) * np.float32(60)) / step)
                       for value in (diag_int, wind_int)]
    except (TypeError, ValueError, OverflowError):
        return "climate diagnostic timestep must be positive finite FP32"
    if any(not np.isfinite(value) or value < 1 or value >= 2**31 for value in periods):
        return "climate diagnostic FP32 periods must be at least 1 and below 2^31 steps"
    return None


def submit_climate_diagnostics(inputs, outputs, *, dt, xtime, secs,
                               diag_int, wind_int, restart):
    if not ENABLED:
        raise RuntimeError("strict climate diagnostics requires WRF exact mode")
    reason = clock_refusal(dt, diag_int, wind_int)
    if reason:
        raise ValueError(reason)
    names = ("t2", "q2", "tsk", "swdown", "swddni", "u10", "v10", "raincv", "rainncv", *PERSISTENT)
    values = (*inputs, *outputs)
    if len(values) != len(names) or not isinstance(values[0], MetalArray):
        raise ValueError("climate diagnostics requires all Metal planes")
    shape = values[0].shape
    context = values[0].buffer.context
    if len(shape) != 2:
        raise ValueError("climate diagnostics requires 2-D planes")
    for name, value in zip(names, values):
        if (not isinstance(value, MetalArray) or value.shape != shape
                or value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise ValueError(f"climate diagnostics {name} must be a contiguous FP32 plane")
    ny, nx = shape
    return context.submit_kernel(
        "climate_diagnostics_exact", dict(zip(names, values)),
        {"count": ny*nx, "ny": ny, "nx": nx, "dt": np.float32(dt),
         "xtime": np.float32(xtime), "secs": np.float32(secs),
         "diag_int": int(diag_int), "wind_int": int(wind_int), "restart": int(restart)},
        exact_policy=True, grid_size=ny*nx, threads_per_group=128)
