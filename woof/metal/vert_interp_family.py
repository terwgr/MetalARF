"""Typed Metal submission for the standalone FP32 log-pressure helper."""

from __future__ import annotations

import numpy as np

from woof.core import constants as c
from .array import MetalArray


def submit_vertical_interpolate_logp(context, field, source_pressure,
                                     target_pressure, output, *, below):
    """Submit the retained log-p interpolation kernel on prepared arrays.

    Inputs must already be validated, C-contiguous FP32 Metal arrays with
    ``field`` and ``source_pressure`` shaped ``(nsource, ny, nx)`` and target
    and output shaped ``(ntarget, ny, nx)``. Pressure broadcasting,
    monotonicity/finiteness checks, and top-bracket validation remain caller
    preparation responsibilities; this low-level submitter performs no host
    readback or implicit array conversion.
    """
    if below not in ("constant", "temperature"):
        raise ValueError("below must be 'constant' or 'temperature'")
    arrays = (field, source_pressure, target_pressure, output)
    if any(not isinstance(array, MetalArray) for array in arrays):
        raise TypeError("Metal vertical interpolation requires MetalArray resources")
    if any(array.dtype != np.dtype(np.float32) for array in arrays):
        raise TypeError("Metal vertical interpolation requires FP32 resources")
    if any(array.ndim != 3 for array in arrays):
        raise ValueError("vertical interpolation arrays must be 3-D")
    nsource, ny, nx = field.shape
    ntarget = target_pressure.shape[0]
    if (nsource < 1 or ny < 1 or nx < 1
            or source_pressure.shape != field.shape
            or target_pressure.shape[1:] != (ny, nx)
            or output.shape != target_pressure.shape):
        raise ValueError("vertical interpolation array shapes are inconsistent")
    ncolumn = ny * nx
    flat = {
        "field": field.view((nsource, ncolumn)),
        "source_p": source_pressure.view((nsource, ncolumn)),
        "target_p": target_pressure.view((ntarget, ncolumn)),
        "output": output.view((ntarget, ncolumn)),
    }
    return context.submit_kernel(
        "vertical_interpolate_logp", flat,
        {"nsource": nsource, "ntarget": ntarget,
         "ncolumn": ncolumn, "below_temperature": int(below == "temperature"),
         "p0": float(np.float32(c.P0)), "rcp": float(np.float32(c.RCP))},
        exact_policy=False, grid_size=ntarget * ncolumn)
