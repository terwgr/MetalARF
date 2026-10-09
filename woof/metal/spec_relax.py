"""Launcher for the ordinary FP32 specified lateral relaxation kernel."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from ..wrf_exact import ENABLED as WRF_EXACT


def submit_specified_relaxation(state, field, tendency, boundary, *, dtbc,
                                width, spec_zone, relax_zone, apply_relax,
                                weights):
    """Submit the CUDA-equivalent specified-zone and Davies relaxation pass.

    ``boundary`` must expose device-resident west/east/south/north value and
    tendency arrays. ``weights`` is the prepared pair ``(fcx, gcx)``. The
    caller owns time-law evaluation and weight scratch preparation so their
    clock and lifetime remain consistent with the lateral-boundary state.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal specified_relaxation supports ordinary non-exact arithmetic")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal specified_relaxation requires a Metal DomainState")
    if not isinstance(field, MetalArray) or not isinstance(tendency, MetalArray):
        raise TypeError("Metal specified_relaxation requires MetalArray fields")
    if field.dtype != np.dtype(np.float32) or tendency.dtype != np.dtype(np.float32):
        raise TypeError("Metal specified_relaxation requires FP32 fields")
    if field.ndim != 3 or tendency.shape != field.shape:
        raise ValueError("field and tendency must have the same 3-D shape")
    nz, ny, nx = field.shape
    width, spec_zone, relax_zone = map(int, (width, spec_zone, relax_zone))
    if width < spec_zone + relax_zone:
        raise ValueError("boundary width is smaller than spec_zone + relax_zone")
    if min(nz, ny, nx) <= 0 or spec_zone < 0 or relax_zone < 0:
        raise ValueError("invalid specified-relaxation domain or zone widths")
    if not np.isfinite(dtbc):
        raise ValueError("boundary time offset must be finite")
    sides = {}
    expected = {
        "west": (nz, ny, width), "east": (nz, ny, width),
        "south": (nz, width, nx), "north": (nz, width, nx),
    }
    for side_name, shape in expected.items():
        side = getattr(boundary, side_name)
        value, side_tendency = side.value, side.tendency
        for name, array in ((side_name, value),
                            (f"{side_name}_t", side_tendency)):
            if (not isinstance(array, MetalArray)
                    or array.dtype != np.dtype(np.float32)
                    or array.shape != shape):
                raise TypeError(
                    f"Metal boundary {name} must be FP32 with shape {shape}")
            sides[name] = array
        if getattr(side, "time_law", None) is not None:
            raise NotImplementedError(
                "Metal specified_relaxation requires caller-evaluated "
                "boundary time laws")
    if not isinstance(weights, (tuple, list)) or len(weights) != 2:
        raise TypeError("weights must be the prepared (fcx, gcx) pair")
    fcx, gcx = weights
    for name, array in (("fcx", fcx), ("gcx", gcx)):
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or array.shape != (width,)):
            raise TypeError(f"{name} must be an FP32 MetalArray of shape {(width,)}")
    arrays = {"field": field, "field_tend": tendency, **sides,
              "fcx": fcx, "gcx": gcx}
    parameters = {
        "dtbc": float(np.float32(dtbc)), "width": width,
        "spec_zone": spec_zone, "relax_zone": relax_zone,
        "apply_relax": int(bool(apply_relax)),
        "nz": nz, "ny": ny, "nx": nx,
    }
    return state._array_module.context.submit_kernel(
        "specified_relaxation", arrays, parameters, exact_policy=False,
        grid_size=nz * ny * nx)
