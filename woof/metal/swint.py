"""Typed Metal launchers for WRF ``swint_opt = 1`` interpolation."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from .runtime import MetalContext


def _validate(values: dict[str, MetalArray], context: MetalContext,
              n: int) -> None:
    for name, value in values.items():
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or value.shape != (n,) or not value.flags.c_contiguous):
            raise ValueError(
                f"swint {name} must be contiguous float32 MetalArray ({n},)")
        if value.buffer.context is not context:
            raise ValueError(f"swint {name} belongs to another Metal context")


def submit_swint_coszen_loc(latitude: MetalArray, longitude: MetalArray,
                            output: MetalArray, *, julian: float,
                            xtime: float, gmt: float, degrad: float,
                            dpd: float) -> MetalArray:
    """Compute the current-time solar cosine for every surface column."""
    if not isinstance(latitude, MetalArray):
        raise TypeError("swint latitude must be a MetalArray")
    n = latitude.size
    context = latitude.buffer.context
    _validate({"latitude": latitude, "longitude": longitude,
               "coszen_loc": output}, context, n)
    if n:
        context.submit_kernel(
            "swint_coszen_loc_fragment",
            {"latitude": latitude, "longitude": longitude,
             "coszen_loc": output},
            {"count": n, "julian": float(julian), "xtime": float(xtime),
             "gmt": float(gmt), "degrad": float(degrad), "dpd": float(dpd)},
            exact_policy=False, grid_size=n, threads_per_group=128)
    return output


def submit_swint_update(*, n: int, coszen: MetalArray,
                        coszen_loc: MetalArray, swddir: MetalArray,
                        swdown: MetalArray, swddir_ref: MetalArray,
                        bb: MetalArray, bx: MetalArray,
                        swdown_ref: MetalArray, gg: MetalArray,
                        gx: MetalArray, coszen_ref: MetalArray) -> None:
    """Fit the direct/global flux exponents and store the radiation anchor."""
    values = {name: value for name, value in locals().items()
              if name != "n" and isinstance(value, MetalArray)}
    if not values:
        return
    context = next(iter(values.values())).buffer.context
    _validate(values, context, int(n))
    if n:
        context.submit_kernel(
            "swint_update_fragment", values, {"count": int(n)},
            exact_policy=False, grid_size=int(n), threads_per_group=128)


def submit_swint_interp(*, n: int, coszen_ref: MetalArray,
                        coszen_loc: MetalArray, swddir_ref: MetalArray,
                        bb: MetalArray, bx: MetalArray,
                        swdown_ref: MetalArray, gg: MetalArray,
                        gx: MetalArray, albedo: MetalArray,
                        swdown: MetalArray, swddir: MetalArray,
                        swddni: MetalArray, swddif: MetalArray,
                        gsw: MetalArray) -> None:
    """Evaluate the fitted shortwave fields at this step's solar angle."""
    values = {name: value for name, value in locals().items()
              if name != "n" and isinstance(value, MetalArray)}
    if not values:
        return
    context = next(iter(values.values())).buffer.context
    _validate(values, context, int(n))
    if n:
        context.submit_kernel(
            "swint_interp_fragment", values, {"count": int(n)},
            exact_policy=False, grid_size=int(n), threads_per_group=128)


__all__ = ["submit_swint_coszen_loc", "submit_swint_update",
           "submit_swint_interp"]
