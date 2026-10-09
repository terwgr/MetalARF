"""Typed Metal launchers for WRF ``advance_ppt`` persistence bookkeeping."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray

MAX_VOLUMES = 12
_VOLUME_NAMES = tuple(f"v{i}" for i in range(MAX_VOLUMES))


def _require_array(name: str, array, shape, context) -> None:
    if not isinstance(array, MetalArray):
        raise TypeError(f"Metal cumulus clock {name} must be a MetalArray")
    if array.shape != shape:
        raise ValueError(
            f"Metal cumulus clock {name} shape {array.shape} != {shape}")
    if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
        raise TypeError(
            f"Metal cumulus clock {name} requires contiguous float32 storage")
    if array.buffer.context is not context:
        raise ValueError("Metal cumulus clock resources must share one context")


def submit_cumulus_clock_advance(*, rainc, pending_rainbl,
                                 surface_raincv, nca, pratec, expiring,
                                 dt):
    """Advance rain accumulators and the per-column cumulus countdown."""
    if not isinstance(nca, MetalArray) or nca.ndim != 2:
        raise TypeError("Metal cumulus clock nca must be a 2-D MetalArray")
    ny, nx = map(int, nca.shape)
    if min(ny, nx) <= 0 or ny * nx > 0x7FFFFFFF:
        raise ValueError("Metal cumulus clock dimensions must be positive int32")
    shape = (ny, nx)
    context = nca.buffer.context
    arrays = {"rainc": rainc, "pending_rainbl": pending_rainbl,
              "nca": nca, "pratec": pratec, "expiring": expiring}
    if surface_raincv is not None:
        arrays["surface_raincv"] = surface_raincv
    for name, array in arrays.items():
        _require_array(name, array, shape, context)
    # The unused optional resource still occupies its ABI slot.
    surface = rainc if surface_raincv is None else surface_raincv
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0:
        raise ValueError(f"cumulus clock dt must be finite and positive: {dt!r}")
    resources = {"rainc": rainc, "pending_rainbl": pending_rainbl,
                 "surface_raincv": surface, "nca": nca,
                 "pratec": pratec, "expiring": expiring}
    resources = {
        name: value.view((ny * nx,)) for name, value in resources.items()
    }
    return context.submit_kernel(
        "cumulus_clock_advance", resources,
        {"dt": float(dt32), "has_surface_raincv": int(surface_raincv is not None),
         "ncol": ny * nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=128)


def submit_cumulus_clock_clear(expiring, volumes):
    """Clear up to twelve held rate volumes under the expiry mask, then reset it."""
    if not isinstance(expiring, MetalArray) or expiring.ndim != 2:
        raise TypeError("Metal cumulus clock expiring must be a 2-D MetalArray")
    volumes = list(volumes)
    if len(volumes) > MAX_VOLUMES:
        raise ValueError(
            f"Metal cumulus clock clears at most {MAX_VOLUMES} volumes")
    ny, nx = map(int, expiring.shape)
    if min(ny, nx) <= 0 or ny * nx > 0x7FFFFFFF:
        raise ValueError("Metal cumulus clock dimensions must be positive int32")
    context = expiring.buffer.context
    _require_array("expiring", expiring, (ny, nx), context)
    if volumes:
        if not isinstance(volumes[0], MetalArray) or volumes[0].ndim != 3:
            raise TypeError("Metal cumulus clock volumes must be 3-D MetalArrays")
        nz = int(volumes[0].shape[0])
    else:
        nz = 0
    if nz < 0 or nz > 0x7FFFFFFF:
        raise ValueError("Metal cumulus clock nz must fit int32")
    expected = (nz, ny, nx)
    for i, volume in enumerate(volumes):
        _require_array(f"volume {i}", volume, expected, context)
    # Unused ABI slots alias expiring, which is not read for j >= nvolume.
    pad = volumes[0] if volumes else expiring
    slots = volumes + [pad] * (MAX_VOLUMES - len(volumes))
    resources = {"expiring": expiring}
    resources.update({name: value for name, value in zip(_VOLUME_NAMES, slots)})
    resources = {
        name: value.view((int(value.size),))
        for name, value in resources.items()
    }
    return context.submit_kernel(
        "cumulus_clock_clear", resources,
        {"nvolume": len(volumes), "nz": nz, "ncol": ny * nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=128)


__all__ = ["MAX_VOLUMES", "submit_cumulus_clock_advance",
           "submit_cumulus_clock_clear"]
