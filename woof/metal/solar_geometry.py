"""Pinned WRF solar geometry on integer IEEE arithmetic in Metal kernels."""

from __future__ import annotations

import numpy as np

from .array import MetalArray, MetalArrayModule


def _word(value):
    return int(np.asarray(value, dtype=np.float32).view(np.int32))


def solar_geometry(julian, xtime, gmt, latitude, longitude, *,
                   generation="wrf_480", declination=None):
    """Return device ``(DECLIN/SOLCON, COSZEN, HRANG)`` float32 arrays.

    At least one coordinate must be a MetalArray, choosing the context.
    Geometry arithmetic stays on the device, including the pinned glibc
    sinf/cosf/asinf. Float32 arrays bind as uint32 views so native FTZ cannot
    alter subnormal inputs, intermediate words, or output storage.
    """
    from woof.core.solar import solar_generation
    from woof.core.rrtmg_legacy import _DEGRAD, _DPD

    solar_generation(generation)
    reference = latitude if isinstance(latitude, MetalArray) else longitude
    if not isinstance(reference, MetalArray):
        raise TypeError("Metal solar geometry needs a Metal coordinate array")
    context = reference.buffer.context
    arrays = MetalArrayModule(context)
    lat = arrays.asarray(latitude, dtype=np.float32)
    lon = arrays.asarray(longitude, dtype=np.float32)
    if lat.shape != lon.shape:
        raise ValueError("radiation latitude/longitude shapes must match")
    if lat.size > 0x7fffffff:
        raise ValueError("Metal solar geometry exceeds signed int32 indexing")
    scalars = np.asarray([julian, xtime, gmt], np.float32)
    if not np.isfinite(scalars).all():
        raise ValueError("solar calendar scalars must be finite")
    pair = arrays.empty((2,), np.float32)
    coszen = arrays.empty(lat.shape, np.float32)
    hrang = arrays.empty(lat.shape, np.float32)
    context.submit_kernel(
        "solar_radconst", {"declin_solcon": pair.view((2,), np.uint32)},
        {"julian": _word(scalars[0]), "degrad": _word(_DEGRAD),
         "dpd": _word(_DPD), "override_declin": int(declination is not None),
         "declin": _word(0.0 if declination is None else declination)},
        exact_policy=False, grid_size=1, threads_per_group=1)
    if lat.size:
        context.submit_kernel(
            "solar_coszen_hrang",
            {"latitude": lat.view((lat.size,), np.uint32),
             "longitude": lon.view((lon.size,), np.uint32),
             "declin_solcon": pair.view((2,), np.uint32),
             "coszen": coszen.view((lat.size,), np.uint32),
             "hrang": hrang.view((lat.size,), np.uint32)},
            {"count": lat.size, "julian": _word(scalars[0]),
             "xtime": _word(scalars[1]), "gmt": _word(scalars[2]),
             "degrad": _word(_DEGRAD), "solar_480": int(generation == "wrf_480")},
            exact_policy=False, grid_size=lat.size, threads_per_group=128)
    return pair, coszen, hrang


__all__ = ["solar_geometry"]
