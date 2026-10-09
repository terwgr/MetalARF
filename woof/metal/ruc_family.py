"""Typed Apple Metal caller for the bounded RUC ``soilvegin`` stage."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_THREADS = 128
_I32_MAX = np.iinfo(np.int32).max
_INT_RESOURCES = {"isltyp", "ivgtyp", "ifortbl", "iforest_out"}
_INPUTS = (
    "isltyp", "ivgtyp", "shdmin", "shdmax", "vegfrac", "znt_in", "lai_in",
    "ifortbl", "z0tbl", "lemitbl", "pctbl", "laitbl", "bb", "drysmc",
    "hc", "maxsmc", "refsmc", "satpsi", "satdk", "wltsmc", "qtz",
)
_OUTPUTS = (
    "iforest_out", "emiss_out", "pc_out", "znt_out", "lai_out",
    "qwrtz_out", "rhocs_out", "bclh_out", "dqm_out", "ksat_out",
    "psis_out", "qmin_out", "ref_out", "wilt_out",
)


def submit_surface_parameters(*, arrays, iswater: int, rdlai2d: bool):
    """Submit WRF ``soilvegin`` for the dominant-category RUC lane.

    ``arrays`` maps the pinned CUDA resource names to caller-owned Metal
    buffers. Column resources are flat; lookup resources are the corresponding
    one-dimensional RUC VEGPARM/SOILPARM columns. Category validation remains
    the responsibility of the public physics caller before table indexing.
    """
    if type(iswater) is not int or not 1 <= iswater <= _I32_MAX:
        raise ValueError("RUC Metal iswater must be a positive int32 category")
    if type(rdlai2d) is not bool:
        raise TypeError("RUC Metal rdlai2d must be bool")
    missing = [name for name in (*_INPUTS, *_OUTPUTS) if name not in arrays]
    if missing:
        raise TypeError(f"missing RUC Metal surface resources: {', '.join(missing)}")
    first = arrays["isltyp"]
    if not isinstance(first, MetalArray) or first.ndim != 1:
        raise TypeError("RUC Metal isltyp must be a flat MetalArray")
    n = int(first.size)
    if n > _I32_MAX - (_THREADS - 1):
        raise ValueError("RUC Metal surface dispatch extent exceeds int32")
    context = first.buffer.context
    checked = {name: arrays[name] for name in (*_INPUTS, *_OUTPUTS)}
    for name, array in checked.items():
        if not isinstance(array, MetalArray):
            raise TypeError(f"RUC Metal {name} must be a MetalArray")
        dtype = np.dtype(np.int32) if name in _INT_RESOURCES else np.dtype(np.float32)
        if array.dtype != dtype or array.ndim != 1 or not array.flags.c_contiguous:
            raise TypeError(f"RUC Metal {name} requires contiguous flat {dtype.name} storage")
        if array.buffer.context is not context:
            raise ValueError("RUC Metal surface resources must share one context")
        if name in _INPUTS and array.size < n:
            raise ValueError(f"RUC Metal {name} has fewer than {n} elements")
        if name in _OUTPUTS and array.size != n:
            raise ValueError(f"RUC Metal {name} must have exactly {n} elements")
    # The kernel indexes caller-supplied parameter tables. Keep this first
    # public slice memory safe by checking the 1-based WRF selectors before
    # any device read; this bounded setup call explicitly accepts that host
    # readback and will not silently accept an invalid class.
    soil = arrays["isltyp"].to_numpy()
    vegetation = arrays["ivgtyp"].to_numpy()
    bad = (soil < 1) | (soil > arrays["bb"].size)
    if np.any(bad):
        raise ValueError(
            f"RUC Metal isltyp {int(soil[np.flatnonzero(bad)[0]])} is outside "
            f"1..{arrays['bb'].size}")
    bad = (vegetation < 1) | (vegetation > arrays["ifortbl"].size)
    if np.any(bad):
        raise ValueError(
            "RUC Metal ivgtyp "
            f"{int(vegetation[np.flatnonzero(bad)[0]])} is outside "
            f"1..{arrays['ifortbl'].size}")
    if n == 0:
        return None
    params = {"iswater": iswater, "rdlai2d": int(rdlai2d), "n": n}
    return context.submit_kernel(
        "ruc_surface_parameters", checked, params, exact_policy=True,
        grid_size=((n + _THREADS - 1) // _THREADS) * _THREADS,
        threads_per_group=_THREADS)


__all__ = ["submit_surface_parameters"]
