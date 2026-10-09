"""Typed launcher for WRF's column Dudhia shortwave kernel."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_dudhia_shortwave(
        *, temperature, pressure, qv, qc, qr, qi, qs, qg, dz, exner,
        cosine_zenith, albedo, heating, swdown, gsw, coszen, solcon,
        swrad_scat, icloud):
    volumes = {"temperature": temperature, "pressure": pressure, "qv": qv,
               "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg,
               "dz": dz, "exner": exner, "heating": heating}
    first = temperature
    if not isinstance(first, MetalArray) or first.ndim != 3:
        raise TypeError("Dudhia radiation requires Metal (nz,ny,nx) columns")
    nz, ny, nx = map(int, first.shape)
    if not 0 < nz <= 0x7fffffff or ny <= 0 or nx <= 0:
        raise ValueError("Dudhia dimensions must be positive int32 values")
    context = first.buffer.context
    arrays = {**volumes, "cosine_zenith": cosine_zenith, "albedo": albedo,
              "swdown": swdown, "gsw": gsw, "coszen": coszen}
    surface_shape = (ny, nx)
    for name, array in arrays.items():
        expected = surface_shape if name in {
            "cosine_zenith", "albedo", "swdown", "gsw", "coszen"} else (nz, ny, nx)
        if (not isinstance(array, MetalArray) or array.shape != expected
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous):
            raise TypeError(f"{name} must be contiguous float32 Metal {expected}")
        if array.buffer.context is not context:
            raise ValueError("Dudhia resources must share one Metal context")
    if max(nz, ny, nx, ny * nx) > 0x7fffffff:
        raise ValueError("Dudhia dimensions exceed signed int32 indexing")
    solcon32 = np.float32(solcon)
    scat32 = np.float32(swrad_scat)
    if not np.isfinite(solcon32) or solcon32 <= 0:
        raise ValueError("Dudhia solar constant must be finite and positive")
    if not np.isfinite(scat32) or scat32 < 0:
        raise ValueError("Dudhia scattering coefficient must be finite and non-negative")
    if int(icloud) not in (0, 1):
        raise ValueError("Dudhia icloud must be zero or one")
    return context.submit_kernel(
        "dudhia_shortwave_columns", arrays,
        {"nz": nz, "ny": ny, "nx": nx, "solcon": float(solcon32),
         "swrad_scat": float(scat32), "icloud": int(icloud)},
        exact_policy=False, grid_size=ny * nx, threads_per_group=64)


__all__ = ["submit_dudhia_shortwave"]
