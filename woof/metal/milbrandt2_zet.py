"""Typed Metal launcher for Milbrandt-Yau's pure radar Z block."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_milbrandt2_zet(temperature, pressure, qr, qi, qs, qg, qh,
                           nr, ni, ns, ng, nh, refl, ck, *,
                           concentration_space=False):
    fields = {"temperature": temperature, "pressure": pressure,
              "qr": qr, "qi": qi, "qs": qs, "qg": qg, "qh": qh,
              "nr": nr, "ni": ni, "ns": ns, "ng": ng, "nh": nh,
              "refl": refl}
    shape = tuple(refl.shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError(f"Milbrandt-Yau Z needs a nonempty 3-D field, got {shape}")
    nz, ny, nx = map(int, shape)
    n = nz * ny * nx
    if max(nz, ny, nx, n) > 0x7FFFFFFF:
        raise ValueError("Milbrandt-Yau Z dimensions must fit the int32 ABI")
    if not isinstance(concentration_space, bool):
        raise TypeError("concentration_space must be bool")
    arrays = {**fields, "ck": ck}
    context = refl.buffer.context
    for name, value in arrays.items():
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise TypeError(
                f"Milbrandt-Yau Z {name} must be same-context contiguous "
                "FP32 Metal storage")
        if name != "ck" and value.shape != shape:
            raise ValueError(f"Milbrandt-Yau Z {name} shape {value.shape} != {shape}")
    if ck.ndim != 1 or ck.size <= 153:
        raise ValueError("Milbrandt-Yau Z constants must contain indices through 153")
    return context.submit_kernel(
        "milbrandt2_zet", arrays,
        {"n": n, "nz": nz, "ny": ny, "nx": nx, "nck": int(ck.size),
         "concentration_space": int(concentration_space),
         "eps_q": float(np.float32(1.0e-14)),
         "eps_n": float(np.float32(1.0e-3)),
         "dielectric_factor": float(np.float32(4.464)),
         "z_factor": float(np.float32(1.0e18)),
         "minimum_zet": float(np.float32(-99.0)),
         "trpl": float(np.float32(273.16)),
         "rgas_d": float(np.float32(287.05)),
         "dbz_scale": float(np.float32(10.0))},
        exact_policy=False, grid_size=n, threads_per_group=256)


__all__ = ["submit_milbrandt2_zet"]
