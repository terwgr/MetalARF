"""Typed Metal submission for P3 homogeneous freezing."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from .p3_constants import P3_CONSTANT_NAMES
from woof.wrf_exact import ENABLED as WRF_EXACT


_VOLUMES = (
    "theta", "temperature", "tmparr1", "qv", "qv_cld", "qc", "nc",
    "qr", "nr", "qi", "qir", "ni", "qib",
)


def submit_p3_homofreeze(*, fields: dict, scratch: dict, flags,
                         constants, xlf) -> None:
    """Apply homogeneous freezing to flagged columns on Metal.

    ``fields`` contains P3's mutable species/theta and read-only temperature;
    ``scratch`` provides ``tmparr1`` and the qv snapshot output. All volume
    arrays are contiguous FP32 ``(nz,ny,nx)`` on one Metal context.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "P3 Metal homogeneous freezing supports ordinary float32 only")
    missing = [name for name in _VOLUMES
               if name not in (fields.keys() | scratch.keys())]
    if missing:
        raise ValueError("P3 Metal homogeneous freezing is missing arrays: "
                         + ", ".join(missing))
    theta = fields.get("theta", scratch.get("theta"))
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("P3 Metal homogeneous freezing theta must be 3-D MetalArray")
    nz, ny, nx = theta.shape
    if min(nz, ny, nx) < 1:
        raise ValueError("P3 Metal homogeneous freezing dimensions must be nonempty")
    context = theta.buffer.context
    arrays = {}
    for name in _VOLUMES:
        value = fields.get(name, scratch.get(name))
        if (not isinstance(value, MetalArray) or value.shape != (nz, ny, nx)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal homogeneous freezing {name} must be contiguous FP32 {(nz, ny, nx)}")
        if value.buffer.context is not context:
            raise ValueError(
                f"P3 Metal homogeneous freezing {name} belongs to another context")
        arrays[name] = value
    if (not isinstance(flags, MetalArray) or flags.shape != (2, ny, nx)
            or flags.dtype != np.dtype(np.float32)
            or not flags.flags.c_contiguous):
        raise TypeError(
            f"P3 Metal homogeneous freezing flags must be contiguous FP32 {(2, ny, nx)}")
    if flags.buffer.context is not context:
        raise ValueError("P3 Metal homogeneous freezing flags use another context")
    arrays["flags"] = flags
    if (not isinstance(constants, MetalArray)
            or constants.shape != (len(P3_CONSTANT_NAMES),)
            or constants.dtype != np.dtype(np.float32)
            or not constants.flags.c_contiguous):
        raise TypeError(
            "P3 Metal homogeneous freezing constants must be contiguous FP32 "
            f"({len(P3_CONSTANT_NAMES)},)")
    if constants.buffer.context is not context:
        raise ValueError("P3 Metal homogeneous freezing constants use another context")
    arrays["constants"] = constants
    latent = np.float32(xlf)
    if not np.isfinite(latent):
        raise ValueError("P3 Metal homogeneous freezing latent heat must be finite FP32")
    context.submit_kernel(
        "p3_homofreeze", arrays,
        {"nz": nz, "ny": ny, "nx": nx, "xlf": float(latent)},
        exact_policy=False, grid_size=ny * nx)


__all__ = ["submit_p3_homofreeze"]
