"""Typed Metal submission for P3's wrapper/entry initialization stage.

This is a production slice, not an MP50 backend: it implements only the
entry initialization preceding P3's prognostic process chain. The caller must
not admit P3 until the remaining process, sedimentation, freezing, finish,
table, and radiation-facing stages have complete Metal routes.
"""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


def submit_p3_entry_prepare(*, theta, pressure, qv, h_diabatic,
                            temperature, tmparr1,
                            zdbz, effc, effi, vmi, di, rhopo,
                            prt_liq, prt_sol, flags, exponent: float) -> None:
    """Initialize P3's temperature carriers, diagnostics, and entry flags.

    Volumes are contiguous level-major FP32 ``(nz, ny, nx)`` fields;
    precipitation outputs are FP32 ``(ny, nx)`` fields and ``flags`` is
    FP32 ``(2, ny, nx)``. The inputs and outputs remain on one Metal context.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "P3 Metal entry supports ordinary float32 arithmetic only; "
            "source-pinned r_pow is not implemented")
    volumes = {
        "theta": theta, "pressure": pressure, "qv": qv,
        "h_diabatic": h_diabatic, "temperature": temperature,
        "tmparr1": tmparr1,
        "zdbz": zdbz, "effc": effc, "effi": effi,
        "vmi": vmi, "di": di, "rhopo": rhopo,
    }
    if not isinstance(theta, MetalArray) or len(theta.shape) != 3:
        raise TypeError("P3 Metal entry theta must be a 3-D MetalArray")
    nz, ny, nx = theta.shape
    if min(nz, ny, nx) < 1:
        raise ValueError("P3 Metal entry dimensions must be nonempty")
    shape3 = (nz, ny, nx)
    context = theta.buffer.context
    for name, value in volumes.items():
        if (not isinstance(value, MetalArray) or value.shape != shape3
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"P3 Metal entry {name} must be contiguous FP32 {shape3}")
        if value.buffer.context is not context:
            raise ValueError(f"P3 Metal entry {name} belongs to another context")

    surface_shape = (ny, nx)
    surfaces = {"prt_liq": prt_liq, "prt_sol": prt_sol}
    for name, value in surfaces.items():
        if (not isinstance(value, MetalArray) or value.shape != surface_shape
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"P3 Metal entry {name} must be contiguous FP32 {surface_shape}")
        if value.buffer.context is not context:
            raise ValueError(f"P3 Metal entry {name} belongs to another context")
    if (not isinstance(flags, MetalArray) or flags.shape != (2, ny, nx)
            or flags.dtype != np.dtype(np.float32)
            or not flags.flags.c_contiguous):
        raise TypeError(
            f"P3 Metal entry flags must be contiguous FP32 {(2, ny, nx)}")
    if flags.buffer.context is not context:
        raise ValueError("P3 Metal entry flags belong to another context")

    parameters = {
        "nz": nz, "ny": ny, "nx": nx,
        "exponent": float(np.float32(exponent)),
    }
    context.submit_kernel(
        "p3_entry_prepare", {**volumes, **surfaces, "flags": flags},
        parameters, exact_policy=False, grid_size=ny * nx)


__all__ = ["submit_p3_entry_prepare"]
