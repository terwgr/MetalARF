"""Metal WRF PD scalar-source folding for admitted regional physics lanes."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_pd_source_fold(q0: MetalArray, physics: MetalArray | None,
                          fixed: MetalArray | None,
                          c1h: MetalArray, c2h: MetalArray,
                          mu0: MetalArray, dt_eff: float,
                          q0_eff: MetalArray, *, lbc_held=None) -> None:
    """Fold fixed, physics, and held LBC sources into the time-t scalar.

    WRF's final PD scalar path folds fixed, physics, and held LBC sources
    into the time-t scalar, in that order, before constructing limiter
    fluxes. This operation keeps the source and result resident on Metal
    and preserves the separate FP32 multiply/add/divide/add rounding points.
    """
    has_physics = physics is not None
    has_fixed = fixed is not None
    has_lbc = lbc_held is not None
    physics = q0 if physics is None else physics
    fixed = q0 if fixed is None else fixed
    lbc_held = q0 if lbc_held is None else lbc_held
    arrays = (q0, physics, fixed, lbc_held, c1h, c2h, mu0, q0_eff)
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError("Metal PD source fold requires MetalArray operands")
    if q0.ndim != 3 or q0.dtype != np.dtype(np.float32):
        raise TypeError("Metal PD source fold needs a 3-D FP32 scalar")
    nz, ny, nx = q0.shape
    expected = (
        (nz, ny, nx), (nz, ny, nx), (nz, ny, nx), (nz, ny, nx),
        (nz,), (nz,), (ny, nx), (nz, ny, nx))
    if tuple(value.shape for value in arrays) != expected:
        raise ValueError("Metal PD source-fold operand shapes do not agree")
    if any(value.dtype != np.dtype(np.float32) for value in arrays):
        raise TypeError("Metal PD source-fold operands must be FP32")
    contiguous = (
        (ny * nx * 4, nx * 4, 4), (ny * nx * 4, nx * 4, 4),
        (ny * nx * 4, nx * 4, 4), (ny * nx * 4, nx * 4, 4),
        (4,), (4,), (nx * 4, 4), (ny * nx * 4, nx * 4, 4))
    if any(value.strides != stride for value, stride in zip(arrays, contiguous)):
        raise ValueError("Metal PD source-fold operands must be contiguous")
    context = q0.buffer.context
    if any(value.buffer.context is not context for value in arrays[1:]):
        raise ValueError("Metal PD source-fold operands need one context")
    out_lo, out_hi = q0_eff.byte_offset, q0_eff.byte_offset + q0_eff.nbytes
    for name, value, active in zip(
            ("q0", "physics", "fixed", "lbc", "c1h", "c2h", "mu0"),
            arrays[:-1],
            (True, has_physics, has_fixed, has_lbc, True, True, True)):
        if not active:
            continue
        if (value.buffer.handle == q0_eff.buffer.handle
                and out_lo < value.byte_offset + value.nbytes
                and value.byte_offset < out_hi):
            raise ValueError(f"PD source-fold output overlaps {name}")
    count = nz * ny * nx
    if count > 0xFFFFFFFF:
        raise ValueError("Metal PD source-fold grid does not fit uint32")
    context.submit_kernel(
        "moist_pd_source_fold",
        {"q0": q0, "physics": physics, "fixed": fixed, "lbc": lbc_held,
         "c1h": c1h, "c2h": c2h, "mu0": mu0, "q0_eff": q0_eff},
        {"dt_eff": float(np.float32(dt_eff)),
         "has_physics": int(has_physics), "has_fixed": int(has_fixed),
         "has_lbc": int(has_lbc),
         "nz": nz, "ny": ny, "nx": nx, "n": count},
        exact_policy=False, grid_size=count)
