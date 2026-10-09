"""Typed Metal caller for the UW moist-turbulence PBL column kernel."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .array import MetalArray, MetalArrayModule
from .runtime import MetalContext
from woof.wrf_exact import ENABLED as WRF_EXACT
from woof.core.physics_inventory import (
    UWPBL_DIAGNOSTICS_FULL, UWPBL_MASS_OUTPUTS, UWPBL_SURFACE_OUTPUTS,
    UWPBL_BLOCK, UWPBL_WORKSPACE_BUDGET_BYTES, uwpbl_chunk_columns,
    uwpbl_workspace_slots,
)
from woof.physics_vertical_contract import (
    UWPBL_VERTICAL_LEVEL_BOUNDS, outside_vertical_bounds,
    refuse_vertical_levels,
)

MASS_INPUTS = ("u", "v", "th", "rho", "qv", "qc", "qi", "qnc", "qni",
               "p", "z", "t", "cldfra", "exner", "rthratenlw", "wsedl3d")
FULL_INPUTS = ("p8w", "z_at_w")
SURFACE_INPUTS = ("hfx", "qfx", "ust", "ht")
CARRIED_FULL = ("kvm3d", "kvh3d")
CARRIED_SURFACE = ("tauresx2d", "tauresy2d")
_TPB = UWPBL_BLOCK


def _check(name, value, context, shape, dtype=np.float32):
    if (not isinstance(value, MetalArray) or value.shape != shape
            or value.dtype != np.dtype(dtype) or not value.flags.c_contiguous):
        raise ValueError(
            f"UW PBL {name} must be contiguous {np.dtype(dtype)} MetalArray "
            f"with shape {shape}")
    if value.buffer.context is not context:
        raise ValueError(f"UW PBL {name} belongs to a different Metal context")


def submit_uwpbl_columns(context: MetalContext, inputs, carried, *, dt: float,
                         itimestep: int,
                         budget_bytes: int = UWPBL_WORKSPACE_BUDGET_BYTES):
    """Submit the UW PBL in bounded column chunks; inputs/state are Metal arrays."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal UW PBL supports default-policy arithmetic only; "
            "WRF-exact arithmetic remains unavailable")
    if not isinstance(context, MetalContext):
        raise TypeError("Metal UW PBL requires a MetalContext")
    if not isinstance(inputs, Mapping) or not isinstance(carried, Mapping):
        raise TypeError("UW PBL inputs and carried state must be mappings")
    missing = set(MASS_INPUTS + FULL_INPUTS + SURFACE_INPUTS) - set(inputs)
    if missing:
        raise KeyError(f"UW PBL inputs are missing {sorted(missing)}")
    missing = set(CARRIED_FULL + CARRIED_SURFACE) - set(carried)
    if missing:
        raise KeyError(f"UW PBL state is missing {sorted(missing)}")
    if isinstance(dt, bool) or not np.isfinite(dt) or float(dt) <= 0:
        raise ValueError(f"UW PBL requires positive finite dt, got {dt}")
    if isinstance(itimestep, bool) or int(itimestep) < 1:
        raise ValueError("UW PBL timestep is one-based and must be positive")

    u = inputs["u"]
    if not isinstance(u, MetalArray) or u.ndim != 3:
        raise TypeError("UW PBL u must be a 3-D MetalArray")
    nz, ny, nx = u.shape
    if nz < 2 or ny < 1 or nx < 1:
        raise ValueError("UW PBL requires at least two levels and nonempty columns")
    if outside_vertical_bounds(nz, UWPBL_VERTICAL_LEVEL_BOUNDS):
        raise refuse_vertical_levels(
            "UW PBL", UWPBL_VERTICAL_LEVEL_BOUNDS, nz,
            breakage=(
                "the implicit diffusion needs at least one interior "
                "interface (compute_vdiff's k = 2..pver rows) and the "
                "scheme's Richardson numbers live on interfaces above the "
                "surface; select at least two model levels."))
    mass_shape, full_shape, surface_shape = (nz, ny, nx), (nz + 1, ny, nx), (ny, nx)
    for name in MASS_INPUTS:
        _check(name, inputs[name], context, mass_shape)
    for name in FULL_INPUTS:
        _check(name, inputs[name], context, full_shape)
    for name in SURFACE_INPUTS:
        _check(name, inputs[name], context, surface_shape)
    for name in CARRIED_FULL:
        _check(name, carried[name], context, full_shape)
    for name in CARRIED_SURFACE:
        _check(name, carried[name], context, surface_shape)

    xp = MetalArrayModule(context)
    outputs = {name: xp.empty(mass_shape, dtype=np.float32)
               for name in UWPBL_MASS_OUTPUTS}
    outputs.update({name: xp.empty(full_shape, dtype=np.float32)
                    for name in UWPBL_DIAGNOSTICS_FULL})
    outputs.update({name: xp.empty(surface_shape, dtype=np.float32)
                    for name in UWPBL_SURFACE_OUTPUTS})
    outputs["kpbl2d"] = xp.empty(surface_shape, dtype=np.int32)

    ncols = ny * nx
    chunk = uwpbl_chunk_columns(nz, ncols, int(budget_bytes))
    r8cap, i4cap = uwpbl_workspace_slots(nz)
    r8pool = xp.empty((r8cap * chunk,), dtype=np.float64)
    i4pool = xp.empty((i4cap * chunk,), dtype=np.int32)
    err = xp.zeros((1,), dtype=np.int32)
    fixed = {name: inputs[name] for name in MASS_INPUTS + FULL_INPUTS + SURFACE_INPUTS}
    fixed.update({name: carried[name] for name in CARRIED_FULL + CARRIED_SURFACE})
    fixed.update(outputs)

    for col0 in range(0, ncols, chunk):
        count = min(chunk, ncols - col0)
        blocks = math.ceil(count / _TPB)
        arrays = dict(fixed, r8pool=r8pool, i4pool=i4pool, err=err)
        context.submit_kernel(
            "uwpbl_columns", arrays,
            {"ncols": ncols, "col0": col0, "nchunk": count,
             "pool_stride": chunk, "nk": nz, "ny": ny, "nx": nx,
             "r8size": r8cap * chunk, "i4size": i4cap * chunk,
             "dt": float(dt),
             "itimestep": int(itimestep), "r8cap": r8cap,
             "i4cap": i4cap},
            exact_policy=False, grid_size=count, threads_per_group=_TPB)
    status = int(err.to_numpy().reshape(-1)[0])
    if status == 2:
        raise RuntimeError(
            "UW PBL diffusion solver refused its field list; outputs are not trustworthy")
    if status:
        if status != 1:
            raise RuntimeError(
                f"UW PBL reported device stop code {status}; outputs are not trustworthy")
        raise RuntimeError(
            f"UW PBL workspace overflowed ({r8cap} binary64 and "
            f"{i4cap} integer slots at nz={nz}); outputs are not trustworthy")
    outputs.update({"du": outputs["rublten"], "dv": outputs["rvblten"],
                    "dtheta": outputs["rthblten"], "dqv": outputs["rqvblten"],
                    "dqc": outputs["rqcblten"], "dqi": outputs["rqiblten"],
                    "dqni": outputs["rqniblten"], "pblh": outputs["pblh2d"],
                    "kpbl": outputs["kpbl2d"]})
    return outputs
