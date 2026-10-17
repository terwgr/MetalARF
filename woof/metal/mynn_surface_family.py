"""Typed Metal launcher for WRF's MYNN surface-layer column kernel."""

from __future__ import annotations

from collections.abc import Mapping
import math
import numpy as np

from .array import MetalArray

MYNN_SURFACE_INPUTS = (
    "u1", "v1", "t1", "qv1", "p1", "rho1", "dz1", "u2", "v2", "dz2",
    "psfc", "tsk", "pblh", "mavail", "hfx", "qfx", "znt", "qsfc", "ust",
    "xland", "snowh",
)
MYNN_SURFACE_OUTPUTS = (
    "regime", "zol", "rmol", "ust", "ustm", "mol", "psim", "psih", "chs",
    "chs2", "cqs2", "ch", "flhc", "flqc", "qgh", "qsfc", "hfx", "qfx",
    "lh", "u10", "v10", "th2", "t2", "q2", "gz1oz0", "wspd", "br", "ck",
    "cka", "cd", "cda", "wstar", "qstar", "cpm", "znt",
)


def submit_mynn_surface_seed(context, u1, v1, qv1, *, ust, mol, qsfc,
                             qstar):
    """Submit WRF's one-time SFCLAY_mynn cold-start field initialization."""
    arrays = {"u1": u1, "v1": v1, "qv1": qv1, "ust": ust,
              "mol": mol, "qsfc": qsfc, "qstar": qstar}
    shape = u1.shape
    if len(shape) != 2 or any(extent <= 0 for extent in shape):
        raise ValueError("MYNN surface seed arrays must be nonempty (ny, nx)")
    count = math.prod(shape)
    flattened = {}
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous
                or array.shape != shape
                or array.buffer.context is not context):
            raise ValueError(
                "MYNN surface seed arrays must be same-context, same-shaped "
                "contiguous FP32 MetalArrays")
        flattened[name] = array.view((count,))
    return context.submit_kernel(
        "mynn_surface_seed", flattened, {"n": count}, exact_policy=False,
        grid_size=count)


def _validate_options(dx, itimestep, isfflx, isftcflx):
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be positive and finite")
    if (isinstance(itimestep, (bool, np.bool_))
            or not isinstance(itimestep, (int, np.integer)) or itimestep < 1):
        raise ValueError("itimestep must be a positive integer")
    if isinstance(isfflx, (bool, np.bool_)) or isfflx not in (0, 1):
        raise ValueError("isfflx must be 0 or 1")
    if (isinstance(isftcflx, (bool, np.bool_))
            or not isinstance(isftcflx, (int, np.integer))
            or isftcflx not in (0, 1, 2, 3)):
        raise ValueError("isftcflx must be one of 0, 1, 2, or 3")


def submit_mynn_surface_layer(context, inputs: Mapping[str, object], mol,
                              ustm, result, *, dx=3000.0, itimestep=1,
                              isfflx=1, isftcflx=0, variant="wrf_461",
                              spp_pbl=0, pattern_spp_pbl=None):
    """Submit one independent surface column per flattened horizontal cell.

    Inputs, the persistent ``mol``/``ustm`` values, and every result field
    must be same-shaped contiguous FP32 MetalArrays. The existing caller
    owns first-step seeding, sea-ice staging, and persistent result storage.
    """
    _validate_options(dx, itimestep, isfflx, isftcflx)
    missing = [name for name in MYNN_SURFACE_INPUTS if name not in inputs]
    if missing:
        raise TypeError(f"missing MYNN surface inputs: {', '.join(missing)}")
    shape = inputs["u1"].shape
    if len(shape) != 2 or any(extent <= 0 for extent in shape):
        raise ValueError("MYNN surface inputs must be nonempty (ny, nx) arrays")

    resources = {}
    for name in MYNN_SURFACE_INPUTS:
        resources[f"input_{name}"] = inputs[name]
    resources["input_mol"] = mol
    resources["input_ustm"] = ustm
    if spp_pbl:
        if not isinstance(pattern_spp_pbl, MetalArray):
            raise ValueError("MYNN surface SPP requires a pattern_spp_pbl[ny,nx] MetalArray")
        if not np.isfinite(pattern_spp_pbl.to_numpy()).all():
            raise ValueError("MYNN surface SPP pattern must be finite")
    # Without SPP the kernel never reads the pattern slot; u1 fills it.
    resources["input_pattern_spp_pbl"] = (
        pattern_spp_pbl if spp_pbl else inputs["u1"])
    for name in MYNN_SURFACE_OUTPUTS:
        try:
            resources[f"output_{name}"] = getattr(result, name)
        except AttributeError as exc:
            raise TypeError(f"missing MYNN surface output: {name}") from exc

    count = math.prod(shape)
    flattened = {}
    for name, array in resources.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous
                or array.shape != shape):
            raise ValueError(
                "MYNN surface resources must be same-shaped contiguous "
                "FP32 MetalArrays")
        flattened[name] = array.view((count,))

    from woof.physics_params import metal_mynn_parameters
    return context.submit_kernel(
        "mynn_surface_column", flattened,
        {"dx": float(np.float32(dx)), "itimestep": int(itimestep),
         "isfflx": int(isfflx), "isftcflx": int(isftcflx), "n": count,
         "gsl_wrf39": int(variant == "gsl_wrf39"),
         "spp_pbl": int(bool(spp_pbl)),
         **metal_mynn_parameters("mynn_surface_column")},
        exact_policy=False, grid_size=count)
