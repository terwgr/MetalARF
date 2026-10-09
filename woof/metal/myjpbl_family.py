"""Typed Metal launcher for the MYJ column PBL kernel."""

from __future__ import annotations

import numpy as np


_COLUMN_INPUTS = ("dz", "u", "v", "t", "th", "exner", "qv", "qc",
                  "qi", "p")
_SURFACE_INPUTS = ("psfc", "ust", "tsk", "chklowq", "xland", "sice",
                   "snow", "akhs", "akms", "elflx", "uz0", "vz0")
_PBL_STATE = ("thz0", "qz0", "qsfc", "ct")
_COLUMN_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                   "rqcblten", "rqiblten", "el_myj", "exch_h")


def submit_myj_pbl(domain_state, columns, surface, pbl_state, tke, *,
                   dtturbl: float, flqi: bool):
    """Submit one bottom-up MYJ column step and return its WRF output map.

    Inputs must already be same-context Metal arrays. This launcher only
    validates and submits resources; it does not emulate CuPy preparation.
    """
    from .array import MetalArray

    if not getattr(domain_state, "_metal_setup_state", False):
        raise TypeError("Metal MYJ requires a Metal DomainState")
    if set(_PBL_STATE) - set(pbl_state):
        raise KeyError("MYJ state requires thz0, qz0, qsfc, and ct")
    if set(_SURFACE_INPUTS) - set(surface):
        raise KeyError("MYJ surface input mapping is incomplete")
    if set(_COLUMN_INPUTS) - set(columns):
        missing = set(_COLUMN_INPUTS) - set(columns)
        if missing != {"qi"} or flqi:
            raise KeyError(f"MYJ column input mapping is missing {sorted(missing)}")

    dz = columns["dz"]
    if not isinstance(dz, MetalArray) or dz.ndim != 3:
        raise TypeError("MYJ dz must be a MetalArray with shape (nz, ny, nx)")
    nz, ny, nx = dz.shape
    if nz < 4 or nz > 128:
        raise ValueError("MYJ requires 4 <= nz <= MYJ_KMAX (128)")
    shape3 = (nz, ny, nx)
    shape2 = (ny, nx)

    arrays = {}
    for name in _COLUMN_INPUTS:
        value = columns.get(name)
        if value is None:
            if name != "qi" or flqi:
                raise KeyError(f"MYJ requires column input {name!r}")
            value = columns["qc"]
        if (not isinstance(value, MetalArray) or value.dtype != np.dtype(np.float32)
                or value.shape != shape3 or not value.flags.c_contiguous):
            raise TypeError(f"MYJ {name} must be contiguous float32 {shape3}")
        arrays[f"{name}_a"] = value
    if (not isinstance(tke, MetalArray) or tke.dtype != np.dtype(np.float32)
            or tke.shape != shape3 or not tke.flags.c_contiguous):
        raise TypeError(f"MYJ TKE must be contiguous float32 {shape3}")
    arrays["tke_a"] = tke

    for name in _SURFACE_INPUTS:
        value = surface[name]
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or value.shape != shape2 or not value.flags.c_contiguous):
            raise TypeError(f"MYJ surface {name} must be contiguous float32 {shape2}")
        arrays[f"{name}_a"] = value
    for name in _PBL_STATE:
        value = pbl_state[name]
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or value.shape != shape2 or not value.flags.c_contiguous):
            raise TypeError(f"MYJ state {name} must be contiguous float32 {shape2}")
        arrays[f"{name}_a"] = value

    outputs = {
        name: domain_state.scratch(shape3, f"myjpbl_{name}")
        for name in _COLUMN_OUTPUTS
    }
    outputs["pblh"] = domain_state.scratch(shape2, "myjpbl_pblh")
    outputs["mixht"] = domain_state.scratch(shape2, "myjpbl_mixht")
    outputs["kpbl"] = domain_state.scratch(
        shape2, "myjpbl_kpbl", dtype=np.int32)
    arrays.update({f"{name}_a": value for name, value in outputs.items()})

    n = ny * nx
    domain_state._array_module.context.submit_kernel(
        "myjpbl_column", arrays,
        {"dtturbl": float(dtturbl), "flqi": int(bool(flqi)),
         "nz": nz, "ny": ny, "nx": nx, "n": n},
        exact_policy=False, grid_size=n)

    outputs.update(
        du=outputs["rublten"], dv=outputs["rvblten"],
        dtheta=outputs["rthblten"], dqv=outputs["rqvblten"],
        dqc=outputs["rqcblten"])
    if flqi:
        outputs["dqi"] = outputs["rqiblten"]
    return outputs
