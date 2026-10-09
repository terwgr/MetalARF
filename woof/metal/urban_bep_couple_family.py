"""Typed Metal submitters for BEP's post-LSM coupling and surface diagnostics."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def submit_urban_bep_couple(context, resources, *, lsm, nz, ny, nx):
    scalar = ("frc_urb2d", "glw", "swdown", "rl_up_urb", "rs_abs_urb",
              "emiss_urb", "grdflx_urb", "ust", "tsk", "hfx", "qfx",
              "lh", "grdflx", "albedo", "emiss", "ts_urb2d", "sh_urb2d",
              "lh_urb2d", "g_urb2d", "rn_urb2d")
    grid = ("dz8w", "rho", "u_phy", "v_phy", "a_u_bep", "a_v_bep",
            "a_t_bep", "a_q_bep", "a_e_bep", "b_u_bep", "b_v_bep",
            "b_t_bep", "b_q_bep", "b_e_bep", "sf_bep", "vl_bep")
    if set(resources) != set(scalar + grid):
        raise ValueError("BEP coupling resources do not match the Metal signature")
    ncol = int(ny) * int(nx)
    nlevel = int(nz) * ncol
    for name in scalar:
        arr = resources[name]
        if (not isinstance(arr, MetalArray) or arr.dtype != np.dtype(np.float32)
                or arr.shape != (ncol,) or arr.buffer.context is not context):
            raise ValueError(f"BEP coupling {name} must be a resident column vector")
    for name in grid:
        arr = resources[name]
        want = nlevel + ncol if name == "sf_bep" and arr.size == nlevel + ncol else nlevel
        if (not isinstance(arr, MetalArray) or arr.dtype != np.dtype(np.float32)
                or arr.ndim != 1 or arr.size != want
                or arr.buffer.context is not context):
            raise ValueError(f"BEP coupling {name} has the wrong level extent")
    if int(lsm) not in (2, 4) or min(int(nz), int(ny), int(nx)) <= 0:
        raise ValueError("BEP coupling requires Noah/Noah-MP and positive dimensions")
    return context.submit_kernel(
        "urban_bep_couple", resources,
        {"lsm": int(lsm), "nz": int(nz), "ny": int(ny), "nx": int(nx),
         "ncol": ncol},
        exact_policy=False, grid_size=ncol, threads_per_group=128)


def submit_urban_bep_sfcdiag(context, resources, *, nz, ny, nx):
    names = ("utype_urb2d", "th_phy", "qv", "u_phy", "v_phy", "psfc",
             "t2", "th2", "q2", "u10", "v10")
    if set(resources) != set(names):
        raise ValueError("BEP surface diagnostic resources do not match signature")
    ncol = int(ny) * int(nx)
    for name in names:
        arr = resources[name]
        dtype = np.int32 if name == "utype_urb2d" else np.float32
        want = ncol * int(nz) if name in ("th_phy", "qv", "u_phy", "v_phy") else ncol
        if (not isinstance(arr, MetalArray) or arr.dtype != np.dtype(dtype)
                or arr.ndim != 1 or arr.size != want
                or arr.buffer.context is not context):
            raise ValueError(f"BEP surface diagnostic {name} has the wrong extent")
    return context.submit_kernel(
        "urban_bep_sfcdiag", resources,
        {"ny": int(ny), "nx": int(nx), "ncol": int(ny) * int(nx)},
        exact_policy=False,
        grid_size=ncol, threads_per_group=128)
