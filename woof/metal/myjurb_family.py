"""Typed Metal launcher for WRF MYJURB (MYJ under BEP/BEP+BEM)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


_COLUMN_INPUTS = ("dz", "u", "v", "t", "th", "exner", "qv", "qc", "p")
_SURFACE_INPUTS = ("psfc", "ust", "tsk", "chklowq", "xland", "sice", "snow",
                   "akhs", "akms", "elflx", "uz0", "vz0")
_PBL_STATE = ("thz0", "qz0", "qsfc", "ct")
_COLUMN_OUTPUTS = ("rublten", "rvblten", "rthblten", "rqvblten",
                   "rqcblten", "el_myj", "exch_h", "exch_m")
_SURFACE_OUTPUTS = ("pblh", "kpbl", "mixht")
_BEP_INPUTS = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "a_e_bep",
               "b_u_bep", "b_v_bep", "b_t_bep", "b_q_bep", "b_e_bep",
               "dlg_bep", "dl_u_bep", "vl_bep", "sf_bep")


def submit_myjurb(domain_state, columns, surface, pbl_state, tke, *,
                  dtturbl: float, bep: Mapping, frc_urb2d, ht) -> dict:
    """Submit one MYJURB step; return the ``launch_myjurb`` output map.

    TKE and the four MYJ surface states change in place. Outputs reuse the
    MYJ scratch slots, since a domain runs MYJ or MYJURB, never both.
    MYJURB diffuses QV and CWM only, so ``rqiblten`` is a published zero.
    """
    from .array import MetalArray

    if not getattr(domain_state, "_metal_setup_state", False):
        raise TypeError("Metal MYJURB requires a Metal DomainState")
    if set(bep) != set(_BEP_INPUTS):
        raise ValueError(f"MYJURB BEP keys must be exactly {_BEP_INPUTS}")
    dz = columns["dz"]
    if not isinstance(dz, MetalArray) or dz.ndim != 3:
        raise TypeError("MYJURB dz must be a MetalArray with shape (nz, ny, nx)")
    nz, ny, nx = dz.shape
    if nz < 4 or nz > 128:
        raise ValueError("MYJURB requires 4 <= nz <= MYJ_KMAX (128)")
    shape3, shape2 = (nz, ny, nx), (ny, nx)

    def take(name, value, shape):
        if (not isinstance(value, MetalArray)
                or value.dtype != np.dtype(np.float32)
                or value.shape != shape or not value.flags.c_contiguous):
            raise TypeError(f"MYJURB {name} must be contiguous float32 {shape}")
        return value

    arrays = {f"{name}_a": take(name, columns[name], shape3)
              for name in _COLUMN_INPUTS}
    arrays["tke_a"] = take("tke", tke, shape3)
    arrays.update({f"{name}_a": take(name, surface[name], shape2)
                   for name in _SURFACE_INPUTS})
    arrays.update({f"{name}_a": take(name, pbl_state[name], shape2)
                   for name in _PBL_STATE})
    arrays.update({
        name: take(name, bep[name], (nz + 1, ny, nx) if name == "sf_bep"
                   else shape3)
        for name in _BEP_INPUTS})
    arrays["frc_urb2d"] = take("frc_urb2d", frc_urb2d, shape2)
    arrays["ht"] = take("ht", ht, shape2)

    outputs = {name: domain_state.scratch(shape3, f"myjpbl_{name}")
               for name in _COLUMN_OUTPUTS}
    outputs["pblh"] = domain_state.scratch(shape2, "myjpbl_pblh")
    outputs["mixht"] = domain_state.scratch(shape2, "myjpbl_mixht")
    outputs["kpbl"] = domain_state.scratch(
        shape2, "myjpbl_kpbl", dtype=np.int32)
    arrays.update({f"{name}_a": value for name, value in outputs.items()})

    n = ny * nx
    domain_state._array_module.context.submit_kernel(
        "myjurb_column", arrays,
        {"dtturbl": float(dtturbl), "flag_bep": 1, "idiff": 0,
         "nz": nz, "ny": ny, "nx": nx, "n": n},
        exact_policy=False, grid_size=n)

    rqiblten = domain_state.scratch(shape3, "myjpbl_rqiblten")
    rqiblten.fill(np.float32(0.0))
    outputs["rqiblten"] = rqiblten
    outputs.update(du=outputs["rublten"], dv=outputs["rvblten"],
                   dtheta=outputs["rthblten"], dqv=outputs["rqvblten"],
                   dqc=outputs["rqcblten"])
    return outputs
