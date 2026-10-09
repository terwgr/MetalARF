"""Metal composition for the SASE split step.

The individual kernels and ABI submitters live in :mod:`sase_family`; this
module preserves the source launch order while keeping the host work limited
to the small FP64 reductions used by the closure's scalar ledger.
"""

from __future__ import annotations

import math
import numpy as np

from .array import MetalArray, MetalArrayModule
from .sase_family import (
    submit_sase_dynamic_solve, submit_sase_zi_column,
    submit_sase_w_sensor_partial, submit_sase_vertical_channel,
    submit_sase_strain, submit_sase_model_stress_gov,
    submit_sase_e_hflux, submit_sase_split_tendencies,
    submit_sase_thomas_momentum, submit_sase_vertical_production,
    submit_sase_taper_weights, submit_sase_blackadar_length,
    submit_sase_split_e_update, submit_sase_thomas_scalar,
)
from woof.core.sase_math import (
    BL89_MIX_EXP, BLACKADAR_LAMBDA, C_E, C_ED, C_ES, C_KS, C_KV,
    CKS_BLEND_EXP, C_MOM_BG, C_S, CP_AIR, E_MIN, G_ACCEL, KARMAN,
    LS_COEF, N2_SCREEN, NU_BLEND_EPS, PR_RANS, PR_LES, RIB_CRIT,
    RIB_WSPD2_FLOOR, SFC_WSPD_FLOOR, SMAG_KM_CAP, _w_bound_tail,
    partition_cap, prandtl_blend,
)


def launch_sase_step_metal(u: MetalArray, v: MetalArray, w: MetalArray,
                           theta: MetalArray, e: MetalArray, *, dx: float,
                           dy: float, dz: float, delta: float, dt: float,
                           n2: MetalArray | None = None,
                           dz_col: MetalArray | None = None,
                           heat: MetalArray | None = None,
                           exclude_boundary_width: int = 0,
                           z0: float = 0.0, zdamp: float | None = None,
                           ust: MetalArray | None = None,
                           wspd_sfc: MetalArray | None = None,
                           n2_moist: MetalArray | None = None,
                           stable_dissipation: bool = False,
                           additive_dissipation: bool = False):
    """Run one Metal SASE split step in place and return its ledger."""
    if not isinstance(u, MetalArray) or u.ndim != 3:
        raise TypeError("SASE winds must be three-dimensional MetalArrays")
    context = u.buffer.context
    if any(not isinstance(a, MetalArray) or a.shape != u.shape
           or a.buffer.context is not context
           for a in (v, w, theta, e)):
        raise ValueError("SASE fields must share shape and Metal context")
    nz, ny, nx = u.shape
    if nz > 128:
        raise ValueError("SASE Metal step supports at most 128 levels")
    if (isinstance(dz, bool) or not np.isfinite(dz) or float(dz) <= 0.0):
        raise ValueError("the SASE solve spacing dz must be positive and finite")
    if dz_col is not None and (dz_col.shape != u.shape
                               or dz_col.buffer.context is not context):
        raise ValueError("dz_col must match the SASE fields and context")
    if n2 is not None and (n2.shape != u.shape
                           or n2.buffer.context is not context):
        raise ValueError("n2 must match the SASE fields and context")
    if n2_moist is not None and (n2 is None or n2_moist.shape != u.shape
                                 or n2_moist.buffer.context is not context):
        raise ValueError("n2_moist requires matching dry n2 and shape")
    if heat is None:
        heat = MetalArrayModule(context).empty(u.shape, dtype=np.float32)
        heat.fill(0.0)
    elif heat.shape != u.shape or heat.buffer.context is not context:
        raise ValueError("heat must match the SASE fields and context")
    bw = int(exclude_boundary_width)
    if bw < 0 or (bw and (2 * bw >= ny or 2 * bw >= nx)):
        raise ValueError("boundary width leaves no SASE interior")
    if wspd_sfc is not None and ust is None:
        raise ValueError("wspd_sfc requires ust")
    xp = MetalArrayModule(context)
    t_mode = 3 if dz_col is not None else 0
    thickness = dz_col
    if thickness is None:
        thickness = xp.empty(u.shape, dtype=np.float32)
        thickness.fill(float(dz))

    c_nu, f_solved = submit_sase_dynamic_solve(
        context, u, v, w, e, dx=dx, dy=dy, dz=float(dz), delta=delta,
        boundary_width=bw)
    zi_col = submit_sase_zi_column(
        context, u, v, theta, thickness, dz=float(dz), t_mode=t_mode,
        rib_crit=RIB_CRIT, wspd2_floor=RIB_WSPD2_FLOOR,
        g_accel=G_ACCEL)
    zi_host = zi_col.to_numpy()
    zi_host = zi_host[bw:-bw, bw:-bw] if bw else zi_host
    zi = float(np.mean(zi_host, dtype=np.float64))
    f_cap = partition_cap(delta, zi)
    wsums = submit_sase_w_sensor_partial(
        context, w, e, n2=n2, n2_screen=N2_SCREEN, e_min=E_MIN,
        boundary_width=bw).to_numpy().sum(axis=1, dtype=np.float64)
    n_int = nz * (ny - 2 * bw) * (nx - 2 * bw)
    count = int(round(float(wsums[3])))
    d2w_4 = (0.5 * float(wsums[2]) / count) if count else 0.0
    wsens = _w_bound_tail(d2w_4, count, n_int,
                          float(wsums[4]) / max(n_int, 1))
    f = min(float(f_solved), float(f_cap), float(wsens.f_w))
    pr_t = prandtl_blend(f)

    n2_eff = n2 if n2_moist is None else n2_moist
    kv, leps = submit_sase_vertical_channel(
        context, e, theta, thickness, dz=float(dz), t_mode=t_mode,
        f_blend=f, z0=z0, c_kv=C_KV, ls_coef=LS_COEF, e_min=E_MIN,
        g_accel=G_ACCEL, karman=KARMAN, blk_lambda=BLACKADAR_LAMBDA,
        c_ks=C_KS, cks_exp=CKS_BLEND_EXP, n2=n2_eff,
        n2_dry=n2 if n2_moist is not None else None)

    strain = submit_sase_strain(
        context, u, v, w, dx=dx, dy=dy,
        dz=float(dz) if dz_col is None else None, dz_col=dz_col)
    stress = submit_sase_model_stress_gov(
        context, e, strain, c_nu=c_nu, f_blend=f, delta=delta,
        c_s=C_S, km_cap=SMAG_KM_CAP, nu_eps=NU_BLEND_EPS, e_min=E_MIN)
    fx_e, fy_e = submit_sase_e_hflux(
        context, e, stress["km"], dx=dx, dy=dy, e_min=E_MIN)
    tendencies = submit_sase_split_tendencies(
        context, stress, u, v, w, e, stress["r"], dx=dx, dy=dy,
        e_min=E_MIN)
    partials_m = submit_sase_thomas_momentum(
        context, u, v, w, tendencies["du"], tendencies["dv"],
        tendencies["dw"], kv, thickness, dz=float(dz), t_mode=t_mode,
        dt=dt, ust=ust, wspd=wspd_sfc, sfc_floor=SFC_WSPD_FLOOR)
    pv = submit_sase_vertical_production(context, u, v, w, kv, thickness)
    taper = (submit_sase_taper_weights(context, thickness, zdamp=zdamp)
             if zdamp is not None and zdamp > 0.0 else None)
    lb = (submit_sase_blackadar_length(context, thickness, z0=z0)
          if additive_dissipation else None)
    partials_e = submit_sase_split_e_update(
        context, e, heat, theta, n2, n2_moist, kv, leps,
        tendencies["ph_e"], tendencies["ph_heat"], pv, fx_e, fy_e,
        dx=dx, dy=dy, dz=float(dz) if dz_col is None else None,
        dz_col=dz_col, dt=dt, f_blend=f, delta=delta, pr_t=pr_t,
        c_e=C_E, ls_coef=LS_COEF, c_es=C_ES, cks_exp=CKS_BLEND_EXP,
        c_ed=C_ED, g_accel=G_ACCEL, e_min=E_MIN,
        stable_dissipation=stable_dissipation, taper=taper, lb=lb)
    submit_sase_thomas_scalar(
        context, e, kv, thickness, dz=float(dz), t_mode=t_mode, dt=dt,
        kfac=2.0, floor_val=E_MIN, has_floor=True)
    sm = partials_m.to_numpy().sum(axis=1, dtype=np.float64)
    se = partials_e.to_numpy().sum(axis=1, dtype=np.float64)
    dke_expl, dke_impl, dke_sfc = (float(x) for x in sm)
    d_e = float(se[0] - se[1] - se[2])
    d_heat = float(se[3])
    d_e_src = 0.0
    if ust is not None:
        ustar = ust.to_numpy().astype(np.float64)
        t0 = (float(dz) if dz_col is None else
              float(np.mean(dz_col.to_numpy()[0], dtype=np.float64)))
        d_e_src = float(dt * np.sum(ustar ** 3 / (KARMAN * 0.5 * t0)))
    return {
        "dKE": dke_expl + dke_impl, "dE": d_e, "dHeat": d_heat,
        "residual": dke_expl + dke_impl + d_e + d_heat - dke_sfc,
        "dKE_sfc": dke_sfc, "dE_sfc_src": d_e_src,
        "sfc_conv_resid": d_e_src + dke_sfc, "c_nu": c_nu, "f": f,
        "dKE_expl": dke_expl, "dKE_impl": dke_impl,
        "f_solved": f_solved, "f_cap": f_cap, "f_w": wsens.f_w,
        "zi": zi, "w_coverage": wsens.coverage, "pr_t": pr_t,
        "kv": kv, "km_h": stress["km"],
    }
