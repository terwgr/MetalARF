"""Metal launcher for the ordinary WRF slow horizontal pressure gradient."""

from __future__ import annotations

import numpy as np

from woof.core import constants as c
from woof.wrf_exact import ENABLED as WRF_EXACT


def submit_slow_pgf(state, cfg, *, cq=None):
    """Submit ``slow_pgf`` for a Metal state without enabling ``dycore.step``.

    Moist runs may pass the stage's already-prepared ``(cqu, cqv, cqw,
    use_cq)`` tuple. Building those fields itself still belongs to the
    unsupported Metal acoustic/scratch path.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal slow_pgf currently supports ordinary non-exact arithmetic")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal slow_pgf requires a Metal DomainState")
    nz, ny, nx = state.p.shape
    if nz < 3:
        raise NotImplementedError(
            "Metal slow_pgf requires nz >= 3 for WRF's surface pressure stencil")

    if cq is None:
        use_cq = bool(getattr(cfg, "moist_cq", True)
                      and getattr(state, "qv", None) is not None)
        if use_cq:
            raise NotImplementedError(
                "Metal slow_pgf moist mode needs precomputed cqu/cqv from "
                "the stage cq preparation; Metal scratch preparation is not "
                "ported")
        cqu = cqv = state.p
    else:
        cqu, cqv, _cqw, use_cq = cq

    rdx = np.float32(1.0 / cfg.dx)
    rdy = np.float32(1.0 / cfg.dy)
    half_rdx = np.float32(0.5 / cfg.dx)
    half_rdy = np.float32(0.5 / cfg.dy)
    parameters = {
        "cf1": float(state.cf1), "cf2": float(state.cf2),
        "cf3": float(state.cf3), "cfn": float(state.cfn),
        "cfn1": float(state.cfn1),
        "rdx": float(rdx), "rdy": float(rdy),
        "half_rdx": float(half_rdx), "half_rdy": float(half_rdy),
        "top_lid": int(cfg.top_lid), "use_cq": int(bool(use_cq)),
        "boundary_x": int(bool(cfg.open_x or getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
        "boundary_y": int(bool(cfg.open_y or getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
        "base3d": int(state.phb.ndim == 3),
        "nz": nz, "ny": ny, "nx": nx,
    }
    arrays = {
        "ru_t": state.ru_t, "rv_t": state.rv_t,
        "p": state.p, "pb": state.pb,
        "al": state.al, "alt": state.alt,
        "php": state.php, "phb": state.phb,
        "mup": state.mup, "mub2d": state.mub2d,
        "c1h": state.c1h, "c2h": state.c2h,
        "rdnw": state.rdnw, "fnm": state.fnm, "fnp": state.fnp,
        "cqu": cqu, "cqv": cqv,
    }
    return state._array_module.context.submit_kernel(
        "slow_pgf", arrays, parameters, exact_policy=False,
        grid_size=nz * (ny + 1) * (nx + 1))


def submit_total_theta(state):
    """Build ``thb + thp`` into reusable state-owned Metal scratch."""
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal total theta requires a Metal DomainState")
    nz, ny, nx = state.thp.shape
    out = state.scratch(state.thp.shape, "rk_total_theta")
    state._array_module.context.submit_kernel(
        "rk_total_theta",
        {"thb": state.thb, "thp": state.thp, "out": out},
        {"base3d": int(state.thb.ndim == 3),
         "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=nz * ny * nx)
    return out


def submit_total_mu(state, *, perturbation=None, scratch_slot="rk_total_mu"):
    """Build base plus a mass perturbation into reusable state-owned scratch."""
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal total mass requires a Metal DomainState")
    perturbation = state.mup if perturbation is None else perturbation
    ny, nx = perturbation.shape
    out = state.scratch((ny, nx), scratch_slot)
    state._array_module.context.submit_kernel(
        "rk_total_mu",
        {"mub2d": state.mub2d, "mup": perturbation, "out": out},
        {"ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx)
    return out


def submit_slow_buoyancy(state, cfg, *, p=None):
    """Submit the ordinary fused vertical pressure-gradient/buoyancy term.

    This caller intentionally remains independent of ``dycore.step``.  The
    kernel supports the pinned CUDA caller's dry, warm-rain, mixed-phase, and
    hail species-sum modes; absent optional species are masked so their
    source-equivalent zero addends do not require Metal scratch allocation.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal slow_buoyancy currently supports ordinary non-exact arithmetic")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal slow_buoyancy requires a Metal DomainState")
    nz, ny, nx = state.p.shape
    if nz < 2:
        raise NotImplementedError(
            "Metal slow_buoyancy requires nz >= 2 for the top moist average")
    p = state.p if p is None else p
    dummy = state.p
    qv = getattr(state, "qv", None)
    if qv is None:
        moist_mode = 0
        qc = qr = qi = qs = qg = qh = dummy
        has_qi = has_qs = has_qg = has_qh = 0
    else:
        qc = getattr(state, "qc", None)
        qr = getattr(state, "qr", None)
        if qc is None or qr is None:
            raise NotImplementedError(
                "Metal slow_buoyancy requires qv, qc, and qr in moist mode")
        qh_state = getattr(state, "qh", None)
        qi_state = getattr(state, "qi", None)
        if qh_state is not None:
            moist_mode = 3
        elif qi_state is not None:
            moist_mode = 2
        else:
            moist_mode = 1
        if moist_mode == 3 and qi_state is None:
            raise NotImplementedError(
                "Metal slow_buoyancy hail mode requires qi")
        qi = qi_state if qi_state is not None else dummy
        qs_state = getattr(state, "qs", None)
        qg_state = getattr(state, "qg", None)
        if (qs_state is None) != (qg_state is None):
            raise NotImplementedError(
                "Metal slow_buoyancy requires qs and qg together when "
                "either field is present")
        qs = qs_state if qs_state is not None else dummy
        qg = qg_state if qg_state is not None else dummy
        qh = qh_state if qh_state is not None else dummy
        has_qi = int(qi_state is not None)
        has_qs = int(qs_state is not None)
        has_qg = int(qg_state is not None)
        has_qh = int(qh_state is not None)
        if moist_mode >= 2 and qi_state is None:
            raise NotImplementedError(
                "Metal slow_buoyancy mixed-phase mode requires qi")

    arrays = {
        "rw_t": state.rw_t, "p": p, "pb": state.pb,
        "mup": state.mup, "mub2d": state.mub2d,
        "qv": qv if qv is not None else dummy,
        "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg, "qh": qh,
        "rdn": state.rdn, "rdnw": state.rdnw,
        "c1f": state.c1f, "c2f": state.c2f, "msft": state.msft,
    }
    parameters = {
        "gravity": 9.81,
        "moist_mode": moist_mode,
        "has_qi": has_qi, "has_qs": has_qs,
        "has_qg": has_qg, "has_qh": has_qh,
        "has_msf": int(bool(state.has_msf)),
        "base3d": int(state.phb.ndim == 3),
        "nz": nz, "ny": ny, "nx": nx,
    }
    return state._array_module.context.submit_kernel(
        "slow_buoyancy", arrays, parameters, exact_policy=False,
        grid_size=nz * ny * nx)


def submit_slow_geopotential(state, cfg, *, ww=None, mux=None, muy=None,
                             mode="fused", add_vertical=False):
    """Submit one ordinary ``rhs_ph`` geopotential tendency family kernel.

    ``mode`` selects the fused face-mass reconstruction, supplied-mass
    compatibility caller, or the vertical-only partial-state failure path.
    The WRF exact profiles have distinct expression trees and remain refused.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal slow_geopotential supports ordinary non-exact arithmetic")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal slow_geopotential requires a Metal DomainState")
    if mode not in ("fused", "faces", "vertical"):
        raise ValueError(f"unknown slow_geopotential mode {mode!r}")
    nz, ny, nx = state.p.shape
    if mode != "vertical":
        if cfg is None:
            raise TypeError("geopotential horizontal modes require RunConfig")
        if cfg.h_sca_adv_order not in (2, 5):
            raise ValueError(
                f"h_sca_adv_order must be 2 or 5, got {cfg.h_sca_adv_order}")
        if cfg.h_sca_adv_order == 5:
            if nx < 7 or ny < 7:
                raise ValueError(
                    f"h_sca_adv_order=5 needs nx, ny >= 7, got {nx} x {ny}")
    dummy = state.p
    if ww is None:
        ww = dummy
    if mux is None:
        mux = state.msfu
    if muy is None:
        muy = state.msfv
    if mode == "vertical":
        order = boundary_x = boundary_y = specified = 0
        quarter_rdx = quarter_rdy = cfn = cfn1 = 0.0
        add_vertical = True
    else:
        order = int(cfg.h_sca_adv_order)
        boundary_x = int(bool(cfg.open_x or getattr(cfg, "specified", False)
                              or getattr(cfg, "nested", False)))
        boundary_y = int(bool(cfg.open_y or getattr(cfg, "specified", False)
                              or getattr(cfg, "nested", False)))
        specified = int(bool(getattr(cfg, "specified", False)
                             or getattr(cfg, "nested", False)))
        quarter_rdx = 0.25 / cfg.dx
        quarter_rdy = 0.25 / cfg.dy
        cfn, cfn1 = float(state.cfn), float(state.cfn1)
    arrays = {
        "rph_t": state.rph_t, "ww": ww, "w": state.w,
        "u": state.u, "v": state.v, "php": state.php, "phb": state.phb,
        "mup": state.mup, "mub2d": state.mub2d, "rdnw": state.rdnw,
        "fnm": state.fnm, "fnp": state.fnp, "c1f": state.c1f,
        "c2f": state.c2f, "msft": state.msft, "msfu": state.msfu,
        "msfv": state.msfv, "mux": mux, "muy": muy,
    }
    parameters = {
        "mode": {"fused": 0, "faces": 1, "vertical": 2}[mode],
        "order": order, "add_vertical": int(bool(add_vertical)),
        "has_msf": int(bool(state.has_msf)),
        "boundary_x": boundary_x, "boundary_y": boundary_y,
        "open_x": int(bool(cfg.open_x)) if cfg is not None else 0,
        "open_y": int(bool(cfg.open_y)) if cfg is not None else 0,
        "specified": specified, "base3d": int(state.phb.ndim == 3),
        "nz": nz, "ny": ny, "nx": nx,
        "quarter_rdx": quarter_rdx, "quarter_rdy": quarter_rdy,
        "cfn": cfn, "cfn1": cfn1,
    }
    return state._array_module.context.submit_kernel(
        "slow_geopotential", arrays, parameters, exact_policy=False,
        grid_size=nz * ny * nx)


def submit_surface_w(state, cfg):
    """Submit the kinematic lower-boundary update for an ordinary state."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal surface-w currently supports ordinary arithmetic only")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal surface-w requires a Metal DomainState")
    ny, nx = state.ht.shape
    parameters = {
        "cf1": float(state.cf1), "cf2": float(state.cf2),
        "cf3": float(state.cf3),
        "half_rdx": float(np.float32(0.5 / cfg.dx)),
        "half_rdy": float(np.float32(0.5 / cfg.dy)),
        "has_msf": int(bool(state.has_msf)),
        "boundary_x": int(bool(cfg.open_x or getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
        "boundary_y": int(bool(cfg.open_y or getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
        "nz": int(state.u.shape[0]),
        "ny": ny, "nx": nx,
    }
    arrays = {"u": state.u, "v": state.v, "ht": state.ht,
              "msft": state.msft, "w": state.w}
    return state._array_module.context.submit_kernel(
        "set_surface_w", arrays, parameters, exact_policy=False,
        grid_size=ny * nx)


def submit_small_step_init(state, cfg):
    """Initialize dry WRF acoustic perturbations for one RK stage."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal small-step initialization supports ordinary arithmetic only")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal small-step initialization requires Metal state")
    nz, ny, nx = state.p.shape
    context = state._array_module.context
    common = {"nz": nz, "ny": ny, "nx": nx}
    uv_arrays = {
        "u_pp": state.u_pp, "v_pp": state.v_pp,
        "u0": state.u0, "v0": state.v0, "u": state.u, "v": state.v,
        "mup0": state.mup0, "mup": state.mup, "mub2d": state.mub2d,
        "c1h": state.c1h, "c2h": state.c2h,
        "msfu": state.msfu, "msfv": state.msfv,
    }
    uv_params = {
        "has_msf": int(bool(state.has_msf)),
        "boundary_x": int(bool(cfg.open_x or getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
        "boundary_y": int(bool(cfg.open_y or getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
        **common,
    }
    context.submit_kernel("small_step_init_uv", uv_arrays, uv_params,
                          exact_policy=False,
                          grid_size=nz*(ny+1)*(nx+1))
    column_arrays = {
        "w_pp": state.w_pp, "th_pp": state.th_pp, "ph_pp": state.ph_pp,
        "mu_pp": state.mu_pp, "al_pp": state.al_pp,
        "p_pp": state.p_pp, "p_pp_old": state.p_pp_old,
        "w0": state.w0, "w": state.w,
        "thp0": state.thp0, "thp": state.thp,
        "php0": state.php0, "php": state.php,
        "p": state.p, "alt": state.alt,
        "mup0": state.mup0, "mup": state.mup, "mub2d": state.mub2d,
        "thb": state.thb, "c1h": state.c1h, "c2h": state.c2h,
        "c1f": state.c1f, "c2f": state.c2f,
        "rdnw": state.rdnw, "msft": state.msft,
    }
    column_params = {
        "has_msf": int(bool(state.has_msf)),
        "base3d": int(state.thb.ndim == 3),
        "gamma": float(np.float32(c.GAMMA)),
        **common,
    }
    context.submit_kernel("small_step_init_column", column_arrays,
                          column_params, exact_policy=False,
                          grid_size=ny*nx)


def submit_small_step_finish(state, cfg, hdiab_dt=0.0):
    """Fold dry WRF acoustic perturbations back into the RK stage fields."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal small-step finalization supports ordinary arithmetic only")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal small-step finalization requires Metal state")
    nz, ny, nx = state.p.shape
    context = state._array_module.context
    common = {"nz": nz, "ny": ny, "nx": nx}
    uv_arrays = {
        "u": state.u, "v": state.v,
        "u_pp": state.u_pp, "v_pp": state.v_pp,
        "mup": state.mup, "mu_pp": state.mu_pp, "mub2d": state.mub2d,
        "c1h": state.c1h, "c2h": state.c2h,
        "msfu": state.msfu, "msfv": state.msfv,
    }
    uv_params = {
        "has_msf": int(bool(state.has_msf)),
        "boundary_x": int(bool(cfg.open_x or getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
        "boundary_y": int(bool(cfg.open_y or getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
        **common,
    }
    context.submit_kernel("small_step_finish_uv", uv_arrays, uv_params,
                          exact_policy=False,
                          grid_size=nz*(ny+1)*(nx+1))
    h_diabatic = (state.h_diabatic if hdiab_dt
                  else state.p)
    column_arrays = {
        "w": state.w, "thp": state.thp, "php": state.php,
        "mup": state.mup, "w_pp": state.w_pp, "th_pp": state.th_pp,
        "ph_pp": state.ph_pp, "mu_pp": state.mu_pp,
        "mub2d": state.mub2d, "thb": state.thb,
        "c1h": state.c1h, "c2h": state.c2h,
        "c1f": state.c1f, "c2f": state.c2f,
        "msft": state.msft, "h_diabatic": h_diabatic,
    }
    column_params = {
        "hdiab_dt": float(np.float32(hdiab_dt)),
        "remove_hdiab": int(bool(hdiab_dt)),
        "has_msf": int(bool(state.has_msf)),
        "base3d": int(state.thb.ndim == 3),
        **common,
    }
    context.submit_kernel("small_step_finish_column", column_arrays,
                          column_params, exact_policy=False,
                          grid_size=ny*nx)


def submit_apply_emdiv(state, cfg, mudf, mu_prev=None):
    """Apply WRF's dry external-mode filter before an acoustic substep."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal emdiv currently supports ordinary arithmetic only")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal emdiv requires a Metal DomainState")
    nz, ny, nx = state.p.shape
    save_mu = mu_prev is not None
    arrays = {
        "u_pp": state.u_pp, "v_pp": state.v_pp, "mudf": mudf,
        "mu_pp": state.mu_pp, "mu_prev": mu_prev if save_mu else mudf,
        "c1h": state.c1h, "msfu": state.msfu, "msfv": state.msfv,
    }
    params = {
        "xscale": float(np.float32(-cfg.emdiv * cfg.dx)),
        "yscale": float(np.float32(-cfg.emdiv * cfg.dy)),
        "has_msf": int(bool(state.has_msf)),
        "boundary_x": int(bool(cfg.open_x or getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
        "boundary_y": int(bool(cfg.open_y or getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
        "specified": int(bool(getattr(cfg, "specified", False)
                              or getattr(cfg, "nested", False))),
        "spec_zone": int(getattr(cfg, "spec_zone", 0)),
        "save_mu": int(save_mu), "nz": nz, "ny": ny, "nx": nx,
    }
    return state._array_module.context.submit_kernel(
        "apply_emdiv", arrays, params, exact_policy=False,
        grid_size=nz*(ny+1)*(nx+1))


def submit_update_mudf(state, cfg, mudf, mu_prev, dtau):
    """Apply the ordinary external-mode ``mudf`` recurrence and edge mask."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal mudf recurrence currently supports ordinary arithmetic only")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal mudf recurrence requires a Metal DomainState")
    ny, nx = state.mup.shape
    params = {
        "dtau": float(np.float32(dtau)),
        "boundary_x": int(bool(getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
        "boundary_y": int(bool(getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
        "width": int(getattr(cfg, "spec_zone", 0)), "ny": ny, "nx": nx,
    }
    arrays = {"mudf": mudf, "mu_pp": state.mu_pp, "mu_prev": mu_prev}
    return state._array_module.context.submit_kernel(
        "update_mudf", arrays, params, exact_policy=False,
        grid_size=ny*nx)
