"""Metal submission helper for ordinary calc_p_alpha diagnostics."""

from __future__ import annotations

from woof.wrf_exact import ENABLED as WRF_EXACT
from woof.wrf_exact import DIAGNOSTICS_ENABLED


def submit_calc_p_alpha(state, hypsometric_opt: int = 1, window=None):
    """Submit ordinary opt-1 or opt-2 diagnostics to Metal.

    The separate WRF exact-D variant requires glibc_flt32 log/pow helpers
    that do not yet exist in the Metal runtime. Refuse that selection rather
    than silently using Metal's different transcendental implementation.
    """
    if DIAGNOSTICS_ENABLED:
        raise NotImplementedError(
            "Metal calc_p_alpha does not yet implement the WRF exact-D "
            "glibc float32 log/pow path")
    if hypsometric_opt not in (1, 2):
        raise ValueError(f"hypsometric_opt must be 1 or 2, got {hypsometric_opt}")
    if hypsometric_opt == 2:
        if state.p_top is None:
            raise RuntimeError(
                "hypsometric_opt=2 needs state.p_top: load_base must run "
                "before the log-pressure EOS diagnostic")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal calc_p_alpha requires a Metal DomainState")

    nz, ny, nx = state.p.shape
    if window is None:
        j0, i0, nyw, nxw = 0, 0, ny, nx
    else:
        j0, i0, nyw, nxw = (int(v) for v in window)
        if (j0 < 0 or i0 < 0 or nyw < 1 or nxw < 1
                or j0 + nyw > ny or i0 + nxw > nx):
            raise ValueError(
                f"diagnostics window {window!r} does not fit ({ny}, {nx})")

    arrays = {
        "thp": state.thp, "php": state.php, "mup": state.mup,
        "thb": state.thb, "phb": state.phb,
        "dphbr": state.dphb_resid, "alb": state.alb,
        "rdnw": state.rdnw, "c1h": state.c1h, "c2h": state.c2h,
        "c3h": state.c3h, "c4h": state.c4h,
        "c3f": state.c3f, "c4f": state.c4f,
        "dc3f": state.dc3f, "dc4f": state.dc4f,
        "mub": state.mub2d,
        # CUDA passes thp as a never-read dummy when moist is false.
        "qv": state.qv if state.qv is not None else state.thp,
        "p": state.p, "al": state.al, "alt": state.alt,
    }
    parameters = {
        "p_top": float(0.0 if state.p_top is None else state.p_top),
        "hypso": int(hypsometric_opt),
        "moist": int(state.qv is not None),
        "base3d": int(state.thb.ndim == 3),
        "nz": nz, "ny": ny, "nx": nx,
        "j0": j0, "i0": i0, "nyw": nyw, "nxw": nxw,
    }
    return state._array_module.context.submit_kernel(
        "diagnostics_calc_p_alpha", arrays, parameters,
        exact_policy=WRF_EXACT, grid_size=nyw * nxw)
