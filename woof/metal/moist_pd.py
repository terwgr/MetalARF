"""Ordinary Metal launchers for positive-definite scalar advection."""

from __future__ import annotations

import numpy as np

from woof.wrf_exact import ADVECTION_ENABLED


def _require_state(state):
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal PD advection requires a Metal DomainState")
    if ADVECTION_ENABLED:
        raise NotImplementedError(
            "Metal PD advection does not implement WRF exact arithmetic")


def submit_pd_fluxes(q, q0, ru, rv, rw, mut, coord, dx, dy, dt,
                     fxl, fxc, fyl, fyc, fzl, fzc, *,
                     msft=None, has_msf=None, open_x=False, open_y=False,
                     vorder=3):
    """Submit the face flux decomposition into caller-owned scratch arrays."""
    _require_state(coord)
    if vorder not in (3, 5):
        raise ValueError("PD vertical advection order must be 3 or 5")
    nz, ny, nx = q.shape
    if has_msf is None:
        has_msf = msft is not None
    if msft is None:
        msft = coord.msft
    arrays = {
        "q": q, "q0": q0, "ru": ru, "rv": rv, "rw": rw, "mut": mut,
        "c1h": coord.c1h, "c2h": coord.c2h, "rdnw": coord.rdnw,
        "fnm": coord.fnm, "fnp": coord.fnp, "msft": msft,
        "fxl": fxl, "fxc": fxc, "fyl": fyl, "fyc": fyc,
        "fzl": fzl, "fzc": fzc,
    }
    params = {"dx": float(np.float32(dx)), "dy": float(np.float32(dy)),
              "dt": float(np.float32(dt)), "has_msf": int(bool(has_msf)),
              "open_x": int(bool(open_x)), "open_y": int(bool(open_y)),
              "vorder": int(vorder),
              "nz": nz, "ny": ny, "nx": nx}
    return coord._array_module.context.submit_kernel(
        "pd_fluxes", arrays, params, exact_policy=False,
        grid_size=(nz + 1) * (ny + 1) * (nx + 1))


def submit_pd_renorm_apply(q0, mu_old, fxl, fxc, fyl, fyc, fzl, fzc,
                           tend, coord, dx, dy, dt, *,
                           msft=None, has_msf=None, open_x=False, open_y=False):
    """Submit donor-cell renormalization and add the flux divergence to tend."""
    _require_state(coord)
    nz, ny, nx = q0.shape
    if has_msf is None:
        has_msf = msft is not None
    if msft is None:
        msft = coord.msft
    arrays = {
        "q0": q0, "mu_old": mu_old, "fxl": fxl, "fxc": fxc,
        "fyl": fyl, "fyc": fyc, "fzl": fzl, "fzc": fzc,
        "c1h": coord.c1h, "c2h": coord.c2h, "rdnw": coord.rdnw,
        "msft": msft, "tend": tend,
    }
    params = {"dx_inv": float(np.float32(1.0 / dx)),
              "dy_inv": float(np.float32(1.0 / dy)),
              "dt": float(np.float32(dt)), "has_msf": int(bool(has_msf)),
              "open_x": int(bool(open_x)), "open_y": int(bool(open_y)),
              "nz": nz, "ny": ny, "nx": nx}
    return coord._array_module.context.submit_kernel(
        "pd_renorm_apply", arrays, params, exact_policy=False,
        grid_size=nz * ny * nx)
