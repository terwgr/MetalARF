"""Typed Metal submission for the WRF Kessler warm-rain column kernel."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


_SURFACE = ("rainnc", "rainncv")
_KMAX = 256


def submit_kessler_column(*, t, qv, qc, qr, rho, pii, z, dz8w,
                          rainnc, rainncv, dt):
    """Submit one CUDA-compatible thread per column, updating fields in place.

    This is the ordinary FP32 translation of ``kessler_column``. WRF exact
    arithmetic remains unavailable for this Metal entry point.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal Kessler does not implement WRF exact-D arithmetic")
    arrays = {"t": t, "qv": qv, "qc": qc, "qr": qr, "rho": rho,
              "pii": pii, "z": z, "dz8w": dz8w, "rainnc": rainnc,
              "rainncv": rainncv}
    if not isinstance(t, MetalArray) or t.ndim != 3:
        raise TypeError("Metal Kessler t must be a three-dimensional MetalArray")
    nz, ny, nx = map(int, t.shape)
    if min(nz, ny, nx) <= 0:
        raise ValueError("Metal Kessler dimensions must be positive")
    if nz < 2:
        raise ValueError("Metal Kessler requires nz >= 2 for top rdzk spacing")
    if nz > _KMAX:
        raise ValueError(f"Metal Kessler supports nz <= {_KMAX}, got {nz}")
    volume_shape = (nz, ny, nx)
    surface_shape = (ny, nx)
    context = t.buffer.context
    for name, value in arrays.items():
        if not isinstance(value, MetalArray):
            raise TypeError(f"Metal Kessler {name} must be a MetalArray")
        expected = surface_shape if name in _SURFACE else volume_shape
        if value.shape != expected:
            raise ValueError(
                f"Metal Kessler {name} shape {value.shape} != {expected}")
        if value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous:
            raise TypeError(
                f"Metal Kessler {name} requires contiguous float32 storage")
        if value.buffer.context is not context:
            raise ValueError("Metal Kessler resources must share one context")

    dt32 = np.float32(dt)
    if not np.isfinite(dt32):
        raise ValueError(f"Metal Kessler dt must be finite, got {dt!r}")
    from woof.core import constants as c
    # Physical constants are passed as parameters for the generated ABI; keep
    # their float32 values identical to the CUDA translation-unit constants.
    params = {
        "dt": float(dt32), "nz": nz, "ny": ny, "nx": nx,
        "svp1": float(np.float32(c.SVP1)),
        "svp2": float(np.float32(c.SVP2)),
        "svp3": float(np.float32(c.SVP3)),
        "svpt0": float(np.float32(c.SVPT0)),
        "xlv": float(np.float32(c.XLV)),
        "cp": float(np.float32(c.CP)),
        "rhowater": float(np.float32(c.RHOWATER)),
        "ep2": float(np.float32(c.EP2)),
    }
    return context.submit_kernel(
        "kessler_column", arrays, params, exact_policy=False,
        grid_size=ny * nx, threads_per_group=128)


def submit_kessler_prepare(state, *, t, rho, pii, z, dz8w, z_interface):
    """Prepare WRF Kessler fields and capture pre-microphysics theta."""
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal Kessler preparation requires a Metal DomainState")
    nz, ny, nx = state.p.shape
    arrays = {
        "thb": state.thb, "thp": state.thp, "p": state.p,
        "alt": state.alt, "phb": state.phb, "php": state.php,
        "t": t, "rho": rho, "pii": pii, "z": z, "dz8w": dz8w,
        "z_interface": z_interface, "h_diabatic": state.h_diabatic,
    }
    if state.h_diabatic is None:
        raise ValueError("Metal Kessler requires allocated h_diabatic")
    context = t.buffer.context
    if any(not isinstance(value, MetalArray) for value in arrays.values()):
        raise TypeError("Metal Kessler preparation resources must be MetalArray")
    if any(value.buffer.context is not context for value in arrays.values()):
        raise ValueError("Metal Kessler preparation resources must share context")
    from woof.core import constants as c
    return context.submit_kernel(
        "kessler_prepare", arrays,
        {"nz": nz, "ny": ny, "nx": nx,
         "thb_3d": int(state.thb.ndim == 3),
         "phb_3d": int(state.phb.ndim == 3),
         "p0": float(np.float32(c.P0)),
         "rcp": float(np.float32(c.RCP)),
         "gravity": float(np.float32(c.G))},
        exact_policy=False, grid_size=ny * nx, threads_per_group=128)


def submit_kessler_finish(state, t, *, dt, mp_tend_lim,
                          no_mp_heating: bool):
    """Apply WRF's clamped latent-heating finish to theta' and h_diabatic."""
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal Kessler finish requires a Metal DomainState")
    if state.h_diabatic is None:
        raise ValueError("Metal Kessler requires allocated h_diabatic")
    nz, ny, nx = state.p.shape
    arrays = {"t": t, "thp": state.thp,
              "h_diabatic": state.h_diabatic}
    if any(not isinstance(value, MetalArray) for value in arrays.values()):
        raise TypeError("Metal Kessler finish resources must be MetalArray")
    context = t.buffer.context
    if any(value.buffer.context is not context for value in arrays.values()):
        raise ValueError("Metal Kessler finish resources must share context")
    dt32 = np.float32(dt)
    limit = np.float32(mp_tend_lim * dt)
    if not np.isfinite(dt32) or dt32 == np.float32(0.0):
        raise ValueError(f"Metal Kessler finish dt must be finite and nonzero, got {dt!r}")
    return context.submit_kernel(
        "kessler_finish", arrays,
        {"dt": float(dt32), "limit": float(limit),
         "no_mp_heating": int(bool(no_mp_heating)),
         "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=nz * ny * nx,
        threads_per_group=128)


__all__ = ["submit_kessler_column", "submit_kessler_prepare",
           "submit_kessler_finish"]
