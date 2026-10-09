"""Metal glue for the existing NSSL runtime's preparation and finish seams."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def _submit(context, name, resources, params, n):
    if n:
        context.submit_kernel(name, resources, params, exact_policy=False,
                              grid_size=n, threads_per_group=256)


def submit_nssl2_runtime_prepare(
        state, theta, rho, pii, dz, *, p0, rcp, gravity):
    nz, ny, nx = tuple(state.p.shape)
    ncol, n = ny * nx, nz * ny * nx
    thb = state.thb
    phb = state.phb
    if thb.ndim == 1:
        thb = thb.view((nz,))
    if phb.ndim == 1:
        phb = phb.view((nz + 1,))
    arrays = (thb, state.thp, state.alt, state.p, phb, state.php,
              theta, rho, pii, dz, state.h_diabatic)
    if any(not isinstance(value, MetalArray)
           or value.dtype != np.dtype(np.float32) for value in arrays):
        raise TypeError("NSSL runtime fields must be FP32 MetalArrays")
    context = state.p.buffer.context
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError("NSSL runtime fields must share one MetalContext")
    _submit(context, "nssl2_runtime_prepare", {
        "thb": thb, "thp": state.thp, "alt": state.alt,
        "pressure": state.p, "phb": phb, "php": state.php,
        "theta": theta, "rho": rho, "pii": pii, "dz": dz,
        "h_diabatic": state.h_diabatic,
    }, {"nz": nz, "ncol": ncol, "ny": ny, "nx": nx,
        "thb_3d": int(thb.ndim == 3),
        "phb_3d": int(phb.ndim == 3), "p0": np.float32(p0),
        "rcp": np.float32(rcp), "gravity": np.float32(gravity)}, n)


def submit_nssl2_runtime_finish(
        state, theta, dt, *, limit, no_mp_heating):
    shape = tuple(theta.shape)
    if any(not isinstance(value, MetalArray) or value.shape != shape
           or value.dtype != np.dtype(np.float32)
           for value in (theta, state.h_diabatic, state.thp)):
        raise ValueError("NSSL finish fields must be matching FP32 MetalArrays")
    context = theta.buffer.context
    _submit(context, "nssl2_runtime_finish", {
        "theta": theta, "heating": state.h_diabatic,
        "theta_perturbation": state.thp,
    }, {"n": theta.size, "dt": np.float32(dt),
        "nz": shape[0], "ny": shape[1], "nx": shape[2],
        "limit": np.float32(limit),
        "no_mp_heating": int(bool(no_mp_heating))}, theta.size)


def submit_nssl2_radius_microns(cloud, ice, snow):
    shape = tuple(cloud.shape)
    if any(not isinstance(value, MetalArray) or value.shape != shape
           or value.dtype != np.dtype(np.float32)
           for value in (cloud, ice, snow)):
        raise ValueError("NSSL radius fields must be matching FP32 MetalArrays")
    context = cloud.buffer.context
    if any(value.buffer.context is not context for value in (ice, snow)):
        raise ValueError("NSSL radius fields must share one MetalContext")
    _submit(context, "nssl2_radius_microns", {
        "cloud": cloud, "ice": ice, "snow": snow,
    }, {"n": cloud.size, "nz": shape[0],
        "ny": shape[1], "nx": shape[2]}, cloud.size)


__all__ = ["submit_nssl2_radius_microns", "submit_nssl2_runtime_finish",
           "submit_nssl2_runtime_prepare"]
