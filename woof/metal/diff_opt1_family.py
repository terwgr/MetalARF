"""Ordinary FP32 launchers for the source-backed diff_opt=1 family."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from ..wrf_exact import ENABLED as WRF_EXACT


def _require_metal_state(state):
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal diff_opt=1 currently supports ordinary non-exact arithmetic")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal diff_opt=1 requires a Metal DomainState")


def _require_fp32(name, array, shape):
    if (not isinstance(array, MetalArray)
            or array.dtype != np.dtype(np.float32)
            or array.shape != tuple(shape)):
        raise TypeError(f"{name} must be an FP32 MetalArray with shape {tuple(shape)}")


def submit_wrf_diff_opt1_horizontal(
        state, field, km, mu, c1, c2, base, msft, msfu, msfv, tendency,
        dx, dy, *, stagger="", boundary_x=False, boundary_y=False,
        perturb=False):
    """Submit one of WRF's mass/u/v/w horizontal coordinate-flux paths.

    Arrays and the reusable tendency are supplied by the caller; this launcher
    does not allocate scratch or construct thermal references.
    """
    _require_metal_state(state)
    stagger_id = {"": 0, "x": 1, "y": 2, "z": 3}.get(stagger)
    if stagger_id is None:
        raise ValueError(f"unsupported diff_opt=1 stagger {stagger!r}")
    if not isinstance(km, MetalArray) or km.ndim != 3:
        raise TypeError("km must be a three-dimensional FP32 MetalArray")
    nz, ny, nx = km.shape
    nys, nxs = ny + int(stagger_id == 2), nx + int(stagger_id == 1)
    shape = (nz + int(stagger_id == 3), nys, nxs)
    coeff_shape = (nz + int(stagger_id == 3),)
    for name, array, expected in (
            ("field", field, shape), ("tendency", tendency, shape),
            ("km", km, (nz, ny, nx)), ("mu", mu, (ny, nx)),
            ("c1", c1, coeff_shape), ("c2", c2, coeff_shape),
            ("base", base, shape), ("msft", msft, (ny, nx)),
            ("msfu", msfu, (ny, nx + 1)), ("msfv", msfv, (ny + 1, nx))):
        _require_fp32(name, array, expected)
    arrays = {"field": field, "km": km, "mu": mu, "c1": c1, "c2": c2,
              "base": base, "msft": msft, "msfu": msfu, "msfv": msfv,
              "tendency": tendency}
    parameters = {
        "rdx": float(np.float32(1.0 / dx)),
        "rdy": float(np.float32(1.0 / dy)),
        "nz": nz, "ny": ny, "nx": nx, "stagger": stagger_id,
        "boundary_x": int(bool(boundary_x)),
        "boundary_y": int(bool(boundary_y)),
        "perturb": int(bool(perturb)),
    }
    return state._array_module.context.submit_kernel(
        "wrf_diff_opt1_horizontal", arrays, parameters, exact_policy=False,
        grid_size=(nz + int(stagger_id == 3)) * nys * nxs)


def submit_wrf_diff_opt1_km4(state, d11, d22, d12, msft, km, kh, dx, dy,
                             cs, pr, *, boundary_x=False, boundary_y=False):
    """Submit WRF's diff_opt=1 deformation-to-Km/Kh coefficient kernel."""
    _require_metal_state(state)
    if not isinstance(d11, MetalArray) or d11.ndim != 3:
        raise TypeError("d11 must be a three-dimensional FP32 MetalArray")
    shape = d11.shape
    if len(shape) != 3:
        raise TypeError("d11 must be a three-dimensional FP32 MetalArray")
    nz, ny, nx = shape
    for name, array, expected in (
            ("d11", d11, shape), ("d22", d22, shape), ("d12", d12, shape),
            ("msft", msft, (ny, nx)), ("km", km, shape), ("kh", kh, shape)):
        _require_fp32(name, array, expected)
    arrays = {"d11": d11, "d22": d22, "d12": d12,
              "msft": msft, "km": km, "kh": kh}
    parameters = {
        "dx": float(np.float32(dx)), "dy": float(np.float32(dy)),
        "cs": float(np.float32(cs)), "pr": float(np.float32(pr)),
        "nz": nz, "ny": ny, "nx": nx,
        "boundary_x": int(bool(boundary_x)),
        "boundary_y": int(bool(boundary_y)),
    }
    return state._array_module.context.submit_kernel(
        "wrf_diff_opt1_km4", arrays, parameters, exact_policy=False,
        grid_size=nz * ny * nx)
