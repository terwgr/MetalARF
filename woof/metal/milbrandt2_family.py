"""Typed Metal launch helpers for translated Milbrandt-Yau kernels."""

from __future__ import annotations

import math

import numpy as np

from .array import MetalArray


def _validate_levels(nz):
    if not 3 <= int(nz) <= 256:
        raise ValueError(f"Milbrandt-Yau requires 3 <= nz <= 256, got {nz}")


def _validate_dt(dt):
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")


def _submit(entrypoint, arrays, parameters, grid_size):
    if not isinstance(arrays, dict) or not arrays:
        raise TypeError("Milbrandt-Yau resources must be a nonempty mapping")
    resources = tuple(arrays.values())
    if any(not isinstance(value, MetalArray) for value in resources):
        raise TypeError(f"{entrypoint} requires MetalArray resources")
    context = resources[0].buffer.context
    for name, value in arrays.items():
        if (value.dtype.name != "float32" or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise TypeError(
                f"{entrypoint}.{name} requires same-context contiguous float32 storage")
    return context.submit_kernel(
        entrypoint, arrays, parameters, exact_policy=False,
        grid_size=int(grid_size))


def submit_prelim(arrays, *, nz, ny, nx, nck):
    """Submit CUDA ``milbrandt2_prelim`` over the source's flat cell order."""
    _validate_levels(nz)
    return _submit("milbrandt2_prelim", arrays,
                   {"nz": nz, "ny": ny, "nx": nx, "nck": nck},
                   int(nz) * int(ny) * int(nx))


def submit_geometry(arrays, *, nz, ny, nx):
    """Submit CUDA ``milbrandt2_geometry`` over ``(k,j,i)`` cells."""
    _validate_levels(nz)
    return _submit("milbrandt2_geometry", arrays,
                   {"nz": nz, "ny": ny, "nx": nx},
                   int(nz) * int(ny) * int(nx))


def submit_cold(arrays, *, dt, nz, ny, nx, nck):
    """Submit the source-ordered, per-cell Part 2 ice-phase kernel."""
    _validate_levels(nz)
    _validate_dt(dt)
    return _submit("milbrandt2_cold", arrays,
                   {"dt": dt, "nz": nz, "ny": ny, "nx": nx, "nck": nck},
                   int(nz) * int(ny) * int(nx))


def submit_warm(arrays, *, dt, nz, ny, nx, nck):
    """Submit the source-ordered, per-cell Part 3 warm-phase kernel."""
    _validate_levels(nz)
    _validate_dt(dt)
    return _submit("milbrandt2_warm", arrays,
                   {"dt": dt, "nz": nz, "ny": ny, "nx": nx, "nck": nck},
                   int(nz) * int(ny) * int(nx))


def submit_diagnostics(arrays, *, nz, ny, nx, nck):
    """Submit CUDA ``milbrandt2_diagnostics`` over the source grid."""
    _validate_levels(nz)
    return _submit("milbrandt2_diagnostics", arrays,
                   {"nz": nz, "ny": ny, "nx": nx, "nck": nck},
                   int(nz) * int(ny) * int(nx))


def submit_sediment_64(arrays, *, dt, nz, ny, nx, nck):
    """Submit the CUDA 64-level sedimentation specialization by column."""
    _validate_levels(nz)
    _validate_dt(dt)
    if int(nz) > 64:
        raise ValueError("milbrandt2_sediment_64 requires 3 <= nz <= 64")
    return _submit("milbrandt2_sediment_64", arrays,
                   {"dt": dt, "nz": nz, "ny": ny, "nx": nx, "nck": nck},
                   int(ny) * int(nx))


def submit_sediment_256(arrays, *, dt, nz, ny, nx, nck):
    """Submit the CUDA 256-level sedimentation specialization by column."""
    _validate_levels(nz)
    _validate_dt(dt)
    if int(nz) > 256:
        raise ValueError("milbrandt2_sediment_256 requires nz <= 256")
    if int(nz) <= 64:
        raise ValueError("milbrandt2_sediment_256 is selected only when nz > 64")
    return _submit("milbrandt2_sediment_256", arrays,
                   {"dt": dt, "nz": nz, "ny": ny, "nx": nx, "nck": nck},
                   int(ny) * int(nx))


def _validate_field_state(state, arrays, *, nz, ny, nx):
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Milbrandt-Yau fields require a Metal DomainState")
    if state.h_diabatic is None:
        raise ValueError("Milbrandt-Yau requires allocated h_diabatic")
    context = state.p.buffer.context
    volume = (nz, ny, nx)
    surface = (ny, nx)
    valid_shapes = {
        "thb": ((nz,), volume),
        "thp": (volume,),
        "p": (volume,),
        "phb": ((nz + 1,), (nz + 1, ny, nx)),
        "php": ((nz + 1, ny, nx),),
        "h_diabatic": (volume,),
    }
    for name, allowed in valid_shapes.items():
        value = getattr(state, name)
        if not isinstance(value, MetalArray):
            raise TypeError(f"Milbrandt-Yau {name} must be a MetalArray")
        if value.shape not in allowed:
            raise ValueError(
                f"Milbrandt-Yau {name} shape {value.shape} is not in {allowed}")
        if value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous:
            raise TypeError(
                f"Milbrandt-Yau {name} requires contiguous float32 storage")
        if value.buffer.context is not context:
            raise ValueError("Milbrandt-Yau resources must share one context")
    return context, volume, surface


def submit_milbrandt2_prepare(state, *, temperature, theta, pii, psfc):
    """Prepare the WRF mp=9 temperature, Exner, saved theta, and p8w fields.

    The column kernel evaluates all vertical levels itself, so it does not
    depend on MetalArray slicing or temporary height arrays.
    """
    if not isinstance(state.p, MetalArray) or state.p.ndim != 3:
        raise TypeError("Milbrandt-Yau pressure must be a 3D MetalArray")
    nz, ny, nx = map(int, state.p.shape)
    if nz < 3:
        raise ValueError("Milbrandt-Yau surface-pressure interpolation needs nz >= 3")
    context, volume, surface = _validate_field_state(
        state, {}, nz=nz, ny=ny, nx=nx)
    arrays = {"thb": state.thb, "thp": state.thp, "p": state.p,
              "phb": state.phb, "php": state.php,
              "theta": theta, "pii": pii, "temperature": temperature,
              "psfc": psfc, "h_diabatic": state.h_diabatic}
    expected = {"theta": volume, "pii": volume, "temperature": volume,
                "psfc": surface}
    for name, shape in expected.items():
        value = arrays[name]
        if not isinstance(value, MetalArray):
            raise TypeError(f"Milbrandt-Yau {name} must be a MetalArray")
        if value.shape != shape:
            raise ValueError(f"Milbrandt-Yau {name} shape {value.shape} != {shape}")
        if value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous:
            raise TypeError(
                f"Milbrandt-Yau {name} requires contiguous float32 storage")
        if value.buffer.context is not context:
            raise ValueError("Milbrandt-Yau resources must share one context")
    from woof.core import constants as c
    params = {"nz": nz, "ny": ny, "nx": nx, "nzi": nz + 1,
              "thb_3d": int(state.thb.shape == volume),
              "phb_3d": int(state.phb.ndim == 3),
              "p0": float(np.float32(c.P0)),
              "rcp": float(np.float32(c.RCP)),
              "gravity": float(np.float32(c.G))}
    return context.submit_kernel(
        "milbrandt2_prepare", arrays, params, exact_policy=False,
        grid_size=ny * nx, threads_per_group=128)


def submit_milbrandt2_finish(state, *, temperature, pii, theta, dt,
                             mp_tend_lim, no_mp_heating):
    """Reconstruct theta and apply WRF's clamped moist-physics finish."""
    if not isinstance(state.p, MetalArray) or state.p.ndim != 3:
        raise TypeError("Milbrandt-Yau pressure must be a 3D MetalArray")
    nz, ny, nx = map(int, state.p.shape)
    context, volume, _ = _validate_field_state(
        state, {}, nz=nz, ny=ny, nx=nx)
    arrays = {"temperature": temperature, "pii": pii, "theta": theta,
              "thp": state.thp, "h_diabatic": state.h_diabatic}
    for name, value in arrays.items():
        if not isinstance(value, MetalArray):
            raise TypeError(f"Milbrandt-Yau {name} must be a MetalArray")
        if value.shape != volume:
            raise ValueError(f"Milbrandt-Yau {name} shape {value.shape} != {volume}")
        if value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous:
            raise TypeError(
                f"Milbrandt-Yau {name} requires contiguous float32 storage")
        if value.buffer.context is not context:
            raise ValueError("Milbrandt-Yau resources must share one context")
    dt32 = np.float32(dt)
    limit = np.float32(np.float32(mp_tend_lim) * dt32)
    if not np.isfinite(dt32) or dt32 <= 0 or not np.isfinite(limit):
        raise ValueError("Milbrandt-Yau dt and heating limit must be finite")
    params = {"nz": nz, "ny": ny, "nx": nx,
              "no_mp_heating": int(bool(no_mp_heating)),
              "dt": float(dt32), "limit": float(limit)}
    return context.submit_kernel(
        "milbrandt2_finish", arrays, params, exact_policy=False,
        grid_size=nz * ny * nx, threads_per_group=128)
