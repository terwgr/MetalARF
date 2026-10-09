"""Typed Metal launcher for the WRF WSM6 single-moment column kernel."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


_VOLUME_RESOURCES = (
    "theta", "qv", "qc", "qi", "qr", "qs", "qg", "den", "pressure",
    "pii", "dz", "effc", "effi", "effs",
)
_SURFACE_RESOURCES = (
    "rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv",
    "sr",
)
_RESOURCES = _VOLUME_RESOURCES[:11] + _SURFACE_RESOURCES + _VOLUME_RESOURCES[11:]
_KMAX = 64


def submit_wsm6_column(*, theta, qv, qc, qi, qr, qs, qg, den, pressure, pii,
                       dz, rainnc, rainncv, snownc, snowncv, graupelnc,
                       graupelncv, sr, effc, effi, effs, dt, hail_opt=0):
    """Submit one CUDA-compatible thread per WSM6 column.

    All arrays must be already prepared, contiguous float32 ``MetalArray``
    instances. The caller retains WRF's field preparation and postprocessing.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal WSM6 does not implement WRF exact-D arithmetic")
    arrays = {
        "theta": theta, "qv": qv, "qc": qc, "qi": qi, "qr": qr,
        "qs": qs, "qg": qg, "den": den, "pressure": pressure,
        "pii": pii, "dz": dz, "rainnc": rainnc, "rainncv": rainncv,
        "snownc": snownc, "snowncv": snowncv, "graupelnc": graupelnc,
        "graupelncv": graupelncv, "sr": sr, "effc": effc, "effi": effi,
        "effs": effs,
    }
    if set(arrays) != set(_RESOURCES):  # Keep the ABI list auditable.
        raise AssertionError("WSM6 Metal resource declaration is inconsistent")
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("Metal WSM6 theta must be a three-dimensional MetalArray")
    nz, ny, nx = map(int, theta.shape)
    if min(nz, ny, nx) <= 0:
        raise ValueError("Metal WSM6 dimensions must be positive")
    if nz < 2:
        raise ValueError("Metal WSM6 requires nz >= 2")
    if nz > _KMAX:
        raise ValueError(f"Metal WSM6 supports nz <= {_KMAX}, got {nz}")
    if max(nz, ny, nx, ny * nx) > 0x7FFFFFFF:
        raise ValueError("Metal WSM6 dimensions must fit CUDA int indexing")
    volume_shape = (nz, ny, nx)
    surface_shape = (ny, nx)
    context = theta.buffer.context
    for name in _RESOURCES:
        array = arrays[name]
        if not isinstance(array, MetalArray):
            raise TypeError(f"Metal WSM6 {name} must be a MetalArray")
        expected_shape = surface_shape if name in _SURFACE_RESOURCES else volume_shape
        if array.shape != expected_shape:
            raise ValueError(
                f"Metal WSM6 {name} shape {array.shape} != {expected_shape}")
        if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
            raise TypeError(f"Metal WSM6 {name} requires contiguous float32 storage")
        if array.buffer.context is not context:
            raise ValueError("Metal WSM6 resources must share one context")
    if not isinstance(hail_opt, (int, np.integer)) or int(hail_opt) not in (0, 1):
        raise ValueError(f"WSM6 hail_opt must be 0 or 1, got {hail_opt!r}")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0:
        raise ValueError(f"WSM6 dt must be finite and positive, got {dt!r}")
    params = {"dt": float(dt32), "hail_opt": int(hail_opt),
              "nz": nz, "ny": ny, "nx": nx}
    return context.submit_kernel(
        "wsm6_column", arrays, params, exact_policy=False,
        grid_size=ny * nx, threads_per_group=32)


def submit_wsm6_prepare(state, *, theta, den, pii, dz, z8w):
    """Prepare WRF full-theta, dry density, Exner, and layer heights."""
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal WSM6 preparation requires a Metal DomainState")
    if state.h_diabatic is None:
        raise ValueError("Metal WSM6 requires allocated h_diabatic")
    nz, ny, nx = map(int, state.p.shape)
    arrays = {"thb": state.thb, "thp": state.thp, "p": state.p,
              "alt": state.alt, "phb": state.phb, "php": state.php,
              "theta": theta, "den": den, "pii": pii, "dz": dz,
              "z8w": z8w, "h_diabatic": state.h_diabatic}
    context = state.p.buffer.context
    for name, array in arrays.items():
        if not isinstance(array, MetalArray):
            raise TypeError(f"Metal WSM6 preparation {name} must be MetalArray")
        if array.buffer.context is not context:
            raise ValueError("Metal WSM6 preparation resources must share context")
    from woof.core import constants as c
    params = {"nz": nz, "ny": ny, "nx": nx, "nzi": nz + 1,
              "thb_3d": int(state.thb.shape == (nz, ny, nx)),
              "phb_3d": int(state.phb.shape == (nz + 1, ny, nx)),
              "p0": float(np.float32(c.P0)),
              "rcp": float(np.float32(c.RCP)),
              "gravity": float(np.float32(c.G))}
    return context.submit_kernel(
        "wsm6_prepare", arrays, params, exact_policy=False,
        grid_size=(nz + 1) * ny * nx, threads_per_group=128)


def submit_wsm6_finish(state, *, theta, dt, mp_tend_lim, no_mp_heating):
    """Apply WRF's retained, clamped microphysics heating to theta prime."""
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal WSM6 finish requires a Metal DomainState")
    if state.h_diabatic is None:
        raise ValueError("Metal WSM6 requires allocated h_diabatic")
    if not isinstance(theta, MetalArray):
        raise TypeError("Metal WSM6 theta must be a MetalArray")
    nz, ny, nx = map(int, state.p.shape)
    arrays = {"theta": theta, "thp": state.thp,
              "h_diabatic": state.h_diabatic}
    context = theta.buffer.context
    if any(not isinstance(array, MetalArray)
           or array.buffer.context is not context for array in arrays.values()):
        raise ValueError("Metal WSM6 finish resources must share one context")
    dt32 = np.float32(dt)
    limit = np.float32(np.float32(mp_tend_lim) * dt32)
    if not np.isfinite(dt32) or dt32 <= 0 or not np.isfinite(limit):
        raise ValueError(f"WSM6 dt and heating limit must be finite: {dt!r}")
    params = {"nz": nz, "ny": ny, "nx": nx,
              "dt": float(dt32),
              "limit": float(limit), "no_mp_heating": int(bool(no_mp_heating))}
    return context.submit_kernel(
        "wsm6_finish", arrays, params, exact_policy=False,
        grid_size=nz * ny * nx, threads_per_group=128)


__all__ = ["submit_wsm6_column", "submit_wsm6_finish", "submit_wsm6_prepare"]
