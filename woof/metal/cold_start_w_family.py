"""Metal launcher for WRF's cold-start vertical velocity diagnosis."""

from __future__ import annotations

import numpy as np


def submit_wrf_cold_start_w(state, mapfac_mx, mapfac_my, cfg, *,
                            periodic_x=False, periodic_y=False):
    """Submit WRF's native cold-start W fill for a Metal resident state.

    The caller supplies the directional map factors from the producing
    ``MAPFAC_MX``/``MAPFAC_MY`` fields (falling back to ``MAPFAC_M`` as the
    CUDA caller does), because they are distinct from the state's isotropic
    map-factor carrier. This launcher does not stage or approximate inputs.
    """
    from .array import MetalArray

    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("WRF cold-start W requires a Metal DomainState")
    for name, value in (("periodic_x", periodic_x),
                        ("periodic_y", periodic_y)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be the producing WRF logical value")

    nz, ny, nx = state.u.shape[0], state.w.shape[-2], state.w.shape[-1]
    if nz < 3 or state.w.shape != (nz + 1, ny, nx):
        raise ValueError(
            "WRF cold-start W requires three wind levels and one extra W interface")
    expected = {
        "u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
        "ht": (ny, nx), "znw": (nz + 1,)
    }
    arrays = {name: getattr(state, name) for name in expected}
    arrays.update({"w": state.w, "msftx": mapfac_mx, "msfty": mapfac_my})
    expected.update({"w": (nz + 1, ny, nx),
                     "msftx": (ny, nx), "msfty": (ny, nx)})
    if any(not isinstance(value, MetalArray) for value in arrays.values()):
        raise TypeError("WRF cold-start W inputs must be MetalArray values")
    for name, value in arrays.items():
        if value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous:
            raise TypeError(f"WRF cold-start W {name} requires contiguous FP32 storage")
        if value.shape != expected[name]:
            raise ValueError(
                f"WRF cold-start W {name} shape {value.shape} != {expected[name]}")

    # Keep WRF's FP32 reciprocal construction and scalar coefficient values.
    rdx = np.float32(np.float32(1.0) / np.float32(cfg.dx))
    rdy = np.float32(np.float32(1.0) / np.float32(cfg.dy))
    submission = state._array_module.context.submit_kernel(
        "wrf_cold_start_w", arrays,
        {"cf1": np.float32(state.cf1), "cf2": np.float32(state.cf2),
         "cf3": np.float32(state.cf3), "rdx": rdx, "rdy": rdy,
         "periodic_x": int(bool(periodic_x)),
         "periodic_y": int(bool(periodic_y)),
         "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx)
    if getattr(state, "w0", None) is not None:
        state.w0[...] = state.w
    return submission
