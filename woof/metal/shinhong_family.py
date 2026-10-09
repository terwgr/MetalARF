"""Public Metal submitter for the WRF Shin-Hong (PBL option 11) column."""

from __future__ import annotations

import math

import numpy as np

from .array import MetalArray, MetalArrayModule
from .runtime import MetalContext
from woof.wrf_exact import ENABLED as WRF_EXACT

_VOLUME_INPUTS = (
    "u", "v", "theta", "qv", "qc", "qi", "p", "p_interface",
    "exner", "dz", "tke",
)
_SURFACE_INPUTS = (
    "psfc", "znt", "ust", "hfx", "qfx", "wspd", "br", "psim",
    "psih", "xland", "u10", "v10", "corf",
)
_VOLUME_OUTPUTS = (
    "du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h", "tke_out",
    "el",
)
_SURFACE_OUTPUTS = ("hpbl", "kpbl", "wstar", "delta")
_TPG = 32
_MAX_TILE_COLUMNS = 1024
_WORKSPACE_SLOTS = 50
_KMAX = 128
_VALIDATE_3D = (
    "du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h", "tke", "el",
)
_VALIDATE_2D = ("hpbl", "wstar", "delta")
_PUBLIC_OUTPUTS = (
    "du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h", "tke",
    "el", "hpbl", "kpbl", "wstar", "delta",
)
VERTICAL_LEVEL_BOUNDS = (4, _KMAX)


def _validate_array(name: str, value, context: MetalContext,
                    shape: tuple[int, ...], dtype=np.float32) -> None:
    if (not isinstance(value, MetalArray) or value.shape != shape
            or value.dtype != np.dtype(dtype)
            or not value.flags.c_contiguous):
        raise ValueError(
            f"Shin-Hong {name} must be contiguous {np.dtype(dtype)} MetalArray "
            f"with shape {shape}")
    if value.buffer.context is not context:
        raise ValueError(f"Shin-Hong {name} belongs to a different Metal context")


def submit_shinhong_column(
    context: MetalContext, u: MetalArray, v: MetalArray, theta: MetalArray,
    qv: MetalArray, qc: MetalArray, qi: MetalArray, p: MetalArray,
    p_interface: MetalArray, exner: MetalArray, dz: MetalArray,
    tke: MetalArray, *, psfc: MetalArray, znt: MetalArray, ust: MetalArray,
    hfx: MetalArray, qfx: MetalArray, wspd: MetalArray, br: MetalArray,
    psim: MetalArray, psih: MetalArray, xland: MetalArray, u10: MetalArray,
    v10: MetalArray, corf: MetalArray, dt: float, dx: float, dy: float,
    tke_diag: int = 1, topo_wind: int = 0,
) -> dict[str, MetalArray]:
    """Submit the implemented arm A column kernel for contiguous FP32 fields.

    The current port implements WRF's default ``topo_wind=0`` arm only.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal Shin-Hong supports default-policy arithmetic only; "
            "WRF-exact arithmetic remains unavailable")
    if not isinstance(context, MetalContext):
        raise TypeError("Metal Shin-Hong requires a MetalContext")
    if topo_wind not in (0, False):
        raise NotImplementedError(
            "Metal Shin-Hong currently supports only WRF topo_wind=0 (arm A)")
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("Shin-Hong theta must be a 3-D MetalArray")
    nz, ny, nx = theta.shape
    if ny <= 0 or nx <= 0:
        raise ValueError("Shin-Hong requires nonempty horizontal dimensions")
    if not 4 <= nz <= _KMAX:
        raise ValueError(f"Shin-Hong requires 4 <= nz <= {_KMAX}, got {nz}")
    for name, value in (("dt", dt), ("dx", dx), ("dy", dy)):
        if (isinstance(value, bool) or not np.isfinite(value)
                or float(value) <= 0.0):
            raise ValueError(f"Shin-Hong requires positive finite {name}, got {value}")
    if tke_diag not in (0, 1, False, True):
        raise ValueError("Shin-Hong tke_diag must be 0 or 1")

    volume_shape = (nz, ny, nx)
    surface_shape = (ny, nx)
    values = dict(zip(_VOLUME_INPUTS, (u, v, theta, qv, qc, qi, p,
                                       p_interface, exner, dz, tke)))
    for name in _VOLUME_INPUTS:
        shape = (nz + 1, ny, nx) if name == "p_interface" else volume_shape
        _validate_array(name, values[name], context, shape)
    surface_values = dict(zip(_SURFACE_INPUTS, (
        psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland, u10, v10, corf)))
    for name, value in surface_values.items():
        _validate_array(name, value, context, surface_shape)

    xp = MetalArrayModule(context)
    outputs = {name: xp.empty(volume_shape, dtype=np.float32)
               for name in _VOLUME_OUTPUTS}
    outputs.update({name: xp.empty(surface_shape,
                                   dtype=np.int32 if name == "kpbl" else np.float32)
                    for name in _SURFACE_OUTPUTS})
    arrays = {**values, **surface_values, **outputs}
    ncol = ny * nx
    tile = min(ncol, _MAX_TILE_COLUMNS)
    for col0 in range(0, ncol, tile):
        tilecol = min(tile, ncol - col0)
        blocks = math.ceil(tilecol / _TPG)
        wskp = nz + 2
        ws = xp.empty((blocks * _WORKSPACE_SLOTS * wskp * _TPG,),
                      dtype=np.float32)
        call_arrays = dict(arrays, ws=ws)
        context.submit_kernel(
            "shinhong_column", call_arrays,
            {"nz": nz, "ny": ny, "nx": nx, "col0": col0,
             "wskp": wskp, "dt": float(dt), "dx": float(dx),
             "dy": float(dy), "tke_diag": int(bool(tke_diag)),
             "ws_count": blocks * _WORKSPACE_SLOTS * wskp * _TPG},
            exact_policy=False, grid_size=tilecol, threads_per_group=_TPG)
    return {
        name: outputs["tke_out"] if name == "tke" else outputs[name]
        for name in _PUBLIC_OUTPUTS
    }


def submit_shinhong_validate_outputs(values, status: MetalArray) -> int:
    """Validate Shin-Hong floating outputs on their owning Metal context."""
    if not isinstance(status, MetalArray) or status.shape != (1,):
        raise TypeError("Shin-Hong validation status must be a MetalArray")
    if status.dtype != np.dtype(np.uint32) or not status.flags.c_contiguous:
        raise TypeError(
            "Shin-Hong validation status must be contiguous uint32")
    shape3 = tuple(values["du"].shape)
    shape2 = tuple(values["hpbl"].shape)
    if len(shape3) != 3 or len(shape2) != 2 or shape2 != shape3[1:]:
        raise ValueError("Shin-Hong validation requires matching 3-D/2-D shapes")
    context = status.buffer.context
    arrays = {}
    for name in _VALIDATE_3D:
        value = values[name]
        if (not isinstance(value, MetalArray) or value.shape != shape3
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise ValueError(
                f"Shin-Hong validation {name} must be a same-context "
                f"contiguous FP32 array with shape {shape3}")
        arrays[name] = value.view((int(value.size),))
    for name in _VALIDATE_2D:
        value = values[name]
        if (not isinstance(value, MetalArray) or value.shape != shape2
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise ValueError(
                f"Shin-Hong validation {name} must be a same-context "
                f"contiguous FP32 array with shape {shape2}")
        arrays[name] = value.view((int(value.size),))
    count3 = int(np.prod(shape3, dtype=np.int64))
    count2 = int(np.prod(shape2, dtype=np.int64))
    if max(count3, count2) > np.iinfo(np.int32).max:
        raise ValueError("Shin-Hong validation extents must fit int32")
    arrays["status"] = status
    status.fill(np.uint32(0))
    context.submit_kernel(
        "shinhong_validate_outputs", arrays,
        {"count3": count3, "count2": count2}, exact_policy=False,
        grid_size=max(count3, count2), threads_per_group=128)
    return int(status.to_numpy()[0])


def submit_shinhong_column_topo(*args, **kwargs):
    """Fail closed for the unimplemented topo-wind arm."""
    raise NotImplementedError(
        "Metal Shin-Hong topo-wind arm B is unavailable; no CPU fallback is provided")
