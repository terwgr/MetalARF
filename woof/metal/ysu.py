"""Typed Metal launcher for the ordinary WRF YSU PBL column kernel.

The translated kernel owns one complete vertical column per GPU thread. Its
18-slot tridiagonal workspace is global scratch, interleaved in 32-lane
blocks exactly like the CUDA launcher. The scratch allocation is bounded by
the launch tile rather than the horizontal domain size.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .array import MetalArray, MetalArrayModule
from .runtime import MetalContext
from woof.wrf_exact import ENABLED as WRF_EXACT


_VOLUME_INPUTS = (
    "u", "v", "theta", "qv", "qc", "qi", "p", "p_interface",
    "exner", "dz", "rthraten",
)
_SURFACE_INPUTS = (
    "psfc", "znt", "ust", "hfx", "qfx", "wspd", "br", "psim",
    "psih", "xland", "u10", "v10",
)
_VOLUME_OUTPUTS = (
    "du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h", "exch_m",
)
_SURFACE_OUTPUTS = (
    "hpbl", "kpbl", "wstar", "delta", "topdown_radsum", "wstar3_2",
    "cloudflg",
)
_FLOAT_SURFACE_OUTPUTS = (
    "hpbl", "wstar", "delta", "topdown_radsum", "wstar3_2",
)
_KERNEL_OUTPUTS = _VOLUME_OUTPUTS + _SURFACE_OUTPUTS
_VALIDATE_FLOAT_OUTPUTS = _VOLUME_OUTPUTS + _FLOAT_SURFACE_OUTPUTS
_TPG = 32
_MAX_TILE_COLUMNS = 1024
_WORKSPACE_SLOTS = 18
_KMAX = 128
VERTICAL_LEVEL_BOUNDS = (4, _KMAX)


def _validate_array(name: str, value, *, context: MetalContext,
                    shape: tuple[int, ...], dtype=np.float32) -> None:
    if (not isinstance(value, MetalArray)
            or value.shape != shape
            or value.dtype != np.dtype(dtype)
            or not value.flags.c_contiguous):
        raise ValueError(
            f"YSU {name} must be a contiguous {np.dtype(dtype)} MetalArray "
            f"with shape {shape}")
    if value.buffer.context is not context:
        raise ValueError(f"YSU {name} belongs to a different Metal context")


def _tile_columns(ncol: int, nz: int) -> int:
    """Choose a whole-block tile with bounded global workspace storage."""
    if ncol <= 0:
        return 0
    # A hard 1024-column ceiling keeps the 18*(nz+1) work arrays independent
    # of domain dimensions. The extra block is already covered by rounding
    # up to complete 32-thread groups.
    return min(ncol, _MAX_TILE_COLUMNS)


def submit_ysu_column(
    context: MetalContext,
    u: MetalArray, v: MetalArray, theta: MetalArray,
    qv: MetalArray, qc: MetalArray, qi: MetalArray,
    p: MetalArray, p_interface: MetalArray, exner: MetalArray,
    dz: MetalArray, rthraten: MetalArray | None = None,
    *, psfc: MetalArray, znt: MetalArray, ust: MetalArray,
    hfx: MetalArray, qfx: MetalArray, wspd: MetalArray, br: MetalArray,
    psim: MetalArray, psih: MetalArray, xland: MetalArray,
    u10: MetalArray, v10: MetalArray, dt: float,
    ysu_topdown_pblmix: int = 1,
    topo: tuple[MetalArray, MetalArray] | None = None,
    u10_out: MetalArray | None = None, v10_out: MetalArray | None = None,
) -> dict[str, MetalArray]:
    """Submit ordinary (non-BEP) YSU PBL columns, optionally topo_wind's.

    All 23 fields must be contiguous float32 arrays resident in ``context``;
    the interface pressure has ``nz+1`` levels. Outputs follow the established
    ``launch_ysu`` order, with int32 ``kpbl`` and ``cloudflg``. No host
    numerical fallback, transfer, or full-domain YSU workspace is used.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal YSU supports ordinary default-policy arithmetic only; "
            "WRF-exact YSU remains unavailable")
    if not isinstance(context, MetalContext):
        raise TypeError("Metal YSU requires a MetalContext")
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("YSU theta must be a 3-D MetalArray")
    nz, ny, nx = theta.shape
    if not 4 <= nz <= _KMAX:
        raise ValueError(f"YSU requires 4 <= nz <= {_KMAX}, got {nz}")
    if (isinstance(dt, bool) or not np.isfinite(dt) or float(dt) <= 0.0):
        raise ValueError(f"YSU requires a positive finite dt, got {dt}")
    if ysu_topdown_pblmix not in (0, 1, False, True):
        raise ValueError("ysu_topdown_pblmix must be 0 or 1")

    volume_shape = (nz, ny, nx)
    surface_shape = (ny, nx)
    input_values = {
        "u": u, "v": v, "theta": theta, "qv": qv, "qc": qc,
        "qi": qi, "p": p, "exner": exner, "dz": dz,
    }
    if rthraten is None:
        rthraten = MetalArrayModule(context).zeros(volume_shape, np.float32)
    input_values["rthraten"] = rthraten
    input_values["p_interface"] = p_interface
    input_values.update({
        "psfc": psfc, "znt": znt, "ust": ust, "hfx": hfx,
        "qfx": qfx, "wspd": wspd, "br": br, "psim": psim,
        "psih": psih, "xland": xland, "u10": u10, "v10": v10,
    })
    for name in _VOLUME_INPUTS:
        shape = (nz + 1, ny, nx) if name == "p_interface" else volume_shape
        _validate_array(name, input_values[name], context=context, shape=shape)
    for name in _SURFACE_INPUTS:
        _validate_array(name, input_values[name], context=context,
                        shape=surface_shape)

    xp = MetalArrayModule(context)
    out: dict[str, MetalArray] = {}
    for name in _VOLUME_OUTPUTS:
        out[name] = xp.empty(volume_shape, dtype=np.float32)
    for name in _FLOAT_SURFACE_OUTPUTS:
        out[name] = xp.empty(surface_shape, dtype=np.float32)
    out["kpbl"] = xp.empty(surface_shape, dtype=np.int32)
    out["cloudflg"] = xp.empty(surface_shape, dtype=np.int32)

    arrays = dict(input_values)
    arrays.update(out)
    # topo_wind (bl_ysu.F90 ctopo-present arm): ctopo/ctopo2 and the
    # hill-top 10 m blend into u10_out/v10_out.  Off, the slots carry the
    # surface fields as unread placeholders.
    if topo is not None:
        for name, value in (("ctopo", topo[0]), ("ctopo2", topo[1]),
                            ("u10o", u10_out), ("v10o", v10_out)):
            _validate_array(name, value, context=context, shape=surface_shape)
        arrays.update(ctopo=topo[0], ctopo2=topo[1], u10o=u10_out,
                      v10o=v10_out)
    else:
        arrays.update(ctopo=xland, ctopo2=xland, u10o=out["hpbl"],
                      v10o=out["hpbl"])
    ncol = ny * nx
    tile = _tile_columns(ncol, nz)
    if tile:
        wskp = nz + 1
        for col0 in range(0, ncol, tile):
            tilecol = min(tile, ncol - col0)
            blocks = math.ceil(tilecol / _TPG)
            ws_count = blocks * _WORKSPACE_SLOTS * wskp * _TPG
            ws = xp.empty((ws_count,), dtype=np.float32)
            call_arrays = dict(arrays, ws=ws)
            params = {
                "ncol": int(ncol), "nx": int(nx), "ny": int(ny),
                "nz": int(nz), "wskp": int(wskp), "col0": int(col0),
                "tilecol": int(tilecol), "ws_count": int(ws_count),
                "dt": float(dt),
                "ysu_topdown_pblmix": int(bool(ysu_topdown_pblmix)),
                "topo": int(topo is not None),
            }
            context.submit_kernel(
                "ysu_column", call_arrays, params, exact_policy=False,
                grid_size=tilecol, threads_per_group=_TPG)
            # The submitted batch retains its argument buffers through queue
            # completion; dropping this host wrapper bounds lifetime to tile.
            del ws, call_arrays
    return {name: out[name] for name in _KERNEL_OUTPUTS}


def validate_ysu_outputs_metal(
    out: Mapping[str, MetalArray], context: MetalContext,
) -> str | None:
    """Return the first non-finite floating output name after one GPU scan.

    The reducer preserves the established launcher field order and reads back
    only one uint32 status word. It never copies model fields to the host.
    """
    if not isinstance(context, MetalContext):
        raise TypeError("Metal YSU validation requires a MetalContext")
    if not isinstance(out, Mapping) or set(out) != set(_KERNEL_OUTPUTS):
        raise ValueError(
            "Metal YSU validation requires the canonical launcher outputs")
    first = out["du"]
    if not isinstance(first, MetalArray) or first.ndim != 3:
        raise TypeError("YSU du must be a 3-D MetalArray")
    nz, ny, nx = first.shape
    if not 4 <= nz <= _KMAX:
        raise ValueError(f"YSU requires 4 <= nz <= {_KMAX}, got {nz}")
    volume_shape = (nz, ny, nx)
    surface_shape = (ny, nx)
    for name in _VOLUME_OUTPUTS:
        _validate_array(name, out[name], context=context,
                        shape=volume_shape)
    for name in _FLOAT_SURFACE_OUTPUTS:
        _validate_array(name, out[name], context=context,
                        shape=surface_shape)
    for name in ("kpbl", "cloudflg"):
        _validate_array(name, out[name], context=context,
                        shape=surface_shape, dtype=np.int32)

    nfloat = int(np.prod(volume_shape))
    ncol = ny * nx
    if not nfloat and not ncol:
        return None

    status = MetalArrayModule(context).empty((1,), dtype=np.uint32)
    status.fill(np.uint32(0xFFFFFFFF))
    arrays = {name: out[name] for name in _VALIDATE_FLOAT_OUTPUTS}
    arrays["status"] = status
    context.submit_kernel(
        "ysu_validate_outputs", arrays,
        {"nfloat": nfloat, "ny": int(ny), "nx": int(nx),
         "ncol": int(ncol), "nz": int(nz)},
        exact_policy=False, grid_size=max(nfloat, ncol))
    first_rank = int(status.to_numpy()[0])
    if first_rank == 0xFFFFFFFF:
        return None
    if first_rank >= len(_VALIDATE_FLOAT_OUTPUTS):
        raise RuntimeError(f"YSU reducer returned invalid field rank {first_rank}")
    return _VALIDATE_FLOAT_OUTPUTS[first_rank]
