"""The source-independent Noah arm of the urban surface overrides."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from ..wrf_exact import ENABLED as WRF_EXACT


_FLOAT_RESOURCES = (
    "state_u10", "state_v10", "state_psim", "state_psih",
    "state_gz1oz0", "state_akms", "field_u10", "field_v10",
    "field_psim", "field_psih", "field_gz1oz0", "field_akhs",
    "field_akms", "field_chs",
)
_UCM_DEVICE_PARAMS = {}
_UCM_STATUS = {}


def submit_ucm_after_lsm(context, state, params, *, lsm, fields,
                         atmosphere, dt, itimestep, solar, cfg):
    """Metal caller seam for the SLUCM canopy stage.

    The Noah-MP canopy body now has a centrally registered 63-resource
    signature. Keeping the profile check at the dispatcher prevents a Metal
    forecast from falling through to the CuPy loader or advancing only the
    post-diagnostics copy stage.
    """
    if context is None:
        raise ValueError("Metal SLUCM dispatch needs its owning context")
    if int(lsm) != 4:
        raise NotImplementedError(
            "Metal SLUCM after-LSM currently has only the Noah-MP source "
            "fragment; Noah's rural handoff needs integration")

    from woof.core.urban_ucm import (
        UCM_STATE_PLANES, _GRID_FIELDS, _jmonth, _packed)
    from .array import MetalArray, MetalArrayModule

    if not isinstance(fields.get("tsk"), MetalArray):
        raise TypeError("Metal SLUCM fields must be MetalArray resources")
    ny, nx = fields["tsk"].shape
    module = MetalArrayModule(context)

    def require(name, value, shape, dtype=np.float32):
        if (not isinstance(value, MetalArray)
                or value.buffer.context is not context
                or value.dtype != np.dtype(dtype)
                or value.shape != tuple(shape)):
            raise TypeError(
                f"SLUCM {name} must be {np.dtype(dtype)} MetalArray "
                f"{tuple(shape)} on the owning context")
        return value

    def plane0(name):
        value = atmosphere[name]
        if not isinstance(value, MetalArray) or value.ndim != 3:
            raise TypeError(f"SLUCM atmosphere {name} must be a 3-D MetalArray")
        return require(name, value.view((ny, nx), byte_offset=value.byte_offset),
                       (ny, nx))

    def interface(level):
        value = atmosphere["p_interface"]
        if not isinstance(value, MetalArray) or value.ndim != 3:
            raise TypeError("SLUCM p_interface must be a 3-D MetalArray")
        offset = value.byte_offset + level * ny * nx * value.dtype.itemsize
        return require(f"p_interface[{level}]",
                       value.view((ny, nx), byte_offset=offset), (ny, nx))

    ucm_state = _packed(params)
    cache_key = (id(context), id(ucm_state))
    cached = _UCM_DEVICE_PARAMS.get(cache_key)
    if cached is None or cached[0] is not ucm_state:
        cached = (ucm_state, {
            "switches": module.asarray(ucm_state.switches),
        })
        _UCM_DEVICE_PARAMS[cache_key] = cached

    urban = require("urban", state.urban_mask, (ny, nx), np.int32)
    arrays = {
        "urban": urban,
        "utype": require("utype", state.utype_urb2d, (ny, nx), np.int32),
    }
    forcing = np.stack((
        require("frc", fields["frc_urb2d"], (ny, nx)).to_numpy(),
        plane0("u").to_numpy(),
        plane0("v").to_numpy(),
        plane0("temperature").to_numpy(),
        plane0("qv").to_numpy(),
        interface(0).to_numpy(),
        interface(1).to_numpy(),
        plane0("dz").to_numpy(),
        require("hrang", solar.hrang, (ny, nx)).to_numpy(),
    )).reshape(-1)
    float_bundle = np.concatenate((
        forcing,
        np.asarray(ucm_state.table, dtype=np.float32).reshape(-1),
        np.asarray(ucm_state.globals_, dtype=np.float32).reshape(-1),
    )).astype(np.float32, copy=False)
    arrays["float_bundle"] = module.asarray(float_bundle)
    arrays.update({f"field_{name}": require(f"field_{name}", fields[name],
                                               (ny, nx))
                   for name in _GRID_FIELDS})
    arrays.update({f"state_{name}": require(
        f"state_{name}", state.fields[name],
        (ny, nx) if layers == 0 else (layers, ny, nx))
        for name, layers in UCM_STATE_PLANES})
    arrays.update(cached[1])
    err = _UCM_STATUS.get(id(state))
    if err is None or err[0] is not state:
        err = (state, module.zeros((1,), dtype=np.uint32))
        _UCM_STATUS[id(state)] = err
    err[1].fill(0)
    arrays["status"] = err[1]

    try:
        return context.submit_kernel(
            "ucm_noahmp_after_lsm", arrays,
            {"ny": int(ny), "nx": int(nx), "dt": float(dt),
             "n_utype": int(ucm_state.n_utype),
             "jmonth": int(_jmonth(solar)),
             "float_bundle_size": int(float_bundle.size)},
            exact_policy=False, grid_size=int(ny * nx))
    except ValueError as exc:
        if "unregistered Metal kernel entry point" in str(exc):
            raise NotImplementedError(
                "Metal SLUCM after-LSM canopy dispatch needs integration: "
                "the UCM shader source/signature fragments are not "
                "registered in the Metal ABI") from exc
        raise


def submit_ucm_noah_overrides(context, arrays):
    """Submit WRF ``ucm_overrides`` for Noah's ``noahmp=0`` branch only.

    The branch copies six UCM state planes to the corresponding surface
    fields, copies ``CHS`` to ``AKHS``, and preserves CUDA's urban mask. The
    Noah-MP branch is unsupported here because it calls binary64-backed
    ``gfk_pow`` for ``TH2`` conversion.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal urban UCM overrides currently support ordinary FP32 only")
    if not isinstance(arrays, dict):
        raise TypeError("urban UCM resources must be a mapping")
    required = {"urban", *_FLOAT_RESOURCES}
    if set(arrays) != required:
        raise ValueError("urban UCM override resources do not match their ABI")
    urban = arrays["urban"]
    if not isinstance(urban, MetalArray) or urban.ndim != 2:
        raise TypeError("urban must be a two-dimensional int32 MetalArray")
    if urban.dtype != np.dtype(np.int32):
        raise TypeError("urban must be an int32 MetalArray")
    if urban.buffer.context is not context:
        raise ValueError("urban mask belongs to another Metal context")
    ny, nx = urban.shape
    for name in _FLOAT_RESOURCES:
        array = arrays[name]
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or array.shape != (ny, nx)):
            raise TypeError(
                f"{name} must be a float32 MetalArray of shape {(ny, nx)}")
        if array.buffer.context is not context:
            raise ValueError("urban UCM arrays must share one Metal context")

    reads = ("urban", "state_u10", "state_v10", "state_psim",
             "state_psih", "state_gz1oz0", "state_akms", "field_chs")
    writes = ("field_u10", "field_v10", "field_psim", "field_psih",
              "field_gz1oz0", "field_akhs", "field_akms")

    def overlaps(left, right):
        return (left.buffer is right.buffer
                and left.byte_offset < right.byte_offset + right.nbytes
                and right.byte_offset < left.byte_offset + left.nbytes)

    for write_name in writes:
        for read_name in reads:
            if overlaps(arrays[write_name], arrays[read_name]):
                raise ValueError(
                    f"urban UCM output {write_name} aliases input {read_name}")
    for index, write_name in enumerate(writes):
        for other_name in writes[index + 1:]:
            if overlaps(arrays[write_name], arrays[other_name]):
                raise ValueError(
                    f"urban UCM outputs {write_name} and {other_name} overlap")
    return context.submit_kernel(
        "ucm_overrides_noah", arrays, {"ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx)


def submit_ucm_noahmp_overrides(context, arrays):
    """Submit Noah-MP's rural/urban blend and post-diagnostic copies."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal urban UCM overrides currently support ordinary FP32 only")
    required = {
        "urban", "frc", "fvegxy", "t2mvxy", "t2mbxy", "q2mvxy", "q2mbxy",
        "field_u10", "field_v10", "field_psim", "field_psih",
        "field_gz1oz0", "field_akhs", "field_akms", "field_t2",
        "field_th2", "field_q2", "field_psfc", "field_chs",
        "state_u10", "state_v10", "state_psim", "state_psih",
        "state_gz1oz0", "state_akms", "state_th2", "state_q2",
    }
    if not isinstance(arrays, dict) or set(arrays) != required:
        raise ValueError("Noah-MP urban UCM resources do not match their ABI")
    urban = arrays["urban"]
    if (not isinstance(urban, MetalArray) or urban.dtype != np.dtype(np.int32)
            or urban.ndim != 2 or urban.buffer.context is not context):
        raise TypeError("urban must be an int32 mask on the owning Metal context")
    ny, nx = urban.shape
    for name in required - {"urban"}:
        array = arrays[name]
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or array.shape != (ny, nx)
                or array.buffer.context is not context):
            raise TypeError(f"{name} must be a float32 {(ny, nx)} MetalArray")
    return context.submit_kernel(
        "ucm_overrides_noahmp", arrays, {"ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx)
