"""Device-resident RRTMGP profile and gas preparation dispatches."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_CLIMATOLOGY = {}
_OZONE = {}
_GAS_GRIDS = {}


def _context(arrays):
    values = tuple(arrays.values())
    if not values or any(not isinstance(a, MetalArray) for a in values):
        raise TypeError("RRTMGP preparation resources must be MetalArray values")
    ctx = values[0].buffer.context
    if any(a.buffer.context is not ctx for a in values):
        raise ValueError("RRTMGP preparation resources must share one context")
    return ctx


def _require_f32(array, shape, name):
    if (not isinstance(array, MetalArray)
            or array.dtype != np.dtype(np.float32)
            or array.shape != tuple(shape) or not array.flags.c_contiguous):
        raise TypeError(f"{name} must be contiguous float32 with shape {shape}")


def _upload(context, host, dtype):
    from .array import MetalArray

    value = np.ascontiguousarray(host, dtype=dtype)
    result = MetalArray(context.buffer(max(value.nbytes, 4)), value.shape,
                        value.dtype)
    if value.size:
        result.buffer.write(value.tobytes(), offset=result.byte_offset)
    return result


def _static_lw_climatology(context):
    key = id(context)
    held = _CLIMATOLOGY.get(key)
    if held is None:
        # These are immutable WRF table literals. All interpolation and
        # temperature arithmetic happens in the Metal kernel.
        from woof.core.rrtmgp import _WRF_LW_PPROF_HPA, _WRF_LW_TPROF_K
        order = np.argsort(_WRF_LW_PPROF_HPA)
        held = (
            _upload(context, _WRF_LW_PPROF_HPA[order], np.float32),
            _upload(context, _WRF_LW_TPROF_K[order], np.float32),
        )
        _CLIMATOLOGY[key] = held
    return held


def _static_ozone(context):
    key = id(context)
    held = _OZONE.get(key)
    if held is None:
        from woof.core.rrtmgp import load_trace_climatology
        table = load_trace_climatology()
        order = np.argsort(table.pressure_layer_pa)
        held = (
            _upload(context, table.pressure_layer_pa[order], np.float64),
            _upload(context, table.ozone_vmr[order], np.float32),
        )
        _OZONE[key] = held
    return held


def _static_gas_grid(context, tables):
    key = (id(context), id(tables))
    held = _GAS_GRIDS.get(key)
    if held is None:
        held = (
            _upload(context, np.asarray(tables.press_ref, np.float32), np.float32),
            _upload(context, np.asarray(tables.temp_ref, np.float32), np.float32),
        )
        _GAS_GRIDS[key] = held
    return held


def submit_pack_columns(array):
    """Pack WOOF's ``(nlay, ny, nx)`` state into bottom-to-top columns."""
    if not isinstance(array, MetalArray) or array.ndim != 3:
        raise TypeError("RRTMGP column packing requires a rank-three MetalArray")
    if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
        raise TypeError("RRTMGP columns must be contiguous float32 state")
    nlay, ny, nx = array.shape
    if min(nlay, ny, nx) < 1:
        raise ValueError("RRTMGP column packing dimensions must be positive")
    ncol = ny * nx
    context = array.buffer.context
    output = MetalArray(context.buffer(max(ncol * nlay * 4, 4)),
                        (ncol, nlay), np.float32)
    context.submit_kernel(
        "rrtmgp_prepare_pack_columns",
        {"input": array, "output": output},
        {"ncol": ncol, "nlay": nlay, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ncol * nlay)
    return output


def submit_clamp_top_interface(plev, pressure_floor):
    """Apply WRF's coefficient-floor clamp to each packed column top."""
    if (not isinstance(plev, MetalArray) or plev.ndim != 2
            or plev.dtype != np.dtype(np.float32)
            or not plev.flags.c_contiguous):
        raise TypeError("RRTMGP interface pressure must be contiguous FP32")
    ncol, nlev = plev.shape
    if ncol < 1 or nlev < 2:
        raise ValueError("RRTMGP interface pressure needs nonempty columns")
    plev.buffer.context.submit_kernel(
        "rrtmgp_prepare_clamp_top", {"plev": plev},
        {"ncol": ncol, "nlev": nlev,
         "pressure_floor": float(np.float32(pressure_floor))},
        exact_policy=False, grid_size=ncol, threads_per_group=128)


def submit_interface_temperatures(play, plev, tlay):
    """Rebuild RRTMGP's source-ordered pressure-weighted level profile."""
    if (not isinstance(play, MetalArray) or play.ndim != 2
            or play.dtype != np.dtype(np.float32)):
        raise TypeError("RRTMGP play must be a rank-two float32 MetalArray")
    ncol, nlay = play.shape
    if ncol < 1 or nlay < 2:
        raise ValueError("RRTMGP interface temperatures need nlay >= 2")
    for name, value, shape in (
            ("plev", plev, (ncol, nlay + 1)),
            ("tlay", tlay, (ncol, nlay))):
        _require_f32(value, shape, name)
    context = _context({"play": play, "plev": plev, "tlay": tlay})
    output = MetalArray(
        context.buffer(max(ncol * (nlay + 1) * 4, 4)),
        (ncol, nlay + 1), np.float32)
    context.submit_kernel(
        "rrtmgp_prepare_interface_temperatures",
        {"play": play, "plev": plev, "tlay": tlay, "tlev": output},
        {"ncol": ncol, "nlay": nlay},
        exact_policy=False, grid_size=ncol * (nlay + 1))
    return output


def submit_above_model_profile(play, plev, tlay, tlev, qv, *, kind,
                               p_top, pressure_floor=1.005183574463,
                               upper_nlay):
    """Append the RRTMGP cap without moving column data off device."""
    if kind not in ("lw", "sw"):
        raise ValueError("above-model profile kind must be lw or sw")
    if int(upper_nlay) < 0 or int(upper_nlay) > 128:
        raise ValueError("RRTMGP upper profile exceeds supported layer count")
    if play.ndim != 2:
        raise ValueError("play must have shape (ncol,nlay)")
    ncol, model_nlay = play.shape
    for name, value, shape in (
            ("play", play, (ncol, model_nlay)),
            ("plev", plev, (ncol, model_nlay + 1)),
            ("tlay", tlay, (ncol, model_nlay)),
            ("tlev", tlev, (ncol, model_nlay + 1)),
            ("qv", qv, (ncol, model_nlay))):
        _require_f32(value, shape, name)
    arrays = {"play": play, "plev": plev, "tlay": tlay,
              "tlev": tlev, "qv": qv}
    ctx = _context(arrays)
    upper_nlay = int(upper_nlay)
    if model_nlay + upper_nlay > 128:
        raise ValueError("RRTMGP cap exceeds maximum total layer count")
    total_nlay = model_nlay + upper_nlay
    out_play = MetalArray(ctx.buffer(max(ncol * total_nlay * 4, 4)),
                          (ncol, total_nlay), np.float32)
    out_plev = MetalArray(ctx.buffer(max(ncol * (total_nlay + 1) * 4, 4)),
                          (ncol, total_nlay + 1), np.float32)
    out_qv = MetalArray(ctx.buffer(max(ncol * total_nlay * 4, 4)),
                        (ncol, total_nlay), np.float32)
    temp_dtype = np.float64 if kind == "lw" else np.float32
    out_tlay = MetalArray(ctx.buffer(max(ncol * total_nlay * temp_dtype().itemsize, 8)),
                          (ncol, total_nlay), temp_dtype)
    out_tlev = MetalArray(ctx.buffer(max(ncol * (total_nlay + 1) * temp_dtype().itemsize, 8)),
                          (ncol, total_nlay + 1), temp_dtype)
    resources = {**arrays, "out_play": out_play, "out_plev": out_plev,
                 "out_tlay": out_tlay, "out_tlev": out_tlev,
                 "out_qv": out_qv}
    if kind == "lw":
        pprof, tprof = _static_lw_climatology(ctx)
        resources.update(climo_pressure=pprof, climo_temperature=tprof)
        kernel = "rrtmgp_prepare_lw_profile"
    else:
        kernel = "rrtmgp_prepare_sw_profile"
    parameters = {
        "ncol": ncol, "model_nlay": model_nlay,
        "upper_nlay": upper_nlay, "total_nlay": total_nlay,
        "pressure_floor": float(np.float32(pressure_floor)),
    }
    if kind == "lw":
        parameters.update(nclimo=60, p_top=float(np.float32(p_top)))
    ctx.submit_kernel(
        kernel, resources,
        parameters,
        exact_policy=False, grid_size=ncol * (total_nlay + 1))
    return (out_play, out_plev, out_tlay, out_tlev, out_qv)


def submit_solver_temperatures(tlay, tlev):
    """Round retained binary64 profiles to their float32 solver copies."""
    if (not isinstance(tlay, MetalArray) or not isinstance(tlev, MetalArray)
            or tlay.dtype != np.dtype(np.float64)
            or tlev.dtype != np.dtype(np.float64)
            or tlay.ndim != 2 or tlev.shape != (tlay.shape[0], tlay.shape[1] + 1)):
        raise TypeError("solver temperature inputs must be matching FP64 profiles")
    ctx = _context({"tlay": tlay, "tlev": tlev})
    out_tlay = MetalArray(ctx.buffer(max(tlay.size * 4, 4)), tlay.shape,
                          np.float32)
    out_tlev = MetalArray(ctx.buffer(max(tlev.size * 4, 4)), tlev.shape,
                          np.float32)
    ctx.submit_kernel(
        "rrtmgp_prepare_solver_temperatures",
        {"tlay": tlay, "tlev": tlev, "out_tlay": out_tlay,
         "out_tlev": out_tlev},
        {"ncol": tlay.shape[0], "nlay": tlay.shape[1]},
        exact_policy=False, grid_size=tlay.size + tlev.size)
    return out_tlay, out_tlev


def submit_clear_upper_fields(values, upper_nlay):
    """Append zero clear-sky layers to the five radiation path fields."""
    if len(values) != 5 or any(not isinstance(v, MetalArray) for v in values):
        raise TypeError("clear-upper preparation requires five Metal fields")
    ctx = _context({str(i): v for i, v in enumerate(values)})
    shape = values[0].shape
    if (len(shape) != 2 or any(v.shape != shape
                               or v.dtype != np.dtype(np.float32)
                               for v in values)):
        raise TypeError("clear-upper fields must share a float32 2-D shape")
    upper_nlay = int(upper_nlay)
    if upper_nlay < 0:
        raise ValueError("upper_nlay cannot be negative")
    if upper_nlay == 0:
        return tuple(values)
    ncol, model_nlay = shape
    nlay = model_nlay + upper_nlay
    outputs = tuple(
        MetalArray(ctx.buffer(max(ncol * nlay * 4, 4)), (ncol, nlay), np.float32)
        for _ in values)
    resources = {f"in_{i}": value for i, value in enumerate(values)}
    resources.update({f"out_{i}": value for i, value in enumerate(outputs)})
    ctx.submit_kernel(
        "rrtmgp_prepare_clear_upper_fields", resources,
        {"ncol": ncol, "model_nlay": model_nlay, "nlay": nlay},
        exact_policy=False, grid_size=ncol * nlay)
    return outputs


def submit_interpolation_metadata(tables, play, tlay):
    """Stage gas-table coordinates on device for one profile chunk."""
    if (not isinstance(play, MetalArray) or not isinstance(tlay, MetalArray)
            or play.dtype != np.dtype(np.float32)
            or tlay.dtype != np.dtype(np.float32)
            or play.shape != tlay.shape or play.ndim != 2):
        raise TypeError("RRTMGP interpolation inputs must be matching float32 profiles")
    ctx = _context({"play": play, "tlay": tlay})
    ncol, nlay = play.shape
    ncell = ncol * nlay
    press_ref, temp_ref = _static_gas_grid(ctx, tables)
    integer = tuple(MetalArray(ctx.buffer(max(ncell * 4, 4)), (ncol, nlay), np.int32)
                    for _ in range(3))
    fraction = tuple(MetalArray(ctx.buffer(max(ncell * 4, 4)), (ncol, nlay), np.float32)
                     for _ in range(2))
    from .rrtmgp_gas_family import submit_interpolation
    submit_interpolation(
        {"play": play.view((ncell,), np.float32),
         "tlay": tlay.view((ncell,), np.float32),
         "press_ref": press_ref, "temp_ref": temp_ref,
         "iatm_out": integer[0].view((ncell,), np.int32),
         "jt_out": integer[1].view((ncell,), np.int32),
         "jp_out": integer[2].view((ncell,), np.int32),
         "ftemp_out": fraction[0].view((ncell,), np.float32),
         "fpress_out": fraction[1].view((ncell,), np.float32)},
        press_ref_trop=tables.press_ref_trop, ncell=ncell,
        ntemp=tables.ntemp, npres=tables.npres)
    return integer + fraction


def submit_gas_vmr(qv, play, *, ngas, h2o_index, o3_index, trace,
                   out=None, h2o_scale=0.028964 / 0.018016):
    """Interpolate ozone in software binary64 and write the complete VMR cube."""
    if (not isinstance(play, MetalArray) or not isinstance(qv, MetalArray)
            or play.dtype != np.dtype(np.float32)
            or qv.dtype != np.dtype(np.float32)
            or play.ndim != 2 or qv.shape != play.shape):
        raise TypeError("gas VMR play/qv must be matching float32 Metal profiles")
    ctx = _context({"play": play, "qv": qv})
    ncol, nlay = play.shape
    ncells = ncol * nlay
    nslot = int(ngas) + 1
    if not (0 <= int(h2o_index) < nslot and 0 <= int(o3_index) < nslot):
        raise ValueError("gas VMR indices exceed the active gas table")
    if out is None:
        out = MetalArray(ctx.buffer(max(ncells * nslot * 4, 4)),
                         (ncol, nlay, nslot), np.float32)
    elif (not isinstance(out, MetalArray) or out.dtype != np.dtype(np.float32)
          or out.shape != (ncol, nlay, nslot)):
        raise TypeError("gas VMR output has incompatible shape or dtype")
    if out.buffer.context is not ctx:
        raise ValueError("gas VMR output belongs to a different context")
    ozone_pressure, ozone_vmr = _static_ozone(ctx)
    active_trace = tuple((int(index), float(value)) for index, value in trace)
    if any(index < 0 or index >= nslot for index, _ in active_trace):
        raise ValueError("trace-gas index exceeds active gas-table slots")
    trace_index = _upload(ctx, np.asarray([i for i, _ in active_trace], np.int32), np.int32)
    trace_value = _upload(ctx, np.asarray([v for _, v in active_trace], np.float32), np.float32)
    ctx.submit_kernel(
        "rrtmgp_prepare_gas_vmr",
        {"play": play.view((ncells,), np.float32),
         "qv": qv.view((ncells,), np.float32), "ozone_pressure": ozone_pressure,
         "ozone_vmr": ozone_vmr, "trace_index": trace_index,
         "trace_value": trace_value,
         "vmr": out.view((ncells, nslot), np.float32)},
        {"ncells": ncells, "nslot": nslot,
         "nozone": ozone_pressure.size, "ntrace": len(active_trace),
         "h2o_index": int(h2o_index), "o3_index": int(o3_index),
         "h2o_scale": float(np.float32(h2o_scale))},
        exact_policy=False, grid_size=ncells * nslot)
    return out


__all__ = ["submit_pack_columns",
           "submit_interface_temperatures",
           "submit_above_model_profile", "submit_solver_temperatures",
           "submit_clear_upper_fields", "submit_gas_vmr",
           "submit_interpolation_metadata"]
