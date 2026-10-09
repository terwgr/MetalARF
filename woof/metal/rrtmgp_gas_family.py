"""Typed Metal launch boundaries for the regional RRTMGP gas kernels."""

from __future__ import annotations

import numpy as np

from woof.wrf_exact import ENABLED as WRF_EXACT


_INTERPOLATION = (
    "play", "tlay", "press_ref", "temp_ref", "iatm_out", "jt_out",
    "jp_out", "ftemp_out", "fpress_out",
)
_GAS_OPTICS = (
    "play", "plev", "tlay", "vmr", "iatm_meta", "jt_meta", "jp_meta",
    "ftemp_meta", "fpress_meta", "vmr_ref", "flavor", "gpoint_flavor",
    "kmajor", "kminor_lower", "kminor_upper", "minor_limits_lower",
    "minor_limits_upper", "density_lower", "density_upper",
    "complement_lower", "complement_upper", "idx_minor_lower",
    "idx_minor_upper", "idx_scaling_lower", "idx_scaling_upper",
    "kminor_start_lower", "kminor_start_upper", "minor_gpt_start_lower",
    "minor_gpt_list_lower", "minor_gpt_start_upper",
    "minor_gpt_list_upper", "rayleigh", "tau", "ssa",
)
_PLANCK = (
    "play", "tlay", "tlev", "tsfc", "vmr", "iatm_meta", "jt_meta",
    "jp_meta", "ftemp_meta", "fpress_meta", "temp_ref", "vmr_ref",
    "flavor", "gpoint_flavor", "gpoint_bands", "planck_fraction",
    "totplnk", "lay_source", "lev_source", "sfc_source",
)
_VMR_FILL = ("qv", "o3_words", "trace_index", "trace_value", "vmr")


def _resource_views(arrays, names, context=None):
    """Present float32 and raw int32 Metal buffers through the typed ABI."""
    if set(arrays) != set(names):
        raise ValueError("RRTMGP Metal resource names do not match the kernel")
    result = {}
    for name in names:
        array = arrays[name]
        if not isinstance(array, MetalArray):
            raise TypeError(f"{name} must be a MetalArray")
        if array.dtype == np.dtype(np.int32):
            array = array.view(array.shape, np.float32)
        elif array.dtype != np.dtype(np.float32):
            raise TypeError(
                f"{name} must use float32 storage or raw int32 metadata")
        if context is not None and array.buffer.context is not context:
            raise ValueError(f"{name} belongs to a different Metal context")
        result[name] = array
        context = array.buffer.context if context is None else context
    return result, context


from .array import MetalArray

_TABLE_CACHE = {}


def _metal_gas_tables(context, tables):
    """Upload the immutable packed gas tables once for this context."""
    key = (id(context), id(tables))
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    values = {}
    for name, host in tables.packed_arrays().items():
        if np.issubdtype(host.dtype, np.floating):
            packed = np.ascontiguousarray(host, dtype=np.float32)
        else:
            # The Metal kernels transport integer table members as raw words
            # in float-typed ABI buffers and recover them with rrtm_i().
            packed = np.ascontiguousarray(host, dtype=np.int32)
        array = MetalArray(context.buffer(max(int(packed.nbytes), 4)),
                           packed.shape, packed.dtype)
        if packed.size:
            array.buffer.write(packed.tobytes(), offset=array.byte_offset)
        values[name] = array
    _TABLE_CACHE[key] = values
    return values


def submit_interpolation(arrays, *, press_ref_trop, ncell, ntemp, npres,
                         reference=False):
    """Run either independent interpolation transcription, writing metadata."""
    name = ("rrtmgp_interpolation_inline_reference" if reference else
            "rrtmgp_interpolation_prepass")
    values, context = _resource_views(arrays, _INTERPOLATION)
    parameters = {"press_ref_trop": float(np.float32(press_ref_trop)),
                  "ncell": int(ncell), "ntemp": int(ntemp),
                  "npres": int(npres)}
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal RRTMGP interpolation does not implement WRF exact math")
    return context.submit_kernel(name, values, parameters, exact_policy=False,
                                 grid_size=int(ncell))


def submit_gas_optics(arrays, parameters, *, reference=False):
    """Run one result per (column, layer, g-point), without shared scratch."""
    values, context = _resource_views(arrays, _GAS_OPTICS)
    name = ("rrtmgp_gas_optics_reference" if reference else
            "rrtmgp_gas_optics")
    ncol, nlay, ngpt = (int(parameters[k]) for k in ("ncol", "nlay", "ngpt"))
    params = {k: int(v) for k, v in parameters.items()}
    params["rayleigh_len"] = int(arrays["rayleigh"].size)
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal RRTMGP gas optics does not implement WRF exact math")
    return context.submit_kernel(
        name, values, params,
        exact_policy=False, grid_size=ncol * nlay * ngpt)


def submit_tables_gas_optics(tables, arrays, parameters, *, reference=False):
    """Bind a GasTables instance and submit its gas-optics stage."""
    from .array import MetalArray

    play = arrays.get("play")
    if not isinstance(play, MetalArray):
        raise TypeError("gas-optics play must be a MetalArray")
    context = play.buffer.context
    table = _metal_gas_tables(context, tables)
    names = {
        "vmr_ref": "vmr_ref", "flavor": "flavor",
        "gpoint_flavor": "gpoint_flavor", "kmajor": "kmajor",
        "kminor_lower": "kminor_lower", "kminor_upper": "kminor_upper",
        "minor_limits_lower": "minor_limits_gpt_lower",
        "minor_limits_upper": "minor_limits_gpt_upper",
        "density_lower": "minor_scales_with_density_lower",
        "density_upper": "minor_scales_with_density_upper",
        "complement_lower": "scale_by_complement_lower",
        "complement_upper": "scale_by_complement_upper",
        "idx_minor_lower": "idx_minor_lower",
        "idx_minor_upper": "idx_minor_upper",
        "idx_scaling_lower": "idx_minor_scaling_lower",
        "idx_scaling_upper": "idx_minor_scaling_upper",
        "kminor_start_lower": "kminor_start_lower",
        "kminor_start_upper": "kminor_start_upper",
        "minor_gpt_start_lower": "minor_gpt_start_lower",
        "minor_gpt_list_lower": "minor_gpt_list_lower",
        "minor_gpt_start_upper": "minor_gpt_start_upper",
        "minor_gpt_list_upper": "minor_gpt_list_upper",
    }
    bound = dict(arrays)
    bound.update({target: table[source] for target, source in names.items()})
    if tables.rayleigh is not None:
        bound["rayleigh"] = table["rayleigh"]
    return submit_gas_optics(bound, parameters, reference=reference)


def submit_planck_sources(arrays, parameters):
    """Run the bounded per-column/g-point Planck source kernel."""
    values, context = _resource_views(arrays, _PLANCK)
    p = {k: int(v) for k, v in parameters.items()}
    ncol, nlay, ngpt = (p[k] for k in ("ncol", "nlay", "ngpt"))
    if nlay > 128:
        raise ValueError(
            "RRTMGP Planck sources require nlay <= 128 for pfrac[128]")
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal RRTMGP Planck sources do not implement WRF exact math")
    return context.submit_kernel(
        "rrtmgp_planck_sources", values, p, exact_policy=False,
        grid_size=ncol * ngpt)


def submit_gas_vmr_fill(arrays, *, h2o_scale, h2o_index, o3_index,
                        ntrace, ncells, nslot):
    """Fuse VMR population, preserving the source's binary64 ozone cast."""
    from .array import MetalArray

    if WRF_EXACT:
        raise NotImplementedError(
            "Metal RRTMGP VMR fill does not implement WRF exact math")
    if set(arrays) != set(_VMR_FILL):
        raise ValueError("RRTMGP VMR fill resource names do not match")
    o3 = arrays["o3_words"]
    if not isinstance(o3, MetalArray) or o3.dtype != np.dtype(np.float64):
        raise TypeError("o3_words must be the original float64 Metal buffer")
    ncells, nslot, ntrace = int(ncells), int(nslot), int(ntrace)
    if o3.size != ncells:
        raise ValueError("ozone interpolation must contain one value per cell")
    words = o3.view((2 * ncells,), np.float32)
    mapped = dict(arrays)
    for name, shape in (("qv", (ncells,)), ("vmr", (ncells, nslot))):
        array = mapped[name]
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or array.size != int(np.prod(shape))):
            raise ValueError(f"{name} has the wrong element count for {shape}")
        mapped[name] = array.view(shape, np.float32)
    mapped["o3_words"] = words
    values, context = _resource_views(mapped, _VMR_FILL)
    parameters = {
        "h2o_scale": float(np.float32(h2o_scale)),
        "h2o_index": int(h2o_index), "o3_index": int(o3_index),
        "ntrace": ntrace, "ncells": ncells, "nslot": nslot,
        "no3_words": 2 * ncells,
    }
    return context.submit_kernel(
        "rrtmgp_gas_vmr_fill", values, parameters, exact_policy=False,
        grid_size=ncells * nslot)


__all__ = ["submit_gas_optics", "submit_gas_vmr_fill",
           "submit_interpolation", "submit_planck_sources"]
