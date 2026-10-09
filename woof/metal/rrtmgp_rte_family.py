"""Metal launchers for the RRTMGP RTE solver and g-point folds."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


_LW_RESOURCES = {
    "tau", "lay_source", "lev_source", "sfc_source", "sfc_emis",
    "incident_flux", "part_up", "part_dn", "fz_cld_tau", "fz_cld_ssa",
    "fz_cld_asy", "fz_gpt_bands", "fz_mask", "pk_play", "pk_tlay",
    "pk_tlev", "pk_tsfc", "pk_vmr", "pk_iatm", "pk_jt", "pk_jp",
    "pk_ftemp", "pk_fpress", "pk_temp_ref", "pk_vmr_ref", "pk_flavor",
    "pk_gpoint_flavor", "pk_gpoint_bands", "pk_planck_fraction", "pk_totplnk",
}
_SW_RESOURCES = {
    "tau", "ssa", "asym", "mu0", "sfc_alb_dir", "sfc_alb_dif",
    "inc_flux", "part_up", "part_dn", "part_dir", "fz_cld_tau",
    "fz_cld_ssa", "fz_cld_asy", "fz_gpt_bands", "fz_mask",
}


def _resources(context, arrays, expected):
    if not isinstance(arrays, dict) or set(arrays) != expected:
        raise ValueError("RRTMGP RTE resource mapping does not match the signature")
    prepared = {}
    for name, value in arrays.items():
        if not isinstance(value, MetalArray) or not value.flags.c_contiguous:
            raise TypeError(f"RRTMGP RTE {name} must be contiguous MetalArray")
        if value.buffer.context is not context:
            raise ValueError("RRTMGP RTE resources must share one Metal context")
        if value.dtype == np.dtype(np.float32):
            prepared[name] = value
        elif value.dtype in (np.dtype(np.int32), np.dtype(np.uint32)):
            # Metadata is passed as a no-copy float view. Shader code reads its
            # exact stored bits with as_type<int>; there is no numeric cast.
            prepared[name] = value.view(value.shape, dtype=np.float32)
        else:
            raise NotImplementedError(
                f"RRTMGP RTE {name} dtype {value.dtype} needs a supported "
                "float32 or 32-bit integer Metal view; byte masks need a "
                "device-side float-mask preparation path")
    return prepared


def _submit(name, arrays, parameters, grid_size, *, expected,
            threads_per_group=128):
    if WRF_EXACT:
        raise NotImplementedError("Metal RRTMGP RTE does not implement WRF exact-D arithmetic")
    if not arrays:
        raise ValueError("RRTMGP RTE requires resources")
    first = next(iter(arrays.values()))
    if not isinstance(first, MetalArray):
        raise TypeError("RRTMGP RTE resources must be MetalArray")
    context = first.buffer.context
    return context.submit_kernel(
        name, _resources(context, arrays, expected), parameters,
        exact_policy=False, grid_size=int(grid_size),
        threads_per_group=int(threads_per_group))


def submit_delta_scale(tau_in, ssa_in, g_in, tau, ssa, asym):
    arrays = {"tau_in": tau_in, "ssa_in": ssa_in, "g_in": g_in,
              "tau": tau, "ssa": ssa, "asym": asym}
    if not isinstance(tau_in, MetalArray) or tau_in.size == 0:
        raise TypeError("delta-scale input must be a nonempty MetalArray")
    n = tau_in.size
    if any(not isinstance(a, MetalArray) or a.shape != tau_in.shape
           for a in arrays.values()):
        raise ValueError("delta-scale arrays must have matching flat shapes")
    flat = {name: value.view((n,)) for name, value in arrays.items()}
    return _submit("rrtmgp_delta_scale", flat, {"n": n}, n,
                   expected=set(flat))


def submit_lw_noscat(arrays, parameters, *, grid_size):
    """Submit one LW g-point tile; caller submits the ordered fold after it."""
    parameters = dict(parameters)
    if parameters.get("warp_fold", 0):
        raise NotImplementedError(
            "RRTMGP RTE warp-fold mode needs dynamic per-column threadgroup "
            "storage and simdgroup/block barriers absent from MetalContext")
    if int(parameters.get("nlay", 0)) > 128:
        raise ValueError("Metal RRTMGP LW local frame supports at most 128 layers")
    return _submit("rrtmgp_lw_noscat", arrays, parameters, grid_size,
                   expected=_LW_RESOURCES)


def submit_lw_flux_reduce(arrays, parameters, *, grid_size):
    return _submit("rrtmgp_lw_flux_reduce", arrays, dict(parameters), grid_size,
                   expected={"part_up", "part_dn", "flux_up", "flux_dn"})


def submit_sw_2stream(arrays, parameters, *, grid_size):
    parameters = dict(parameters)
    if parameters.get("warp_fold", 0):
        raise NotImplementedError(
            "RRTMGP RTE warp-fold mode needs dynamic per-column threadgroup "
            "storage and simdgroup/block barriers absent from MetalContext")
    if int(parameters.get("nlay", 0)) > 128:
        raise ValueError(
            "Metal RRTMGP SW local frame is fixed at 128 layers; CUDA specializes to nlay")
    return _submit("rrtmgp_sw_2stream", arrays, parameters, grid_size,
                   expected=_SW_RESOURCES)


def submit_sw_flux_reduce(arrays, parameters, *, grid_size):
    return _submit("rrtmgp_sw_flux_reduce", arrays, dict(parameters), grid_size,
                   expected={"part_up", "part_dn", "part_dir", "flux_up",
                             "flux_dn", "flux_dir"})


def submit_fluxes_to_radiation(lw_up, lw_dn, sw_up, sw_dn, plev, exner, *,
                               ny, nx, heating_factor):
    """Convert broadband column fluxes to WOOF radiation outputs on Metal."""
    values = {"lw_up": lw_up, "lw_dn": lw_dn, "sw_up": sw_up,
              "sw_dn": sw_dn, "plev": plev, "exner": exner}
    if any(not isinstance(value, MetalArray) for value in values.values()):
        raise TypeError("RRTMGP flux assembly requires MetalArray resources")
    context = lw_up.buffer.context
    if any(value.buffer.context is not context for value in values.values()):
        raise ValueError("RRTMGP flux resources must share one Metal context")
    if any(value.dtype != np.dtype(np.float32)
           or not value.flags.c_contiguous for value in values.values()):
        raise TypeError("RRTMGP flux resources must be contiguous float32")
    if exner.ndim != 2:
        raise ValueError("RRTMGP Exner must have shape (ncol,nlay)")
    ncol, nlay = exner.shape
    nlev = nlay + 1
    for name in ("lw_up", "lw_dn", "sw_up", "sw_dn", "plev"):
        if values[name].shape != (ncol, nlev):
            raise ValueError(
                f"RRTMGP {name} must have shape {(ncol, nlev)}")
    ny, nx = int(ny), int(nx)
    if min(ny, nx) < 1 or ny * nx != ncol:
        raise ValueError("RRTMGP surface shape must cover every column")
    tendency_shape = (nlay, ny, nx)
    surface_shape = (ny, nx)
    outputs = {
        "rthratenlw": MetalArray(
            context.buffer(max(ncol * nlay * 4, 4)), tendency_shape,
            np.float32),
        "rthratensw": MetalArray(
            context.buffer(max(ncol * nlay * 4, 4)), tendency_shape,
            np.float32),
        "swdown": MetalArray(context.buffer(max(ncol * 4, 4)), surface_shape,
                             np.float32),
        "glw": MetalArray(context.buffer(max(ncol * 4, 4)), surface_shape,
                          np.float32),
        "olr": MetalArray(context.buffer(max(ncol * 4, 4)), surface_shape,
                          np.float32),
        "gsw": MetalArray(context.buffer(max(ncol * 4, 4)), surface_shape,
                          np.float32),
    }
    arrays = {**values, **outputs}
    params = {"ncol": ncol, "nlay": nlay, "nlev": nlev,
              "ny": ny, "nx": nx, "ncell": ncol * nlay,
              "heating_factor": float(np.float32(heating_factor))}
    _submit("rrtmgp_fluxes_to_radiation", arrays, params,
            max(ncol * nlay, ncol), expected=set(arrays))
    return outputs


__all__ = ["submit_delta_scale", "submit_lw_noscat", "submit_lw_flux_reduce",
           "submit_sw_2stream", "submit_sw_flux_reduce",
           "submit_fluxes_to_radiation"]
