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
            threads_per_group=128, exact_policy=False):
    if WRF_EXACT and not exact_policy:
        raise NotImplementedError("Metal RRTMGP RTE does not implement WRF exact-D arithmetic")
    if not arrays:
        raise ValueError("RRTMGP RTE requires resources")
    first = next(iter(arrays.values()))
    if not isinstance(first, MetalArray):
        raise TypeError("RRTMGP RTE resources must be MetalArray")
    context = first.buffer.context
    return context.submit_kernel(
        name, _resources(context, arrays, expected), parameters,
        exact_policy=exact_policy, grid_size=int(grid_size),
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
    name = "rrtmgp_exact_delta_scale" if WRF_EXACT else "rrtmgp_delta_scale"
    return _submit(name, flat, {"n": n}, n,
                   expected=set(flat), exact_policy=WRF_EXACT)


def submit_lw_noscat(arrays, parameters, *, grid_size):
    """Submit one LW g-point tile; caller submits the ordered fold after it."""
    parameters = dict(parameters)
    if parameters.get("warp_fold", 0):
        raise NotImplementedError(
            "RRTMGP RTE warp-fold mode needs dynamic per-column threadgroup "
            "storage and simdgroup/block barriers absent from MetalContext")
    if int(parameters.get("nlay", 0)) > 128:
        raise ValueError("Metal RRTMGP LW local frame supports at most 128 layers")
    if WRF_EXACT:
        return _submit("rrtmgp_exact_lw_noscat", arrays, parameters, grid_size,
                       expected=_LW_RESOURCES | {"pk_plev"}, exact_policy=True)
    return _submit("rrtmgp_lw_noscat", arrays, parameters, grid_size,
                   expected=_LW_RESOURCES)


def submit_lw_flux_reduce(arrays, parameters, *, grid_size):
    name = "rrtmgp_exact_lw_flux_reduce" if WRF_EXACT else "rrtmgp_lw_flux_reduce"
    return _submit(name, arrays, dict(parameters), grid_size,
                   expected={"part_up", "part_dn", "flux_up", "flux_dn"},
                   exact_policy=WRF_EXACT)



_SW_WORK_RESOURCES = {"work_optics", "work_direct", "work_adding"}
_SW_PART_RESOURCES = {"part_up", "part_dn", "part_dir"}


class SW2StreamWorkspace:
    """Reusable column/g-point frame and partials for the tall SW route.

    The caller retains input/output profiles. This additional storage uses
    at most 1/16 of the device's recommended working set and fits every
    individual buffer and signed shader index. A single frame is reused
    across all column slabs, g-point tiles, and paired clear-sky solves.
    """

    def __init__(self, context, ncol, nlay, ngpt, *, gpt_tile=None):
        ncol, nlay, ngpt = int(ncol), int(nlay), int(ngpt)
        if min(ncol, nlay, ngpt) < 1:
            raise ValueError("SW workspace needs positive column/layer/g-point counts")
        index_limit = 0x7fffffff
        if nlay * ngpt > index_limit or nlay + 1 > index_limit:
            raise ValueError("SW column exceeds signed int32 shader indexing")
        working_set = int(context.recommended_working_set_size)
        max_buffer = int(context.max_buffer_length)
        if working_set < 1 or max_buffer < 1:
            raise ValueError("SW workspace needs the device's allocation limits")
        nlev = nlay + 1
        # Four optics, one direct, two adding, and three partial-flux words.
        bytes_per_point = 4 * (4 * nlay + 6 * nlev)
        budget = working_set // 16
        points = min(budget // bytes_per_point,
                     max_buffer // (16 * nlay),
                     max_buffer // (8 * nlev),
                     max_buffer // (4 * nlev))
        if points < 1:
            raise MemoryError(
                "SW workspace cannot fit one column/g-point in the device budget")
        requested_tile = min(32, ngpt) if gpt_tile is None else int(gpt_tile)
        if not 1 <= requested_tile <= ngpt:
            raise ValueError("SW workspace tile must cover 1 through ngpt g-points")
        tile = min(requested_tile, points)
        indexed_per_column = max(nlay * ngpt, 4 * tile * nlay,
                                 2 * tile * nlev, tile * nlev)
        columns = min(ncol, points // tile,
                      index_limit // indexed_per_column)
        if columns < 1:
            raise ValueError("SW workspace exceeds signed int32 shader indexing")
        self.context, self.nlay, self.ngpt = context, nlay, ngpt
        self.columns, self.gpt_tile = columns, tile
        self.budget_bytes = budget
        self.allocated_bytes = columns * tile * bytes_per_point
        shapes = {
            "work_optics": (columns, tile, nlay, 4),
            "work_direct": (columns, tile, nlev),
            "work_adding": (columns, tile, nlev, 2),
            **{name: (tile, nlev, columns) for name in _SW_PART_RESOURCES},
        }
        self._arrays = {name: MetalArray(
            context.buffer(int(np.prod(shape)) * 4), shape, np.float32)
            for name, shape in shapes.items()}

    def views(self, ncol):
        ncol = int(ncol)
        if not 1 <= ncol <= self.columns:
            raise ValueError("SW workspace view exceeds its column capacity")
        nlay, tile, nlev = self.nlay, self.gpt_tile, self.nlay + 1
        shapes = {
            "work_optics": (ncol, tile, nlay, 4),
            "work_direct": (ncol, tile, nlev),
            "work_adding": (ncol, tile, nlev, 2),
            **{name: (tile, nlev, ncol) for name in _SW_PART_RESOURCES},
        }
        return {name: self._arrays[name].view(shape)
                for name, shape in shapes.items()}


def solve_sw_workspace(arrays, parameters, outputs, *, workspace=None):
    """Run the actual SW solver through budgeted contiguous column slabs."""
    parameters = dict(parameters)
    ncol, nlay, ngpt = (int(parameters[name])
                       for name in ("ncol", "nlay", "ngpt"))
    context = arrays["tau"].buffer.context
    if set(arrays) != _SW_RESOURCES - _SW_PART_RESOURCES:
        raise ValueError("SW workspace input mapping does not match the solver")
    if workspace is None:
        workspace = SW2StreamWorkspace(context, ncol, nlay, ngpt)
    if (workspace.context is not context or workspace.nlay != nlay
            or workspace.ngpt != ngpt):
        raise ValueError("SW workspace must match the solver context and dimensions")
    nlev, tile_width = nlay + 1, workspace.gpt_tile
    for start in range(0, ncol, workspace.columns):
        stop = min(ncol, start + workspace.columns)
        count = stop - start
        # Every profile is column-contiguous. The band map is global;
        # absent fused-cloud pointers are full-profile valid dummies.
        inputs = {name: (value if name == "fz_gpt_bands" and value.ndim == 1
                         else value[start:stop])
                  for name, value in arrays.items()}
        scratch = workspace.views(count)
        destinations = tuple(value[start:stop] for value in outputs)
        for gpt0 in range(0, ngpt, tile_width):
            tile = min(tile_width, ngpt - gpt0)
            submit_sw_2stream(
                {**inputs, **scratch},
                {**parameters, "ncol": count, "gpt0": gpt0,
                 "gpt_tile": tile_width}, grid_size=count * tile_width)
            submit_sw_flux_reduce(
                {**{name: scratch[name].view((tile, nlev, count))
                    for name in _SW_PART_RESOURCES},
                 **dict(zip(("flux_up", "flux_dn", "flux_dir"), destinations))},
                {"ncol": count, "nlev": nlev, "gpt0": gpt0,
                 "ntile": tile}, grid_size=count * nlev)
    return outputs


def submit_sw_2stream(arrays, parameters, *, grid_size):
    parameters = dict(parameters)
    if parameters.get("warp_fold", 0):
        raise NotImplementedError(
            "RRTMGP RTE warp-fold mode needs dynamic per-column threadgroup "
            "storage and simdgroup/block barriers absent from MetalContext")
    use_workspace = bool(_SW_WORK_RESOURCES.intersection(arrays))
    if (WRF_EXACT or int(parameters.get("nlay", 0)) > 128) and not use_workspace:
        context = arrays["tau"].buffer.context
        frame = SW2StreamWorkspace(
            context, parameters["ncol"], parameters["nlay"], parameters["ngpt"],
            gpt_tile=parameters["gpt_tile"])
        if (frame.columns < int(parameters["ncol"])
                or frame.gpt_tile != int(parameters["gpt_tile"])):
            raise MemoryError(
                "SW tile exceeds the device workspace budget; use the column-slab solver")
        arrays = {**arrays, **{name: value for name, value
                             in frame.views(parameters["ncol"]).items()
                             if name in _SW_WORK_RESOURCES}}
        use_workspace = True
    if use_workspace:
        name = ("rrtmgp_exact_sw_2stream_workspace" if WRF_EXACT
                else "rrtmgp_sw_2stream_workspace")
        return _submit(name, arrays, parameters, grid_size,
                       expected=_SW_RESOURCES | _SW_WORK_RESOURCES,
                       exact_policy=WRF_EXACT)
    return _submit("rrtmgp_sw_2stream", arrays, parameters, grid_size,
                   expected=_SW_RESOURCES)


def submit_sw_flux_reduce(arrays, parameters, *, grid_size):
    name = "rrtmgp_exact_sw_flux_reduce" if WRF_EXACT else "rrtmgp_sw_flux_reduce"
    return _submit(name, arrays, dict(parameters), grid_size,
                   expected={"part_up", "part_dn", "part_dir", "flux_up",
                             "flux_dn", "flux_dir"}, exact_policy=WRF_EXACT)


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
    name = ("rrtmgp_exact_fluxes_to_radiation" if WRF_EXACT
            else "rrtmgp_fluxes_to_radiation")
    _submit(name, arrays, params, max(ncol * nlay, ncol),
            expected=set(arrays), exact_policy=WRF_EXACT)
    return outputs


__all__ = ["submit_delta_scale", "submit_lw_noscat", "submit_lw_flux_reduce",
           "submit_sw_2stream", "submit_sw_flux_reduce",
           "submit_fluxes_to_radiation", "SW2StreamWorkspace", "solve_sw_workspace"]
