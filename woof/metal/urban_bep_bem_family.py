"""Fail-closed admission for WRF urban BEP+BEM CUDA exports.

Both exports transitively use source-pinned transcendental and exact
rounding routines. Their experimental shared Metal math prerequisites have
not been reviewed, so this packet admits no kernels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .array import MetalArray


_UNSUPPORTED_EXPORTS = {
    "bep_bem_class_init": (
        "class initialization calls bb_init_para and bb_icbep, which reach "
        "the pinned gfk_log/gfk_pow routines and source-ordered arithmetic"),
    "bep_bem_columns": (
        "the column solve requires pinned gfk_exp/gfk_log/gfk_pow, "
        "CORE-MATH log10f, glibc/CORE-MATH trigonometry, and exact "
        "non-FTZ RN FP32 plus binary64 arithmetic"),
}


def submit_bep_bem_albedo(context, p_num: MetalArray, q_num: MetalArray,
                          albwin: MetalArray, error: MetalArray):
    """Submit the BEM window-albedo integral source fragment.

    This single prerequisite stage is separate from, and does not admit, the
    BEM class initializer or full column solver.
    """
    arrays = (("p_num", p_num, np.int32), ("q_num", q_num, np.int32),
              ("albwin", albwin, np.float32), ("error", error, np.int32))
    if p_num.ndim != 1:
        raise ValueError("BEM p_num must be a one-dimensional MetalArray")
    ncase = int(p_num.shape[0])
    for name, array, dtype in arrays:
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(dtype)
                or tuple(array.shape) != (ncase,)
                or array.buffer.context is not context):
            raise TypeError(f"BEM {name} must be a same-context typed vector")
    if ncase == 0:
        return None
    return context.submit_kernel(
        "urban_bep_bem_albedo_fragment",
        {"p_num": p_num, "q_num": q_num,
         "albwin": albwin, "error": error},
        {"ncase": ncase}, exact_policy=False, grid_size=ncase * 256,
        threads_per_group=256)


def launch_bep_bem_albedos_metal(context, nclass: int):
    """Build the shared BEM window albedo for the configured urban classes."""
    from .array import MetalArrayModule

    nclass = int(nclass)
    if not 1 <= nclass <= 11:
        raise ValueError("BEM class count must be in 1..11")
    module = MetalArrayModule(context)
    p_num = module.asarray(np.full((nclass,), 2, dtype=np.int32))
    q_num = module.asarray(np.full((nclass,), 4, dtype=np.int32))
    albwin = module.zeros((nclass,), dtype=np.float32)
    error = module.zeros((nclass,), dtype=np.int32)
    submit_bep_bem_albedo(context, p_num, q_num, albwin, error)
    return {"albwin": albwin, "error": error}


def require_urban_bep_bem_metal_supported(cuda_symbol: str) -> None:
    """Refuse BEP+BEM Metal use until all transitive math is reviewed."""
    reason = _UNSUPPORTED_EXPORTS.get(cuda_symbol)
    if reason is None:
        raise ValueError(f"unknown urban BEP+BEM CUDA export {cuda_symbol!r}")
    raise NotImplementedError(
        f"Metal {cuda_symbol} has no faithful translation: {reason}; "
        "no kernel signature is admitted")


def submit_bep_bem_view_factors(context, resources, *, nz_u: int, nd: int):
    """Submit the integrated per-column BEM canyon view-factor stage."""
    expected = {
        "z": (np.float32, 19),
        "dxy": (np.float32, "nd"),
        "street_width": (np.float32, "nd"),
        "fww": (np.float32, "nd", 18, 18),
        "fwg": (np.float32, "nd", 18),
        "fgw": (np.float32, "nd", 18),
        "fsw": (np.float32, "nd", 18),
        "fws": (np.float32, "nd", 18),
        "fsg": (np.float32, "nd"),
        "error": (np.int32, "nd"),
    }
    if set(resources) != set(expected):
        raise ValueError("BEM view-factor resources do not match fragment")
    ncol = None
    for name, (dtype, *tail_spec) in expected.items():
        array = resources[name]
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"BEM view-factor {name} has wrong Metal dtype")
        tail = tuple(int(nd) if extent == "nd" else int(extent)
                     for extent in tail_spec)
        if array.ndim != 1 + len(tail) or tuple(array.shape[1:]) != tail:
            raise ValueError(f"BEM view-factor {name} has wrong shape")
        ncol = int(array.shape[0]) if ncol is None else ncol
        if array.shape[0] != ncol or array.buffer.context is not context:
            raise ValueError("BEM view-factor arrays disagree on context/extent")
    if not 1 <= int(nz_u) <= 18 or int(nd) != 2:
        raise ValueError("BEM view-factor extents are outside WRF limits")
    if not ncol:
        return None
    return context.submit_kernel(
        "urban_bem_view_factors_fragment", resources,
        {"ncol": ncol, "nz_u": int(nz_u), "nd": int(nd)},
        exact_policy=False, grid_size=ncol * int(nd), threads_per_group=64)


def submit_bem_icbep_xy(context, resources, *, ncol: int, nurbm: int):
    """Submit per-column ``bb_icbep_xy`` using each column's own profile."""
    ncol, nurbm = int(ncol), int(nurbm)
    if ncol < 1 or not 1 <= nurbm <= 11:
        raise ValueError("BEM canyon factor extents are outside WRF limits")
    expected_shapes = {
        "active_cols": (ncol,), "utype": (ncol,), "class_int": (37,),
        "nzurban": (ncol,), "z_u": (19,), "strd": (ncol, 2),
        "ws": (ncol, 2), "fww": (ncol, 2, 18, 18),
        "fwg": (ncol, 2, 18), "fgw": (ncol, 2, 18),
        "fsw": (ncol, 2, 18), "fws": (ncol, 2, 18),
        "fsg": (ncol, 2), "error": (ncol, 2),
    }
    if set(resources) != set(expected_shapes):
        raise ValueError("BEM bb_icbep_xy resources do not match its source ABI")
    for name, shape in expected_shapes.items():
        if tuple(resources[name].shape) != shape:
            raise ValueError(f"BEM bb_icbep_xy {name} must have shape {shape}")
    return submit_bep_bem_fragment(
        context, "urban_bem_icbep_xy_fragment", resources,
        {"ncol": ncol, "nurbm": nurbm}, grid_size=ncol * 2)


def submit_bem_building_level_scan(context, resources, *, ncol: int):
    """Compact active positive BEM floor profiles after ``bb_param``."""
    ncol = int(ncol)
    if ncol < 1:
        raise ValueError("BEM building scan needs a positive column count")
    expected_shapes = {
        "active_cols": (ncol,), "ss_urb": (ncol, 19),
        "nlev": (ncol, 18), "d_urb": (ncol, 18),
        "nbui": (ncol,), "building_nlev": (ncol, 15),
        "error": (ncol,),
    }
    if set(resources) != set(expected_shapes):
        raise ValueError("BEM building scan resources do not match source ABI")
    for name, shape in expected_shapes.items():
        if tuple(resources[name].shape) != shape:
            raise ValueError(f"BEM building scan {name} must have shape {shape}")
    return submit_bep_bem_fragment(
        context, "urban_bem_building_level_scan_fragment", resources,
        {"ncol": ncol}, grid_size=ncol)


def submit_bem_atmosphere_gather(context, resources, *, ncol: int, nz: int):
    """Prepare the BEM one-dimensional atmospheric state and zero tendencies."""
    ncol, nz = int(ncol), int(nz)
    if ncol < 1 or nz < 1:
        raise ValueError("BEM atmosphere gather extents must be positive")
    expected_shapes = {
        "active_cols": (ncol,), "column_x": (ncol, 15, nz + 1),
        "qv": (nz, ncol), "ua1d": (ncol, nz), "va1d": (ncol, nz),
        "pt1d": (ncol, nz), "da1d": (ncol, nz), "pr1d": (ncol, nz),
        "pt01d": (ncol, nz), "z1d": (ncol, nz + 1),
        "qv1d": (ncol, nz), "a_u1d": (ncol, nz),
        "a_v1d": (ncol, nz), "a_t1d": (ncol, nz),
        "a_e1d": (ncol, nz), "b_u1d": (ncol, nz),
        "b_v1d": (ncol, nz), "b_t1d": (ncol, nz),
        "b_ac1d": (ncol, nz), "b_e1d": (ncol, nz),
        "error": (ncol,),
    }
    if set(resources) != set(expected_shapes):
        raise ValueError("BEM atmosphere resources do not match source ABI")
    for name, shape in expected_shapes.items():
        if tuple(resources[name].shape) != shape:
            raise ValueError(f"BEM atmosphere {name} must have shape {shape}")
    return submit_bep_bem_fragment(
        context, "urban_bem_atmosphere_gather_fragment", resources,
        {"ncol": ncol, "nz": nz}, grid_size=ncol)


def submit_bem_atmosphere_interp(context, resources, *, ncol: int, nz: int,
                                 nurbm: int):
    """Submit the seven WRF ``bb_interpol`` atmospheric overlaps."""
    ncol, nz, nurbm = int(ncol), int(nz), int(nurbm)
    if ncol < 1 or nz < 1 or not 1 <= nurbm <= 11:
        raise ValueError("BEM atmosphere interpolation extents are invalid")
    expected_shapes = {
        "active_cols": (ncol,), "utype": (ncol,),
        "nzu_class": (nurbm,), "z": (ncol, nz + 1), "z_u": (19,),
        "ua": (ncol, nz), "va": (ncol, nz), "pt": (ncol, nz),
        "pt0": (ncol, nz), "pr": (ncol, nz), "da": (ncol, nz),
        "qv": (ncol, nz), "interp": (ncol, 7, 18), "error": (ncol,),
    }
    if set(resources) != set(expected_shapes):
        raise ValueError("BEM atmosphere interpolation resources do not match source ABI")
    for name, shape in expected_shapes.items():
        if tuple(resources[name].shape) != shape:
            raise ValueError(f"BEM atmosphere interpolation {name} must have shape {shape}")
    return submit_bep_bem_fragment(
        context, "urban_bem_atmosphere_interp_fragment", resources,
        {"ncol": ncol, "nz": nz, "nurbm": nurbm}, grid_size=ncol)


def submit_bep_bem_profile_init(context, resources, *, nz_u: int = 18):
    """Submit the BEM building-height distribution profile setup."""
    del nz_u  # The transcribed bb_icbep uses all 18 input bins.
    return _submit_bep_bem_profile_stage(
        context, "urban_bem_profile_init_fragment", resources,
        {"z_u": (np.float32, (19,)), "h_b": (np.float32, (None, 18)),
         "d_b": (np.float32, (None, 18)), "ss_u": (np.float32, (None, 19)),
         "pb_u": (np.float32, (None, 19)), "nz_u": (np.int32, (None,)),
         "error": (np.int32, (None,))}, {"nurbm": None})


def submit_bep_bem_height_init(context, resources, *, dz_u: float):
    """Submit the BEM vertical-height profile setup."""
    return _submit_bep_bem_profile_stage(
        context, "urban_bem_height_init_fragment", resources,
        {"z_u": (np.float32, (19,)), "hi_urb1d": (np.float32, (None, 18)),
         "hb_u": (np.float32, (None, 18)), "ss_u": (np.float32, (None, 19)),
         "pb_u": (np.float32, (None, 19)), "nzu_out": (np.int32, (None,)),
         "error": (np.int32, (None,))},
        {"nitem": None, "dz_u": float(dz_u)})


def _submit_bep_bem_profile_stage(context, entrypoint, resources, expected,
                                  parameters):
    if set(resources) != set(expected):
        raise ValueError(f"{entrypoint} resources do not match its source ABI")
    nitem = None
    for name, (dtype, shape) in expected.items():
        array = resources[name]
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"{entrypoint} {name} has wrong Metal dtype")
        if shape == (19,):
            if tuple(array.shape) != shape:
                raise ValueError(f"{entrypoint} {name} must have shape {shape}")
            if array.buffer.context is not context:
                raise ValueError(f"{entrypoint} arrays belong to another context")
            continue
        if nitem is None:
            nitem = int(array.shape[0])
        resolved = tuple(nitem if extent is None else extent for extent in shape)
        if tuple(array.shape) != resolved:
            raise ValueError(f"{entrypoint} {name} must have shape {resolved}")
        if array.buffer.context is not context:
            raise ValueError(f"{entrypoint} arrays belong to another context")
    parameters = dict(parameters)
    if entrypoint == "urban_bem_height_init_fragment":
        parameters["nitem"] = int(nitem or 0)
        if parameters["nitem"] < 1:
            raise ValueError(f"{entrypoint} nitem must be positive")
    else:
        parameters["nurbm"] = int(nitem or 0)
        if not 1 <= parameters["nurbm"] <= 11:
            raise ValueError(f"{entrypoint} nurbm must be in 1..11")
    if entrypoint == "urban_bem_height_init_fragment":
        if not np.isfinite(parameters["dz_u"]) or parameters["dz_u"] <= 0:
            raise ValueError("BEM dz_u must be finite and positive")
    grid_size = (parameters["nitem"] if entrypoint == "urban_bem_height_init_fragment"
                 else parameters["nurbm"])
    return context.submit_kernel(entrypoint, resources, parameters,
        exact_policy=False, grid_size=grid_size, threads_per_group=64)


_BEM_FRAGMENT_PATHS = (
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bep_bem_family.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_view_factors_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_profile_init_family.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_solver_family.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_radiation_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_shadow_mas_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_surface_temp_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_rain_update_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bep_bem_upward_rad_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_interior_flux_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_interior_radiation_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_indoor_radiation_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_indoor_fluxes_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_room_balance_inputs_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_indoor_state_update_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_consump_total_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_occupancy_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_occupancy_inputs_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_room_control_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bep_bem_flat_roof_flux_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_wall_flux_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_flat_flux_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bep_bem_state_gather_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bep_bem_state_scatter_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_material_solve_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_buildings_tendency_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_urban_meso_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_interp_length_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_flat_flux_pack_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bep_bem_profile_select_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bep_bem_hi_profile_gather_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_param_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_icbep_xy_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_building_level_scan_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_atmosphere_gather_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_atmosphere_interp_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_temperature_profile_average_fragment.json",
    Path(__file__).resolve().parent / "signature_fragments" / "urban_bem_final_output_fragment.json",
)
_BEM_FRAGMENT_SIGNATURES = {}
for _path in _BEM_FRAGMENT_PATHS:
    _BEM_FRAGMENT_SIGNATURES.update(
        (entry["name"], entry) for entry in json.loads(_path.read_text())["kernels"])


def submit_bep_bem_fragment(context, entrypoint, resources, parameters, *, grid_size=None):
    """Submit one integrated, typed BEM source slice by its registered name."""
    try:
        spec = _BEM_FRAGMENT_SIGNATURES[entrypoint]
    except KeyError as exc:
        raise ValueError(f"BEM fragment {entrypoint!r} is not registered here") from exc
    resource_specs = {row[0]: row[1] for row in spec["resources"]}
    parameter_specs = {row[0]: row[1] for row in spec["parameters"]}
    if set(resources) != set(resource_specs):
        raise ValueError(f"{entrypoint} resources do not match source signature")
    if set(parameters) != set(parameter_specs):
        raise ValueError(f"{entrypoint} parameters do not match source signature")
    dtype_map = {"float": np.dtype(np.float32), "float32": np.dtype(np.float32),
                 "int32": np.dtype(np.int32)}
    for name, type_name in resource_specs.items():
        array = resources[name]
        if (not isinstance(array, MetalArray) or array.dtype != dtype_map[type_name]
                or not array.flags.c_contiguous
                or array.buffer.context is not context):
            raise TypeError(f"{entrypoint}.{name} has wrong dtype, layout, or context")
    packed_parameters = {}
    for name, type_name in parameter_specs.items():
        value = np.asarray(parameters[name], dtype=dtype_map[type_name])
        if value.ndim != 0:
            raise TypeError(f"{entrypoint}.{name} must be scalar")
        packed_parameters[name] = value.item()
    if grid_size is None:
        if "nitem" in packed_parameters:
            grid_size = packed_parameters["nitem"]
        elif "nurbm" in packed_parameters:
            grid_size = packed_parameters["nurbm"]
        elif "ncol" in packed_parameters:
            grid_size = packed_parameters["ncol"] * packed_parameters.get("nd", 1)
        else:
            raise ValueError(f"{entrypoint} has no default grid extent")
    grid_size = int(grid_size)
    if grid_size <= 0:
        return None
    return context.submit_kernel(entrypoint, resources, packed_parameters,
        exact_policy=False, grid_size=grid_size, threads_per_group=64)


def submit_bem_state_scatter(context, resources, *, ncol: int,
                             green_roof: int):
    """Commit packed BEM column state back to the resident WRF arrays.

    This is the inverse of the pre-solve state gather and must be dispatched
    only after the complete per-building solve succeeds.
    """
    ncol = int(ncol)
    green_roof = int(green_roof)
    if ncol < 1 or green_roof not in (0, 1):
        raise ValueError("BEM state scatter needs positive ncol and a binary green-roof flag")
    return submit_bep_bem_fragment(
        context, "urban_bep_bem_state_scatter_fragment", resources,
        {"ncol": ncol, "green_roof": green_roof}, grid_size=ncol)


def submit_bem_final_output(context, resources, *, ncol: int, nz: int,
                            class_float_len: int):
    """Aggregate building fluxes and scatter the completed BEM PBL column."""
    ncol, nz, class_float_len = int(ncol), int(nz), int(class_float_len)
    if ncol < 1 or nz < 2 or class_float_len < 1129:
        raise ValueError("BEM final-output extents are outside WRF limits")
    expected = {
        "active_cols": (ncol,), "fraction": (ncol,), "utype": (ncol,),
        "class_float": (class_float_len,), "bs": (ncol, 2), "ws": (ncol, 2),
        "building_nlev": (ncol, 15), "d_urb": (ncol, 18),
        "sflev": (ncol, 15, 18), "lflev": (ncol, 15, 18),
        "consum": (ncol, 15, 18), "sfvlev": (ncol, 15, 18),
        "lfvlev": (ncol, 15, 18), "sf": (ncol, nz + 1),
        "vl": (ncol, nz), "a_u": (ncol, nz), "a_v": (ncol, nz),
        "a_t": (ncol, nz), "a_e": (ncol, nz), "b_u": (ncol, nz),
        "b_v": (ncol, nz), "b_t": (ncol, nz), "b_e": (ncol, nz),
        "b_q": (ncol, nz), "dlg": (ncol, nz), "dl_u": (ncol, nz),
        "rl_up_in": (ncol,), "rs_abs_in": (ncol,), "emiss_in": (ncol,),
        "grdflx_in": (ncol,), "sf_ac": (ncol,), "lf_ac": (ncol,),
        "cm_ac": (ncol,), "sf_vent": (ncol,), "lf_vent": (ncol,),
        "ep_pv": (ncol,), "drain_green": (ncol,), "qgr": (ncol,),
        "tgr": (ncol,), "rl_up": (ncol,), "rs_abs": (ncol,),
        "emiss": (ncol,), "grdflx": (ncol,), "out_sf": (nz + 1, ncol),
        "out_vl": (nz, ncol), "out_a_u": (nz, ncol),
        "out_a_v": (nz, ncol), "out_a_t": (nz, ncol),
        "out_a_e": (nz, ncol), "out_b_u": (nz, ncol),
        "out_b_v": (nz, ncol), "out_b_t": (nz, ncol),
        "out_b_e": (nz, ncol), "out_b_q": (nz, ncol),
        "out_dlg": (nz, ncol), "out_dl_u": (nz, ncol),
    }
    if set(resources) != set(expected):
        raise ValueError("BEM final-output resources do not match its ABI")
    for name, shape in expected.items():
        if tuple(resources[name].shape) != shape:
            raise ValueError(f"BEM final-output {name} must have shape {shape}")
    return submit_bep_bem_fragment(
        context, "urban_bem_final_output_fragment", resources,
        {"ncol": ncol, "nz": nz, "nz_plus_one": nz + 1,
         "class_float_len": class_float_len}, grid_size=ncol)


def submit_bem_buildings_tendency(context, resources, *, ncol: int,
                                  nurbm: int, class_float_len: int,
                                  cp_u: float, latent: float, dz_u: float):
    """Run the ordered post-solve wall and flat-surface tendencies."""
    ncol, nurbm, class_float_len = int(ncol), int(nurbm), int(class_float_len)
    if ncol < 1 or not 1 <= nurbm <= 11 or class_float_len < 1239:
        raise ValueError("BEM building-tendency extents are outside WRF limits")
    shapes = {
        "active_cols": (ncol,), "building_nlev": (ncol, 15),
        "utype": (ncol,), "d_urb": (ncol, 18), "pb": (ncol, 19),
        "bs": (ncol, 2), "class_float": (class_float_len,),
        "interp": (ncol, 7, 18),
        "sflev": (ncol, 15, 18), "lflev": (ncol, 15, 18),
        "sfvlev": (ncol, 15, 18), "lfvlev": (ncol, 15, 18),
        "wall_uva": (ncol, 15, 18, 4), "wall_vva": (ncol, 15, 18, 4),
        "wall_uvb": (ncol, 15, 18, 4), "wall_vvb": (ncol, 15, 18, 4),
        "wall_sfw": (ncol, 15, 18, 4), "wall_sfwin": (ncol, 15, 18, 4),
        "wall_evb": (ncol, 15, 18, 4),
        "flat_uhb": (ncol, 2, 18), "flat_vhb": (ncol, 2, 18),
        "flat_thb": (ncol, 2, 18), "flat_ehb": (ncol, 2, 18),
        "flat_qhb": (ncol, 2, 18),
        "uva": (ncol, 4, 18), "vva": (ncol, 4, 18),
        "uvb": (ncol, 4, 18), "vvb": (ncol, 4, 18),
        "tvb": (ncol, 4, 18), "tvb_ac": (ncol, 4, 18),
        "evb": (ncol, 4, 18), "qvb": (ncol, 4, 18),
        "uhb": (ncol, 2, 18), "vhb": (ncol, 2, 18),
        "thb": (ncol, 2, 18), "ehb": (ncol, 2, 18),
        "qhb_flat": (ncol, 2, 18), "error": (ncol, 2, 18),
    }
    if set(resources) != set(shapes):
        raise ValueError("BEM building-tendency resources do not match ABI")
    for name, shape in shapes.items():
        if tuple(resources[name].shape) != shape:
            raise ValueError(f"BEM building-tendency {name} must have shape {shape}")
    return submit_bep_bem_fragment(
        context, "urban_bem_buildings_tendency_fragment", resources,
        {"ncol": ncol, "nitem": ncol * 2 * 18, "nurbm": nurbm,
         "class_float_len": class_float_len, "cp_u": float(cp_u),
         "latent": float(latent), "dz_u": float(dz_u)},
        grid_size=ncol * 2 * 18)


def submit_bem_urban_meso(context, resources, *, ncol: int, nz: int):
    """Map the per-column building tendencies to atmospheric mass levels."""
    ncol, nz = int(ncol), int(nz)
    if ncol < 1 or nz < 2:
        raise ValueError("BEM urban-meso extents must have ncol>0 and nz>=2")
    shapes = {
        "active_cols": (ncol,), "z": (ncol, nz + 1), "dz": (nz, ncol),
        "z_u": (19,), "pb": (ncol, 19), "ss": (ncol, 19),
        "bs": (ncol, 2), "ws": (ncol, 2),
        "uva": (ncol, 4, 18), "vva": (ncol, 4, 18),
        "uvb": (ncol, 4, 18), "vvb": (ncol, 4, 18),
        "tvb": (ncol, 4, 18), "tvb_ac": (ncol, 4, 18),
        "evb": (ncol, 4, 18), "uhb": (ncol, 2, 18),
        "vhb": (ncol, 2, 18), "thb": (ncol, 2, 18),
        "ehb": (ncol, 2, 18), "qhb": (ncol, 2, 18),
        "qvb": (ncol, 4, 18),
        **{name: (ncol, nz) for name in (
            "a_u", "a_v", "a_t", "a_e", "b_u", "b_v", "b_t",
            "b_e", "b_q", "b_ac", "vl")},
        "sf": (ncol, nz + 1), "error": (ncol,),
    }
    if set(resources) != set(shapes):
        raise ValueError("BEM urban-meso resources do not match ABI")
    for name, shape in shapes.items():
        if tuple(resources[name].shape) != shape:
            raise ValueError(f"BEM urban-meso {name} must have shape {shape}")
    return submit_bep_bem_fragment(
        context, "urban_bem_urban_meso_fragment", resources,
        {"ncol": ncol, "nz": nz, "nedge": nz + 1}, grid_size=ncol)


def submit_bem_interp_length(context, resources, *, ncol: int, nz: int):
    """Compute the WRF urban and building drag length profiles."""
    ncol, nz = int(ncol), int(nz)
    if ncol < 1 or nz < 2:
        raise ValueError("BEM interpolation-length extents must have ncol>0 and nz>=2")
    shapes = {
        "active_cols": (ncol,), "z": (ncol, nz + 1), "z_u": (19,),
        "ss": (ncol, 19), "ws": (ncol, 2), "bs": (ncol, 2),
        "dlg": (ncol, nz), "dl_u": (ncol, nz), "error": (ncol,),
    }
    if set(resources) != set(shapes):
        raise ValueError("BEM interpolation-length resources do not match ABI")
    for name, shape in shapes.items():
        if tuple(resources[name].shape) != shape:
            raise ValueError(f"BEM interpolation-length {name} must have shape {shape}")
    return submit_bep_bem_fragment(
        context, "urban_bem_interp_length_fragment", resources,
        {"ncol": ncol, "nz": nz, "nedge": nz + 1}, grid_size=ncol)


def submit_bem_material_solve(context, resources, *, ncol: int, dt: float):
    """Run the ordinary-profile WRF material stacks for packed buildings."""
    ncol = int(ncol)
    nitem = int(resources["building_nlev"].size)
    if ncol < 1 or nitem != ncol * 15 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("BEM material solve needs ncol*15 rows and positive finite dt")
    return submit_bep_bem_fragment(
        context, "urban_bem_material_solve_fragment", resources,
        {"ncol": ncol, "nitem": nitem, "dt": float(dt)},
        grid_size=nitem)


def submit_bem_flat_flux_pack(context, resources, *, ncol: int):
    """Pack flat surface fluxes and gather roof forcing in WRF index order."""
    ncol = int(ncol)
    if ncol < 1:
        raise ValueError("BEM flat-flux packing needs positive ncol")
    return submit_bep_bem_fragment(
        context, "urban_bem_flat_flux_pack_fragment", resources,
        {"ncol": ncol}, grid_size=ncol)


def submit_bem_param(context, resources, *, ncol: int, nurbm: int):
    """Submit WRF ``bb_param`` for each domain column.

    The arrays follow the standalone fragment ABI: class tables retain their
    packed Fortran offsets, while per-column workspaces are row-major with
    column first. The caller allocates zeroed outputs so inactive columns
    retain WRF's pre-initialized zero values.
    """
    ncol, nurbm = int(ncol), int(nurbm)
    if ncol < 1 or not 1 <= nurbm <= 11:
        raise ValueError("BEM bb_param extents are outside WRF limits")
    expected_shapes = {
        "active_cols": (ncol,), "utype": (ncol,),
        "class_float": (1320,), "class_int": (37,),
        "ss_class": (nurbm, 19), "pb_class": (nurbm, 19),
        "nzu_class": (nurbm,), "ss_urban": (ncol, 19),
        "pb_urban": (ncol, 19), "nzurb_column": (ncol,),
        "lp_urb": (ncol,), "lb_urb": (ncol,), "hgt_urb": (ncol,),
        "frc_urb": (ncol,), "error": (ncol,), "nzurban": (ncol,),
        "ss": (ncol, 19), "pb": (ncol, 19),
        "ss_urb": (ncol, 19), "pb_urb": (ncol, 19),
        "csg": (ncol, 10), "alag": (ncol, 10),
        "csgb": (ncol, 10), "alagb": (ncol, 11),
        "csf": (ncol, 10), "alaf": (ncol, 11),
        "csr": (ncol, 10), "alar": (ncol, 11),
        "csw": (ncol, 10), "alaw": (ncol, 11),
        "z0": (ncol, 2, 18), "strd": (ncol, 2),
        "drst": (ncol, 2), "ws": (ncol, 2), "bs": (ncol, 2),
        "ws_urb": (ncol, 2), "bs_urb": (ncol, 2),
        "dzw": (ncol, 10), "dzr": (ncol, 10),
        "dzf": (ncol, 10), "dzgb": (ncol, 10),
    }
    if set(resources) != set(expected_shapes):
        raise ValueError("BEM bb_param resources do not match its source ABI")
    for name, shape in expected_shapes.items():
        if tuple(resources[name].shape) != shape:
            raise ValueError(f"BEM bb_param {name} must have shape {shape}")
    return submit_bep_bem_fragment(
        context, "urban_bem_param_fragment", resources,
        {"ncol": ncol, "nurbm": nurbm}, grid_size=ncol)


def submit_bem_temperature_average(context, resources, *, ncol: int):
    """Submit WRF ``bb_averaging_temp`` for each urban column."""
    ncol = int(ncol)
    expected_shapes = {
        "ss": (ncol, 19), "pb": (ncol, 19),
        "tw": (ncol, 15, 5, 18, 4),
        "twlev": (ncol, 15, 18, 4), "sfw": (ncol, 15, 18, 4),
        "sfwin": (ncol, 15, 18, 4),
        "tw_av": (ncol, 4, 18), "twlev_av": (ncol, 4, 18),
        "sfw_av": (ncol, 4, 18), "sfwind_av": (ncol, 4, 18),
        "error": (ncol,),
    }
    if ncol < 1 or set(resources) != set(expected_shapes):
        raise ValueError("BEM temperature-average resources do not match source ABI")
    for name, shape in expected_shapes.items():
        if tuple(resources[name].shape) != shape:
            raise ValueError(f"BEM temperature-average {name} must have shape {shape}")
    return submit_bep_bem_fragment(
        context, "urban_bem_temperature_average_fragment", resources,
        {"ncol": ncol}, grid_size=ncol)


def submit_bem_temperature_profile_average(context, resources, *, ncol: int):
    """Submit WRF ``tg_av`` and green-roof weighted ``tr_av`` preparation."""
    ncol = int(ncol)
    expected_shapes = {
        "active_cols": (ncol,), "utype": (ncol,), "class_float": (1320,),
        "tg": (ncol, 10, 2), "tr": (ncol, 5, 18, 2),
        "trv": (ncol, 10, 18, 2), "tg_av": (ncol, 2),
        "tr_av": (ncol, 18, 2), "error": (ncol,),
    }
    if ncol < 1 or set(resources) != set(expected_shapes):
        raise ValueError("BEM temperature-profile resources do not match source ABI")
    for name, shape in expected_shapes.items():
        if tuple(resources[name].shape) != shape:
            raise ValueError(f"BEM temperature-profile {name} must have shape {shape}")
    return submit_bep_bem_fragment(
        context, "urban_bem_temperature_profile_average_fragment", resources,
        {"ncol": ncol}, grid_size=ncol)


def submit_bem_occupancy(context, resources, *, nitem: int, nhourday: float):
    """Submit the BEM occupancy and internal-gain stage."""
    return submit_bep_bem_fragment(
        context, "urban_bem_occupancy_fragment", resources,
        {"nitem": int(nitem), "nhourday": float(nhourday)},
        grid_size=int(nitem))


def submit_bem_occupancy_inputs(context, resources, *, ncol: int,
                                class_float_len: int):
    """Pack per-class equipment/occupancy inputs for each building slot."""
    ncol = int(ncol)
    if ncol < 1:
        raise ValueError("BEM occupancy selection requires columns")
    return submit_bep_bem_fragment(
        context, "urban_bem_occupancy_inputs_fragment", resources,
        {"ncol": ncol, "nitem": ncol * 15,
         "class_float_len": int(class_float_len)}, grid_size=ncol * 15)


def submit_bem_room_control(context, resources, *, nitem: int,
                            nhourday: float, dt: float):
    """Submit ventilation, thermostat, and room humidity control."""
    return submit_bep_bem_fragment(
        context, "urban_bem_room_control_fragment", resources,
        {"nitem": int(nitem), "nhourday": float(nhourday), "dt": float(dt)},
        grid_size=int(nitem))


def submit_bem_roof_radiation(context, resources, *, nitem: int,
                              has_pv: int, level: int, nz: int,
                              sigma: float):
    """Submit roof radiative exchange for each building item."""
    return submit_bep_bem_fragment(
        context, "urban_bem_roof_radiation_fragment", resources,
        {"nitem": int(nitem), "has_pv": int(has_pv), "level": int(level),
         "nz": int(nz), "sigma": float(sigma)}, grid_size=int(nitem))


def submit_bem_shortwave(context, resources, *, ncol: int, nurbm: int):
    """Submit diffuse shortwave with per-column urban class selectors."""
    ncol, nurbm = int(ncol), int(nurbm)
    return submit_bep_bem_fragment(
        context, "urban_bem_shortwave_fragment", resources,
        {"ncol": ncol, "nurbm": nurbm}, grid_size=ncol * 2)


def submit_bem_longwave(context, resources, *, ncol: int, nurbm: int,
                        sigma: float):
    """Submit longwave with per-column urban class selectors."""
    ncol, nurbm = int(ncol), int(nurbm)
    return submit_bep_bem_fragment(
        context, "urban_bem_longwave_fragment", resources,
        {"ncol": ncol, "nurbm": nurbm, "sigma": float(sigma)},
        grid_size=ncol * 2)


def submit_bem_shadow_mas(context, resources, *, ncol: int, nurbm: int,
                          declin: float):
    """Submit WRF direct-beam canyon shadow geometry by column."""
    ncol, nurbm = int(ncol), int(nurbm)
    return submit_bep_bem_fragment(
        context, "urban_bem_shadow_mas_fragment", resources,
        {"ncol": ncol, "nurbm": nurbm, "declin": float(declin)},
        grid_size=ncol)


def submit_bem_upward_rad(context, resources, *, nitem: int, nurbm: int,
                          sigma: float, dz_u: float):
    """Submit WRF's aggregate upward radiation and surface fluxes per column."""
    nitem, nurbm = int(nitem), int(nurbm)
    return submit_bep_bem_fragment(
        context, "urban_bep_bem_upward_rad_fragment", resources,
        {"nitem": nitem, "nurbm": nurbm,
         "sigma": float(sigma), "dz_u": float(dz_u)},
        grid_size=nitem)


def submit_bem_surface_temp(context, resources, *, ncol: int, nurbm: int,
                            dt: float, sigma: float):
    """Submit BEM ground soil-temperature solve and surface flux update."""
    ncol, nurbm = int(ncol), int(nurbm)
    return submit_bep_bem_fragment(
        context, "urban_bem_surface_temp_fragment", resources,
        {"ncol": ncol, "nurbm": nurbm,
         "dt": float(dt), "sigma": float(sigma)}, grid_size=ncol)


def submit_bem_rain_update(context, resources, *, ncol: int, nurbm: int,
                           dt: float, latent: float):
    """Submit WRF's bounded rainfall/drainage update for road and roof soil."""
    ncol, nurbm = int(ncol), int(nurbm)
    return submit_bep_bem_fragment(
        context, "urban_bem_rain_update_fragment", resources,
        {"ncol": ncol, "nurbm": nurbm,
         "dt": float(dt), "latent": float(latent)}, grid_size=ncol)


def submit_bem_interior_flux(context, resources, *, ncol: int, nlev: int,
                             pwin: float):
    """Submit indoor sensible fluxes and room-level heat/moisture sums."""
    ncol, nlev = int(ncol), int(nlev)
    return submit_bep_bem_fragment(
        context, "urban_bem_interior_flux_fragment", resources,
        {"ncol": ncol, "nlev": nlev, "pwin": float(pwin)},
        grid_size=ncol * nlev)


def submit_bem_interior_radiation(context, resources, *, nitem: int,
                                   albwal: float, albwin: float,
                                   emwal: float, emwin: float, sigma: float,
                                   pwin: float, bw: float, bl: float,
                                   zw: float):
    """Submit the six-surface indoor shortwave and longwave exchange solve."""
    return submit_bep_bem_fragment(
        context, "urban_bem_interior_radiation_fragment", resources,
        {"nitem": int(nitem), "albwal": float(albwal),
         "albwin": float(albwin), "emwal": float(emwal),
         "emwin": float(emwin), "sigma": float(sigma),
         "pwin": float(pwin), "bw": float(bw), "bl": float(bl),
         "zw": float(zw)}, grid_size=int(nitem))


def submit_bem_indoor_radiation(context, resources, *, ncol: int,
                                nitem: int, nurbm: int,
                                class_float_len: int, dz_u: float):
    """Run WRF's per-building six-surface indoor radiation solves."""
    ncol, nitem, nurbm = int(ncol), int(nitem), int(nurbm)
    if ncol < 1 or nitem != ncol * 15 * 18 or not 1 <= nurbm <= 11:
        raise ValueError("BEM indoor radiation dimensions are invalid")
    return submit_bep_bem_fragment(
        context, "urban_bem_indoor_radiation_fragment", resources,
        {"ncol": ncol, "nitem": nitem, "nurbm": nurbm,
         "class_float_len": int(class_float_len), "dz_u": float(dz_u)},
        grid_size=nitem)


def submit_bem_indoor_fluxes(context, resources, *, ncol: int,
                             nitem: int, nurbm: int,
                             class_float_len: int, dz_u: float):
    """Run BEM wall/window radiation, convection, and surface sums."""
    ncol, nitem, nurbm = int(ncol), int(nitem), int(nurbm)
    if ncol < 1 or nitem != ncol * 15 * 18 or not 1 <= nurbm <= 11:
        raise ValueError("BEM indoor flux dimensions are invalid")
    return submit_bep_bem_fragment(
        context, "urban_bem_indoor_fluxes_fragment", resources,
        {"ncol": ncol, "nitem": nitem, "nurbm": nurbm,
         "class_float_len": int(class_float_len), "dz_u": float(dz_u)},
        grid_size=nitem)


def submit_bem_room_balance_inputs(context, resources, *, ncol: int,
                                   nitem: int, nurbm: int,
                                   class_float_len: int, class_int_len: int,
                                   dz_u: float):
    """Pack WRF thermodynamics, ventilation, and room-control inputs."""
    ncol, nitem, nurbm = int(ncol), int(nitem), int(nurbm)
    if ncol < 1 or nitem != ncol * 15 * 18 or not 1 <= nurbm <= 11:
        raise ValueError("BEM room-balance dimensions are invalid")
    return submit_bep_bem_fragment(
        context, "urban_bem_room_balance_inputs_fragment", resources,
        {"ncol": ncol, "nitem": nitem, "nitem_building": ncol * 15,
         "nurbm": nurbm,
         "class_float_len": int(class_float_len),
         "class_int_len": int(class_int_len), "dz_u": float(dz_u)},
        grid_size=nitem)


def submit_bem_indoor_state_update(context, resources, *, ncol: int,
                                   nitem: int, dt: float):
    """Write staged BEM indoor temperature and humidity outputs."""
    ncol, nitem = int(ncol), int(nitem)
    if ncol < 1 or nitem != ncol * 15 * 18:
        raise ValueError("BEM indoor state-update dimensions are invalid")
    return submit_bep_bem_fragment(
        context, "urban_bem_indoor_state_update_fragment", resources,
        {"ncol": ncol, "nitem": nitem, "dt": float(dt)}, grid_size=nitem)


def submit_bem_consump_total(context, resources, *, nitem: int, nzcanm: int):
    """Reduce each building's room-level sensible and consumption totals."""
    return submit_bep_bem_fragment(
        context, "urban_bem_consump_total_fragment", resources,
        {"nitem": int(nitem), "nzcanm": int(nzcanm)},
        grid_size=int(nitem))


def submit_bem_wall_flux(context, resources, *, ncol: int, p0: float,
                         rcp_u: float, cp_u: float, dt: float):
    """Submit the BEM per-building, per-level four-face wall flux solve."""
    ncol = int(ncol)
    if ncol < 1 or not float(dt) > 0.0:
        raise ValueError("BEM wall flux requires columns and positive dt")
    return submit_bep_bem_fragment(
        context, "urban_bem_wall_flux_fragment", resources,
        {"ncol": ncol, "nitem": ncol * 15 * 18 * 4,
         "p0": float(p0), "rcp_u": float(rcp_u),
         "cp_u": float(cp_u), "dt": float(dt)},
        grid_size=ncol * 15 * 18 * 4)


def submit_bem_flat_flux(context, resources, *, ncol: int, g_u: float,
                         vk: float, p0: float, rcp_u: float, cp_u: float,
                         latent: float, dz_u: float, dgmax: float,
                         drmax: float):
    """Submit standard BEM ground and flat-roof surface exchange."""
    ncol = int(ncol)
    if ncol < 1 or not float(dz_u) > 0.0:
        raise ValueError("BEM flat flux requires columns and positive dz_u")
    return submit_bep_bem_fragment(
        context, "urban_bem_flat_flux_fragment", resources,
        {"ncol": ncol, "nitem": ncol * 2 * 18, "g_u": float(g_u),
         "vk": float(vk), "p0": float(p0), "rcp_u": float(rcp_u),
         "cp_u": float(cp_u), "latent": float(latent),
         "dz_u": float(dz_u), "dgmax": float(dgmax),
         "drmax": float(drmax)},
        grid_size=ncol * 2 * 18)
