"""Typed resident launch proposals for Urban BEP setup and column preparation."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


TABLE_FLOATS = 1386
TABLE_INTS = 23
CLASS_FLOATS = 1085
CLASS_INTS = 23
VIEW_FLOATS = 794
COLUMN_GATHER_FIELDS = 15
URBAN_PROFILE_FIELDS = 4
COLUMN_PARAM_FLOATS = 140
COLUMN_STATE_FLOATS = 1210
COLUMN_BUILDING_FLOATS = 648
COLUMN_MESO_FIELDS = 10
_THREADS = 64


def submit_urban_bep_radiation(context, resources, *, nz, declin):
    """Submit direct-solar shadowing from canonical column buffers."""
    expected = {
        "column_x": (np.float32, None), "column_param": (np.float32, (140,)),
        "column_param_ix": (np.int32, (3,)),
        "active_cols": (np.int32, ()), "swdown": (np.float32, ()),
        "cosz": (np.float32, ()), "hour_angle": (np.float32, ()),
        "rsg": (np.float32, (2,)), "rsw": (np.float32, (4, 18)),
        "error": (np.int32, ()),
    }
    if set(resources) != set(expected):
        raise ValueError("urban BEP radiation resources do not match fragment")
    ncol = None
    for name, (dtype, tail) in expected.items():
        array = resources[name]
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"urban BEP radiation {name} has wrong Metal dtype")
        if array.ndim != (3 if tail is None else len(tail) + 1):
            raise ValueError(f"urban BEP radiation {name} has wrong rank")
        if tail is not None and tuple(array.shape[1:]) != tuple(tail):
            raise ValueError(f"urban BEP radiation {name} has wrong shape")
        ncol = int(array.shape[0]) if ncol is None else ncol
        if array.shape[0] != ncol or array.buffer.context is not context:
            raise ValueError("urban BEP radiation arrays disagree on extent/context")
    if (resources["column_x"].ndim != 3
            or resources["column_x"].shape[1:] != (15, int(nz) + 1)):
        raise ValueError("urban BEP radiation column_x must have 15 channels")
    if not ncol:
        return None
    return context.submit_kernel(
        "urban_bep_radiation_fragment", resources,
        {"ncol": ncol, "nz": int(nz), "declin": float(declin)},
        exact_policy=False, grid_size=ncol,
        threads_per_group=_THREADS)


def submit_urban_bep_surface_temperature(context, resources, *, nz, dt):
    """Advance the packed ground, roof, and wall profiles in-place."""
    expected = {
        "column_param": ((140,), np.float32),
        "column_param_ix": ((3,), np.int32),
        "class_x": (None, np.float32),
        "utype_urb": ((), np.int32), "active_cols": ((), np.int32),
        "column_state": ((1210,), np.float32),
        "column_interp_x": ((108,), np.float32),
        "swdown": ((), np.float32), "glw": ((), np.float32),
        "rlw": ((2, 2, 18), np.float32), "rlg": ((2,), np.float32),
        "rsw": ((4, 18), np.float32), "rsg": ((2,), np.float32),
        "surface_x": ((330,), np.float32), "error": ((), np.int32),
    }
    if set(resources) != set(expected):
        raise ValueError("urban BEP surface resources do not match fragment")
    ncol = None
    for name, (tail, dtype) in expected.items():
        array = resources[name]
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"urban BEP surface {name} has wrong Metal dtype")
        if name == "class_x":
            if array.shape != (1, 1085) or array.buffer.context is not context:
                raise ValueError("urban BEP surface class_x must be one class row")
            continue
        if array.ndim != 1 + len(tail) or tuple(array.shape[1:]) != tail:
            raise ValueError(f"urban BEP surface {name} has wrong shape")
        ncol = int(array.shape[0]) if ncol is None else ncol
        if array.shape[0] != ncol or array.buffer.context is not context:
            raise ValueError("urban BEP surface arrays disagree on extent/context")
    if not 1 <= int(nz) <= np.iinfo(np.int32).max or not np.isfinite(dt) or dt <= 0:
        raise ValueError("urban BEP surface nz/dt is outside the source domain")
    if not ncol:
        return None
    return context.submit_kernel(
        "urban_bep_surface_temperature_fragment", resources,
        {"ncol": ncol, "dt": float(dt)},
        exact_policy=False, grid_size=ncol * 108,
        threads_per_group=_THREADS)


def submit_urban_bep_radiosity(context, resources):
    """Submit radiosity from canonical BEP column, view, and state buffers."""
    expected = {
        "column_param": ((140,), np.float32),
        "column_param_ix": ((3,), np.int32),
        "column_view_x": ((794,), np.float32),
        "column_state": ((1210,), np.float32),
        "class_x": ((1, 1085), np.float32),
        "utype_urb": ((), np.int32), "active_cols": ((), np.int32),
        "glw": ((), np.float32), "direct_rsg": ((2,), np.float32),
        "direct_rsw": ((4, 18), np.float32),
        "rlw": ((2, 2, 18), np.float32), "rlg": ((2,), np.float32),
        "rsw": ((4, 18), np.float32), "rsg": ((2,), np.float32),
        "error": ((2,), np.int32),
    }
    if set(resources) != set(expected):
        raise ValueError("urban BEP radiosity resources do not match fragment")
    ncol = None
    for name, (tail, dtype) in expected.items():
        array = resources[name]
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"urban BEP radiosity {name} has wrong Metal dtype")
        if name == "class_x":
            if array.shape != (1, 1085,) or array.buffer.context is not context:
                raise ValueError("urban BEP radiosity class_x must be one class row")
            continue
        if array.ndim != len(tail) + 1:
            raise ValueError(f"urban BEP radiosity {name} has wrong rank")
        if tail and tuple(array.shape[1:]) != tail:
            raise ValueError(f"urban BEP radiosity {name} has wrong shape")
        ncol = int(array.shape[0]) if ncol is None else ncol
        if array.shape[0] != ncol or array.buffer.context is not context:
            raise ValueError("urban BEP radiosity arrays disagree on extent/context")
    if not ncol:
        return None
    return context.submit_kernel(
        "urban_bep_radiosity_fragment", resources,
        {"ncol": ncol}, exact_policy=False, grid_size=ncol * 2,
        threads_per_group=_THREADS)


def submit_urban_bep_bem_wall(context, resources, *, nz, dt):
    """Submit one BEP+BEM wall/roof conduction profile per item."""
    rank2 = ("dz", "cs", "flux", "temp", "k1", "k2", "kc")
    if set(resources) != set(rank2 + (
            "k", "swwall", "green_roof", "green_fraction", "hfgr", "error")):
        raise ValueError("BEP+BEM wall resources do not match fragment")
    tails = {"dz": (20,), "k": (21,), "cs": (20,), "flux": (2,),
             "temp": (20,), "k1": (20,), "k2": (20,), "kc": (20,),
             "swwall": (), "green_roof": (), "green_fraction": (),
             "hfgr": (), "error": ()}
    integer = {"swwall", "green_roof", "error"}
    nitem = None
    for name, tail in tails.items():
        array = resources[name]
        dtype = np.int32 if name in integer else np.float32
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"BEP+BEM wall {name} has wrong Metal dtype")
        if array.ndim != 1 + len(tail):
            raise ValueError(f"BEP+BEM wall {name} has wrong rank")
        if tail and tuple(array.shape[1:]) != tail:
            raise ValueError(f"BEP+BEM wall {name} has wrong shape")
        nitem = int(array.shape[0]) if nitem is None else nitem
        if array.shape[0] != nitem or array.buffer.context is not context:
            raise ValueError("BEP+BEM wall arrays disagree on extent/context")
    if not 2 <= int(nz) <= 20 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("BEP+BEM wall nz/dt is outside the source domain")
    if not nitem:
        return None
    return context.submit_kernel(
        "urban_bep_bem_wall_fragment", resources,
        {"nitem": nitem, "nz": int(nz), "dt": float(dt)},
        exact_policy=False, grid_size=nitem, threads_per_group=_THREADS)


def submit_urban_bep_column_buildings(context, resources, *, dt):
    """Submit the exchange stage from canonical column and surface buffers."""
    expected = {
        "column_param": ((140,), np.float32),
        "column_param_ix": ((3,), np.int32),
        "column_interp_x": ((108,), np.float32),
        "surface_x": ((330,), np.float32),
        "column_building_x": ((648,), np.float32),
    }
    if set(resources) != set(expected):
        raise ValueError("urban BEP buildings resource names do not match fragment")
    ncol = None
    for name, (tail, dtype) in expected.items():
        array = resources[name]
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"urban BEP buildings {name} has wrong Metal dtype")
        if array.ndim != len(tail) + 1 or tuple(array.shape[1:]) != tail:
            raise ValueError(f"urban BEP buildings {name} has wrong shape")
        ncol = int(array.shape[0]) if ncol is None else ncol
        if array.shape[0] != ncol:
            raise ValueError("urban BEP buildings columns have inconsistent extents")
        if array.buffer.context is not context:
            raise ValueError("urban BEP buildings arrays belong to another context")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("urban BEP buildings dt must be finite and positive")
    if not ncol:
        return None
    return context.submit_kernel(
        "urban_bep_column_buildings_fragment", resources,
        {"ncol": ncol, "dt": float(dt)},
        exact_policy=False, grid_size=ncol, threads_per_group=_THREADS)


def submit_urban_bep_class_init(context, table_x: MetalArray,
                                table_ix: MetalArray, class_x: MetalArray,
                                class_ix: MetalArray):
    """Initialize flat BEP class records from split typed table records.

    ``table_x`` and ``class_x`` use the canonical float-only field layouts
    documented in the packet handoff; integer members are carried separately.
    This performs no host conversion or numerical work.
    """
    if not isinstance(table_x, MetalArray) or table_x.dtype != np.dtype(np.float32):
        raise TypeError("urban BEP table_x must be a float32 MetalArray")
    if not isinstance(table_ix, MetalArray) or table_ix.dtype != np.dtype(np.int32):
        raise TypeError("urban BEP table_ix must be an int32 MetalArray")
    if not isinstance(class_x, MetalArray) or class_x.dtype != np.dtype(np.float32):
        raise TypeError("urban BEP class_x must be a float32 MetalArray")
    if not isinstance(class_ix, MetalArray) or class_ix.dtype != np.dtype(np.int32):
        raise TypeError("urban BEP class_ix must be an int32 MetalArray")
    if table_x.ndim != 2:
        raise ValueError("urban BEP table_x must have shape (ncase, 1386)")
    ncase = int(table_x.shape[0])
    expected = (
        (table_x, (ncase, TABLE_FLOATS), "table_x"),
        (table_ix, (ncase, TABLE_INTS), "table_ix"),
        (class_x, (ncase, CLASS_FLOATS), "class_x"),
        (class_ix, (ncase, CLASS_INTS), "class_ix"),
    )
    for array, shape, name in expected:
        if tuple(array.shape) != shape:
            raise ValueError(f"urban BEP {name} must have shape {shape}")
    if not ncase:
        return None
    return context.submit_kernel(
        "urban_bep_class_init",
        {"table_x": table_x, "table_ix": table_ix,
         "class_x": class_x, "class_ix": class_ix},
        {"ncase": ncase}, exact_policy=False, grid_size=ncase,
        threads_per_group=_THREADS)


def submit_urban_bep_view_init(context, class_x: MetalArray,
                               class_ix: MetalArray, view_x: MetalArray,
                               view_ix: MetalArray):
    """Build per-class BEP view factors from resident class records.

    The class rows expand into eleven output rows, each carrying 794 floats
    and one int32 error code. This is source-only and not production admitted.
    """
    if not isinstance(class_x, MetalArray) or class_x.dtype != np.dtype(np.float32):
        raise TypeError("urban BEP class_x must be a float32 MetalArray")
    if not isinstance(class_ix, MetalArray) or class_ix.dtype != np.dtype(np.int32):
        raise TypeError("urban BEP class_ix must be an int32 MetalArray")
    if not isinstance(view_x, MetalArray) or view_x.dtype != np.dtype(np.float32):
        raise TypeError("urban BEP view_x must be a float32 MetalArray")
    if not isinstance(view_ix, MetalArray) or view_ix.dtype != np.dtype(np.int32):
        raise TypeError("urban BEP view_ix must be an int32 MetalArray")
    if class_x.ndim != 2:
        raise ValueError("urban BEP class_x must have shape (ncase, 1085)")
    ncase = int(class_x.shape[0])
    if ncase > np.iinfo(np.int32).max // 11:
        raise ValueError("urban BEP class count exceeds int32 grid addressing")
    expected = (
        (class_x, (ncase, CLASS_FLOATS), "class_x"),
        (class_ix, (ncase, CLASS_INTS), "class_ix"),
        (view_x, (ncase, 11, VIEW_FLOATS), "view_x"),
        (view_ix, (ncase, 11), "view_ix"),
    )
    for array, shape, name in expected:
        if tuple(array.shape) != shape:
            raise ValueError(f"urban BEP {name} must have shape {shape}")
    if not ncase:
        return None
    return context.submit_kernel(
        "urban_bep_view_init",
        {"class_x": class_x, "class_ix": class_ix,
         "view_x": view_x, "view_ix": view_ix},
        {"ncase": ncase}, exact_policy=False, grid_size=ncase * 11,
        threads_per_group=_THREADS)


def submit_urban_bep_column_gather(
        context, frc_urb: MetalArray, dz8w: MetalArray,
        u_phy: MetalArray, v_phy: MetalArray, th_phy: MetalArray,
        rho: MetalArray, p_phy: MetalArray, column_x: MetalArray,
        active_cols: MetalArray):
    """Gather the source column inputs and initialize solver coefficients.

    ``column_x`` is ``(ncol, 15, nz + 1)`` in channel order
    ``z, ua, va, pt, da, pr, pt0, a_u, a_v, a_t, a_e, b_u, b_v, b_t, b_e``.
    Its final axis preserves the one-based CUDA scratch convention: slot zero
    is unused for the gathered mass-level fields, while ``z[0]`` is zero.
    This is a preparation stage only; it does not claim to run the BEP solve.
    """
    arrays = (frc_urb, dz8w, u_phy, v_phy, th_phy, rho, p_phy)
    for name, array in zip(
            ("frc_urb", "dz8w", "u_phy", "v_phy", "th_phy", "rho", "p_phy"),
            arrays):
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(np.float32):
            raise TypeError(f"urban BEP {name} must be a float32 MetalArray")
    if not isinstance(column_x, MetalArray) or column_x.dtype != np.dtype(np.float32):
        raise TypeError("urban BEP column_x must be a float32 MetalArray")
    if not isinstance(active_cols, MetalArray) or active_cols.dtype != np.dtype(np.int32):
        raise TypeError("urban BEP active_cols must be an int32 MetalArray")
    if frc_urb.ndim != 1 or dz8w.ndim != 2:
        raise ValueError("urban BEP column inputs must be (ncol,) and (nz, ncol)")
    ncol = int(frc_urb.shape[0])
    if dz8w.shape[1] != ncol:
        raise ValueError("urban BEP dz8w column extent must match frc_urb")
    nz = int(dz8w.shape[0])
    if nz < 1 or ncol > np.iinfo(np.int32).max:
        raise ValueError("urban BEP gather requires nz >= 1 and int32 column indexing")
    expected = ((u_phy, (nz, ncol), "u_phy"),
                (v_phy, (nz, ncol), "v_phy"),
                (th_phy, (nz, ncol), "th_phy"),
                (rho, (nz, ncol), "rho"),
                (p_phy, (nz, ncol), "p_phy"),
                (column_x, (ncol, COLUMN_GATHER_FIELDS, nz + 1), "column_x"),
                (active_cols, (ncol,), "active_cols"))
    for array, shape, name in expected:
        if tuple(array.shape) != shape:
            raise ValueError(f"urban BEP {name} must have shape {shape}")
    if not ncol:
        return None
    return context.submit_kernel(
        "urban_bep_column_gather",
        {"frc_urb": frc_urb, "dz8w": dz8w, "u_phy": u_phy,
         "v_phy": v_phy, "th_phy": th_phy, "rho": rho, "p_phy": p_phy,
         "column_x": column_x, "active_cols": active_cols},
        {"nz": nz, "ncol": ncol}, exact_policy=False, grid_size=ncol,
        threads_per_group=_THREADS)


def submit_urban_bep_column_profile(
        context, class_x: MetalArray, class_ix: MetalArray,
        hi_urb: MetalArray, active_cols: MetalArray,
        urban_profile: MetalArray, urban_profile_ix: MetalArray,
        *, num_urban_hi: int):
    """Translate the column-specific ``bep_icbephi_xy`` preparation.

    The profile fields are ``hb_u, hi_urb1d, ss_urb, pb_urb`` with shape
    ``(ncol, 4, 18)``; the integer pair is ``(nzurb, error)`` per column.
    This stage does not run ``bep_param`` or the column radiation/energy solve.
    """
    for name, array, dtype in (
            ("class_x", class_x, np.float32),
            ("hi_urb", hi_urb, np.float32),
            ("class_ix", class_ix, np.int32),
            ("active_cols", active_cols, np.int32),
            ("urban_profile", urban_profile, np.float32),
            ("urban_profile_ix", urban_profile_ix, np.int32)):
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"urban BEP {name} must be a {np.dtype(dtype)} MetalArray")
    if class_x.shape != (1, CLASS_FLOATS) or class_ix.shape != (1, CLASS_INTS):
        raise ValueError("urban BEP profile requires one packed class record")
    if hi_urb.ndim != 2 or hi_urb.shape[0] != 18:
        raise ValueError("urban BEP hi_urb must have shape (18, ncol)")
    ncol = int(hi_urb.shape[1])
    if active_cols.shape != (ncol,):
        raise ValueError("urban BEP active_cols must match hi_urb columns")
    if urban_profile.shape != (ncol, URBAN_PROFILE_FIELDS, 18):
        raise ValueError("urban BEP urban_profile must have shape (ncol, 4, 18)")
    if urban_profile_ix.shape != (ncol, 2):
        raise ValueError("urban BEP urban_profile_ix must have shape (ncol, 2)")
    if not np.iinfo(np.int32).min <= int(num_urban_hi) <= np.iinfo(np.int32).max:
        raise ValueError("urban BEP num_urban_hi must fit int32")
    if ncol > np.iinfo(np.int32).max:
        raise ValueError("urban BEP profile column count exceeds int32 indexing")
    if not ncol:
        return None
    return context.submit_kernel(
        "urban_bep_column_profile",
        {"class_x": class_x, "class_ix": class_ix, "hi_urb": hi_urb,
         "active_cols": active_cols, "urban_profile": urban_profile,
         "urban_profile_ix": urban_profile_ix},
        {"num_urban_hi": int(num_urban_hi), "ncol": ncol},
        exact_policy=False, grid_size=ncol, threads_per_group=_THREADS)


def submit_urban_bep_column_params(
        context, class_x: MetalArray, class_ix: MetalArray,
        urban_profile: MetalArray, urban_profile_ix: MetalArray,
        active_cols: MetalArray, utype_urb: MetalArray,
        frc_urb: MetalArray, lp_urb: MetalArray, lb_urb: MetalArray,
        hgt_urb: MetalArray, column_param: MetalArray,
        column_param_ix: MetalArray):
    """Translate ``bep_param`` selection and per-column geometry overrides.

    Produces 140 floats per column and integer fields ``nzurban, ndu, error``.
    Output float offsets are documented in the packet handoff.
    """
    typed = (("class_x", class_x, np.float32),
             ("class_ix", class_ix, np.int32),
             ("urban_profile", urban_profile, np.float32),
             ("urban_profile_ix", urban_profile_ix, np.int32),
             ("active_cols", active_cols, np.int32),
             ("utype_urb", utype_urb, np.int32),
             ("frc_urb", frc_urb, np.float32),
             ("lp_urb", lp_urb, np.float32),
             ("lb_urb", lb_urb, np.float32),
             ("hgt_urb", hgt_urb, np.float32),
             ("column_param", column_param, np.float32),
             ("column_param_ix", column_param_ix, np.int32))
    for name, array, dtype in typed:
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"urban BEP {name} must be a {np.dtype(dtype)} MetalArray")
    if class_x.shape != (1, CLASS_FLOATS) or class_ix.shape != (1, CLASS_INTS):
        raise ValueError("urban BEP parameterization requires one packed class record")
    if urban_profile.ndim != 3 or urban_profile.shape[1:] != (4, 18):
        raise ValueError("urban BEP urban_profile must have shape (ncol, 4, 18)")
    ncol = int(urban_profile.shape[0])
    expected = ((urban_profile_ix, (ncol, 2), "urban_profile_ix"),
                (active_cols, (ncol,), "active_cols"),
                (utype_urb, (ncol,), "utype_urb"),
                (frc_urb, (ncol,), "frc_urb"),
                (lp_urb, (ncol,), "lp_urb"),
                (lb_urb, (ncol,), "lb_urb"),
                (hgt_urb, (ncol,), "hgt_urb"),
                (column_param, (ncol, COLUMN_PARAM_FLOATS), "column_param"),
                (column_param_ix, (ncol, 3), "column_param_ix"))
    for array, shape, name in expected:
        if tuple(array.shape) != shape:
            raise ValueError(f"urban BEP {name} must have shape {shape}")
    if ncol > np.iinfo(np.int32).max:
        raise ValueError("urban BEP parameter column count exceeds int32 indexing")
    if not ncol:
        return None
    resources = {name: array for name, array, _ in typed}
    return context.submit_kernel(
        "urban_bep_column_params", resources,
        {"ncol": ncol}, exact_policy=False, grid_size=ncol,
        threads_per_group=_THREADS)


def submit_urban_bep_column_views(
        context, class_x: MetalArray, column_param: MetalArray,
        column_param_ix: MetalArray, active_cols: MetalArray,
        column_view_x: MetalArray,
        column_view_ix: MetalArray):
    """Compute per-column view factors from effective ``bep_param`` values."""
    for name, array, dtype in (
            ("class_x", class_x, np.float32),
            ("column_param", column_param, np.float32),
            ("column_param_ix", column_param_ix, np.int32),
            ("active_cols", active_cols, np.int32),
            ("column_view_x", column_view_x, np.float32),
            ("column_view_ix", column_view_ix, np.int32)):
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"urban BEP {name} must be a {np.dtype(dtype)} MetalArray")
    if class_x.shape != (1, CLASS_FLOATS) or column_param.ndim != 2:
        raise ValueError("urban BEP view recompute requires one class row and column params")
    ncol = int(column_param.shape[0])
    expected = ((column_param, (ncol, COLUMN_PARAM_FLOATS), "column_param"),
                (column_param_ix, (ncol, 3), "column_param_ix"),
                (active_cols, (ncol,), "active_cols"),
                (column_view_x, (ncol, VIEW_FLOATS), "column_view_x"),
                (column_view_ix, (ncol,), "column_view_ix"))
    for array, shape, name in expected:
        if tuple(array.shape) != shape:
            raise ValueError(f"urban BEP {name} must have shape {shape}")
    if ncol > np.iinfo(np.int32).max:
        raise ValueError("urban BEP view column count exceeds int32 indexing")
    if not ncol:
        return None
    return context.submit_kernel(
        "urban_bep_column_views",
        {"class_x": class_x, "column_param": column_param,
         "column_param_ix": column_param_ix, "active_cols": active_cols,
         "column_view_x": column_view_x, "column_view_ix": column_view_ix},
        {"ncol": ncol}, exact_policy=False, grid_size=ncol,
        threads_per_group=_THREADS)


def submit_urban_bep_column_state_gather(
        context, active_cols: MetalArray, column_param_ix: MetalArray,
        trb_urb: MetalArray, tw1_urb: MetalArray, tw2_urb: MetalArray,
        tgb_urb: MetalArray, sfw1_urb: MetalArray, sfw2_urb: MetalArray,
        sfr_urb: MetalArray, sfg_urb: MetalArray,
        column_state: MetalArray, column_state_ix: MetalArray):
    """Gather persistent BEP column state into CUDA local-array order.

    The state row is ``tw[720], tg[20], tr[360], sfw[72], sfg[2], sfr[36]``.
    ``column_state_ix`` reports the carried params error or source gate 2736.
    """
    typed = (("active_cols", active_cols, np.int32),
             ("column_param_ix", column_param_ix, np.int32),
             ("trb_urb", trb_urb, np.float32),
             ("tw1_urb", tw1_urb, np.float32),
             ("tw2_urb", tw2_urb, np.float32),
             ("tgb_urb", tgb_urb, np.float32),
             ("sfw1_urb", sfw1_urb, np.float32),
             ("sfw2_urb", sfw2_urb, np.float32),
             ("sfr_urb", sfr_urb, np.float32),
             ("sfg_urb", sfg_urb, np.float32),
             ("column_state", column_state, np.float32),
             ("column_state_ix", column_state_ix, np.int32))
    for name, array, dtype in typed:
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"urban BEP {name} must be a {np.dtype(dtype)} MetalArray")
    if active_cols.ndim != 1:
        raise ValueError("urban BEP active_cols must be one-dimensional")
    ncol = int(active_cols.shape[0])
    expected = (
        (column_param_ix, (ncol, 3), "column_param_ix"),
        (trb_urb, (18, 10, 2, ncol), "trb_urb"),
        (tw1_urb, (18, 10, 2, ncol), "tw1_urb"),
        (tw2_urb, (18, 10, 2, ncol), "tw2_urb"),
        (tgb_urb, (10, 2, ncol), "tgb_urb"),
        (sfw1_urb, (18, 2, ncol), "sfw1_urb"),
        (sfw2_urb, (18, 2, ncol), "sfw2_urb"),
        (sfr_urb, (18, 2, ncol), "sfr_urb"),
        (sfg_urb, (2, ncol), "sfg_urb"),
        (column_state, (ncol, COLUMN_STATE_FLOATS), "column_state"),
        (column_state_ix, (ncol,), "column_state_ix"))
    for array, shape, name in expected:
        if tuple(array.shape) != shape:
            raise ValueError(f"urban BEP {name} must have shape {shape}")
    if ncol > np.iinfo(np.int32).max:
        raise ValueError("urban BEP state column count exceeds int32 indexing")
    if not ncol:
        return None
    resources = {name: array for name, array, _ in typed}
    return context.submit_kernel(
        "urban_bep_column_state_gather", resources, {"ncol": ncol},
        exact_policy=False, grid_size=ncol, threads_per_group=_THREADS)


def submit_urban_bep_column_state_scatter(
        context, active_cols: MetalArray, column_state_ix: MetalArray,
        column_state: MetalArray, trb_urb: MetalArray,
        tw1_urb: MetalArray, tw2_urb: MetalArray, tgb_urb: MetalArray,
        sfw1_urb: MetalArray, sfw2_urb: MetalArray,
        sfr_urb: MetalArray, sfg_urb: MetalArray):
    """Scatter successful solver state back to persistent WRF arrays."""
    typed = (("active_cols", active_cols, np.int32),
             ("column_state_ix", column_state_ix, np.int32),
             ("column_state", column_state, np.float32),
             ("trb_urb", trb_urb, np.float32),
             ("tw1_urb", tw1_urb, np.float32),
             ("tw2_urb", tw2_urb, np.float32),
             ("tgb_urb", tgb_urb, np.float32),
             ("sfw1_urb", sfw1_urb, np.float32),
             ("sfw2_urb", sfw2_urb, np.float32),
             ("sfr_urb", sfr_urb, np.float32),
             ("sfg_urb", sfg_urb, np.float32))
    for name, array, dtype in typed:
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"urban BEP {name} must be a {np.dtype(dtype)} MetalArray")
    if active_cols.ndim != 1:
        raise ValueError("urban BEP active_cols must be one-dimensional")
    ncol = int(active_cols.shape[0])
    expected = ((column_state_ix, (ncol,), "column_state_ix"),
                (column_state, (ncol, COLUMN_STATE_FLOATS), "column_state"),
                (trb_urb, (18, 10, 2, ncol), "trb_urb"),
                (tw1_urb, (18, 10, 2, ncol), "tw1_urb"),
                (tw2_urb, (18, 10, 2, ncol), "tw2_urb"),
                (tgb_urb, (10, 2, ncol), "tgb_urb"),
                (sfw1_urb, (18, 2, ncol), "sfw1_urb"),
                (sfw2_urb, (18, 2, ncol), "sfw2_urb"),
                (sfr_urb, (18, 2, ncol), "sfr_urb"),
                (sfg_urb, (2, ncol), "sfg_urb"))
    for array, shape, name in expected:
        if tuple(array.shape) != shape:
            raise ValueError(f"urban BEP {name} must have shape {shape}")
    if ncol > np.iinfo(np.int32).max:
        raise ValueError("urban BEP state column count exceeds int32 indexing")
    if not ncol:
        return None
    resources = {name: array for name, array, _ in typed}
    return context.submit_kernel(
        "urban_bep_column_state_scatter", resources, {"ncol": ncol},
        exact_policy=False, grid_size=ncol, threads_per_group=_THREADS)


def submit_urban_bep_column_interp(
        context, column_x: MetalArray, class_x: MetalArray,
        column_param_ix: MetalArray, active_cols: MetalArray,
        column_interp_x: MetalArray, column_interp_ix: MetalArray, *, nz: int):
    """Compute layer thicknesses and Urban BEP's vertical overlap means."""
    for name, array, dtype in (
            ("column_x", column_x, np.float32),
            ("class_x", class_x, np.float32),
            ("column_param_ix", column_param_ix, np.int32),
            ("active_cols", active_cols, np.int32),
            ("column_interp_x", column_interp_x, np.float32),
            ("column_interp_ix", column_interp_ix, np.int32)):
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"urban BEP {name} must be a {np.dtype(dtype)} MetalArray")
    if class_x.shape != (1, CLASS_FLOATS) or active_cols.ndim != 1:
        raise ValueError("urban BEP interpolation requires one class row and active mask")
    ncol = int(active_cols.shape[0])
    nz = int(nz)
    if nz < 1 or nz > np.iinfo(np.int32).max or ncol > np.iinfo(np.int32).max:
        raise ValueError("urban BEP interpolation dimensions must fit positive int32")
    expected = ((column_x, (ncol, COLUMN_GATHER_FIELDS, nz + 1), "column_x"),
                (column_param_ix, (ncol, 3), "column_param_ix"),
                (column_interp_x, (ncol, 6, 18), "column_interp_x"),
                (column_interp_ix, (ncol,), "column_interp_ix"))
    for array, shape, name in expected:
        if tuple(array.shape) != shape:
            raise ValueError(f"urban BEP {name} must have shape {shape}")
    if not ncol:
        return None
    return context.submit_kernel(
        "urban_bep_column_interp",
        {"column_x": column_x, "class_x": class_x,
         "column_param_ix": column_param_ix, "active_cols": active_cols,
         "column_interp_x": column_interp_x,
         "column_interp_ix": column_interp_ix},
        {"nz": nz, "ncol": ncol}, exact_policy=False, grid_size=ncol,
        threads_per_group=_THREADS)


def submit_urban_bep_column_lengths(
        context, column_x: MetalArray, class_x: MetalArray,
        column_param: MetalArray, column_param_ix: MetalArray,
        active_cols: MetalArray, column_state_ix: MetalArray,
        column_lengths: MetalArray,
        column_lengths_ix: MetalArray, *, nz: int):
    """Compute ``bep_interp_length`` urban and grid mixing lengths."""
    for name, array, dtype in (
            ("column_x", column_x, np.float32),
            ("class_x", class_x, np.float32),
            ("column_param", column_param, np.float32),
            ("column_param_ix", column_param_ix, np.int32),
            ("active_cols", active_cols, np.int32),
            ("column_state_ix", column_state_ix, np.int32),
            ("column_lengths", column_lengths, np.float32),
            ("column_lengths_ix", column_lengths_ix, np.int32)):
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"urban BEP {name} must be a {np.dtype(dtype)} MetalArray")
    if class_x.shape != (1, CLASS_FLOATS) or active_cols.ndim != 1:
        raise ValueError("urban BEP lengths require one class row and active mask")
    ncol = int(active_cols.shape[0])
    nz = int(nz)
    if nz < 1 or nz > np.iinfo(np.int32).max or ncol > np.iinfo(np.int32).max:
        raise ValueError("urban BEP length dimensions must fit positive int32")
    expected = ((column_x, (ncol, COLUMN_GATHER_FIELDS, nz + 1), "column_x"),
                (column_param, (ncol, COLUMN_PARAM_FLOATS), "column_param"),
                (column_param_ix, (ncol, 3), "column_param_ix"),
                (column_state_ix, (ncol,), "column_state_ix"),
                (column_lengths, (ncol, 2, nz), "column_lengths"),
                (column_lengths_ix, (ncol,), "column_lengths_ix"))
    for array, shape, name in expected:
        if tuple(array.shape) != shape:
            raise ValueError(f"urban BEP {name} must have shape {shape}")
    if not ncol:
        return None
    return context.submit_kernel(
        "urban_bep_column_lengths",
        {"column_x": column_x, "class_x": class_x,
         "column_param": column_param,
         "column_param_ix": column_param_ix, "active_cols": active_cols,
         "column_state_ix": column_state_ix,
         "column_lengths": column_lengths,
         "column_lengths_ix": column_lengths_ix},
        {"nz": nz, "ncol": ncol}, exact_policy=False, grid_size=ncol,
        threads_per_group=_THREADS)


def submit_urban_bep_column_meso(
        context, column_x: MetalArray, class_x: MetalArray,
        column_param: MetalArray, column_param_ix: MetalArray,
        column_building_x: MetalArray, active_cols: MetalArray,
        column_state_ix: MetalArray,
        column_meso_x: MetalArray, column_meso_ix: MetalArray, *, nz: int):
    """Translate the urban-to-mesoscale coefficient/source accumulation."""
    for name, array, dtype in (
            ("column_x", column_x, np.float32),
            ("class_x", class_x, np.float32),
            ("column_param", column_param, np.float32),
            ("column_param_ix", column_param_ix, np.int32),
            ("column_building_x", column_building_x, np.float32),
            ("active_cols", active_cols, np.int32),
            ("column_state_ix", column_state_ix, np.int32),
            ("column_meso_x", column_meso_x, np.float32),
            ("column_meso_ix", column_meso_ix, np.int32)):
        if not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype):
            raise TypeError(f"urban BEP {name} must be a {np.dtype(dtype)} MetalArray")
    if class_x.shape != (1, CLASS_FLOATS) or active_cols.ndim != 1:
        raise ValueError("urban BEP meso stage requires one class row and active mask")
    ncol = int(active_cols.shape[0])
    nz = int(nz)
    if nz < 1 or nz > np.iinfo(np.int32).max or ncol > np.iinfo(np.int32).max:
        raise ValueError("urban BEP meso dimensions must fit positive int32")
    expected = ((column_x, (ncol, COLUMN_GATHER_FIELDS, nz + 1), "column_x"),
                (column_param, (ncol, COLUMN_PARAM_FLOATS), "column_param"),
                (column_param_ix, (ncol, 3), "column_param_ix"),
                (column_building_x, (ncol, COLUMN_BUILDING_FLOATS), "column_building_x"),
                (column_state_ix, (ncol,), "column_state_ix"),
                (column_meso_x, (ncol, COLUMN_MESO_FIELDS, nz + 1), "column_meso_x"),
                (column_meso_ix, (ncol,), "column_meso_ix"))
    for array, shape, name in expected:
        if tuple(array.shape) != shape:
            raise ValueError(f"urban BEP {name} must have shape {shape}")
    if not ncol:
        return None
    return context.submit_kernel(
        "urban_bep_column_meso",
        {"column_x": column_x, "class_x": class_x,
         "column_param": column_param, "column_param_ix": column_param_ix,
         "column_building_x": column_building_x, "active_cols": active_cols,
         "column_state_ix": column_state_ix, "column_meso_x": column_meso_x,
         "column_meso_ix": column_meso_ix},
        {"nz": nz, "ncol": ncol}, exact_policy=False, grid_size=ncol,
        threads_per_group=_THREADS)


__all__ = ["submit_urban_bep_class_init", "submit_urban_bep_view_init",
           "submit_urban_bep_radiation",
           "submit_urban_bep_surface_temperature",
           "submit_urban_bep_bem_wall",
           "submit_urban_bep_column_buildings",
           "submit_urban_bep_column_gather", "submit_urban_bep_column_profile",
           "submit_urban_bep_column_params", "submit_urban_bep_column_views",
           "submit_urban_bep_column_state_gather", "submit_urban_bep_column_state_scatter",
           "submit_urban_bep_column_interp", "submit_urban_bep_column_lengths",
           "submit_urban_bep_column_meso"]
