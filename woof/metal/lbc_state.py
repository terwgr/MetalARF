"""Resident Metal launch helpers for WRF lateral-boundary state transforms.

The shader/signature slice covers the seven pinned lbc_state entrypoints.
These helpers are source-level launchers; central lateral-BC routing and
generated registration remain coordinator integration work.
"""

from __future__ import annotations

import numpy as np

from woof.wrf_exact import ENABLED as WRF_EXACT


_FIELD_KIND = {"u": 0, "v": 1, "t": 2, "ph": 3, "mu": 4,
               "qv": 5, "w": 6}


def submit_couple_nest_field(state, field_name: str, *, out,
                            window=None, frame_width=None):
    """Couple one field into caller-owned resident storage.

    A window is ``(j0, i0, nj, ni)`` in parent mass-cell coordinates. A frame
    writes only the requested perimeter rings; all other output cells remain
    untouched. The frame/window forms retain the full parent field strides.
    """
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal nesting coupling requires a Metal DomainState")
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal nesting coupling currently refuses WRF exact arithmetic")
    target_name = {"t": "thp", "ph": "php"}.get(field_name, field_name)
    target = (state.mup.view((1, *state.mup.shape)) if field_name == "mu"
              else getattr(state, target_name, None))
    if target is None or tuple(target.shape) != tuple(out.shape):
        raise ValueError(f"invalid nesting field/output for {field_name!r}")
    if not isinstance(field_name, str) or not field_name:
        raise ValueError("nesting field name must be a nonempty string")
    kind = _FIELD_KIND.get(field_name, 7)
    nz, ny, nx = (int(n) for n in target.shape)
    mny, mnx = (int(n) for n in state.mup.shape)
    if window is not None and frame_width is not None:
        raise ValueError("coupling cannot enumerate window and frame together")
    name = "couple_nest_field"
    count = nz * ny * nx
    params = {"has_msf": int(bool(state.has_msf)),
              "thb_3d": int(state.thb.ndim == 3), "kind": kind,
              "nz": nz, "ny": ny, "nx": nx, "mny": mny, "mnx": mnx,
              "j0": 0, "i0": 0, "nj": ny, "ni": nx,
              "width": 0, "points": 0, "wrf_exact": int(WRF_EXACT)}
    if window is not None:
        j0, i0, nj, ni = (int(v) for v in window)
        if min(j0, i0, nj, ni) < 0 or j0 + nj > ny or i0 + ni > nx:
            raise ValueError("nesting window is outside the parent field")
        if not nj or not ni:
            return out
        name, count = "couple_nest_field_window", nz * nj * ni
        params.update(j0=j0, i0=i0, nj=nj, ni=ni)
    elif frame_width is not None:
        width = min(max(int(frame_width), 0), (min(ny, nx) + 1) // 2)
        if not width:
            return out
        points = _perimeter_count(ny, nx, width)
        name, count = "couple_nest_field_frame", nz * points
        params.update(width=width, points=points)
    thb, c1h, c2h, c1f, c2f = (state.thb, state.c1h, state.c2h,
                                 state.c1f, state.c2f)
    if kind == 4:
        thb, c1h, c2h, c1f, c2f = (x.view((1,)) for x in
                                     (thb, c1h, c2h, c1f, c2f))
    arrays = {"target": target, "mub2d": state.mub2d, "mup": state.mup,
              "thb": thb, "c1h": c1h, "c2h": c2h,
              "c1f": c1f, "c2f": c2f, "msft": state.msft,
              "msfu": state.msfu, "msfv": state.msfv, "out": out}
    state._array_module.context.submit_kernel(
        name, arrays, params, exact_policy=False, grid_size=count)
    return out


def _perimeter_count(ny: int, nx: int, width: int) -> int:
    """Count unique perimeter cells, with each ring corner owned once."""
    count = 0
    for d in range(width):
        rows = 2 if ny - 1 - d > d else 1
        cols = 2 if nx - 1 - d > d else 1
        count += rows * (nx - 2 * d) + cols * max(ny - 2 * d - 2, 0)
    return count


def _metal_boundary_arrays(state, boundary):
    """Bind the eight resident side value/tendency arrays in CUDA order."""
    xp = state._array_module
    arrays = []
    for side in (boundary.west, boundary.east,
                 boundary.south, boundary.north):
        arrays.extend((xp.asarray(side.value, dtype=np.float32),
                       xp.asarray(side.tendency, dtype=np.float32)))
    return arrays


def submit_state_specified_relaxation(
        state, field_name: str, tendency, boundary, *, dtbc: float,
        width: int, spec_zone: int, relax_zone: int, apply_relax: bool,
        fcx, gcx, clear_specified: bool = False, held=None,
        divide_msf: bool = False, source_field=None, source_mup=None,
        relax_sides: int = 15):
    """Apply specified and Davies relaxation directly to resident arrays.

    The kernel covers every cell so the legacy divide/add ufunc work on the
    interior is preserved without asking MetalArray to implement arithmetic.
    Relaxation itself still runs only in the selected edge zones.
    """
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal state relaxation requires a Metal DomainState")
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal state relaxation currently refuses WRF exact arithmetic")
    from woof.ingest.lateral_bc import COUPLED_SCALAR_STATE_FIELDS
    if field_name not in ({"u", "v", "w", "theta", "phi", "mu"}
                          | COUPLED_SCALAR_STATE_FIELDS):
        raise ValueError(f"unsupported Metal state boundary field {field_name!r}")
    if tendency.ndim != 3:
        raise ValueError("state boundary tendency must be a 3-D Metal array")
    nz, ny, nx = (int(n) for n in tendency.shape)
    if int(width) < int(spec_zone) + int(relax_zone):
        raise ValueError("boundary width is smaller than spec_zone + relax_zone")
    active_width = max(int(spec_zone), int(relax_zone))
    if relax_sides == 15:
        from woof.grid_requirements import boundary_axis
        if active_width < 1 or min(ny, nx) <= boundary_axis(active_width):
            raise ValueError("specified relaxation leaves no unique frame")
    else:
        for n, lo_bit, hi_bit, axis in ((ny, 1, 2, "y"),
                                        (nx, 4, 8, "x")):
            owned = int(bool(relax_sides & lo_bit)) + int(bool(relax_sides & hi_bit))
            if owned and n <= owned * active_width:
                raise ValueError(
                    f"{axis} tile window cannot hold the owned relaxation zones")
    kind = {"u": 0, "v": 1, "theta": 2, "phi": 3,
            "mu": 4, "qv": 5, "w": 6}.get(field_name, 7)
    target_name = {"theta": "thp", "phi": "php"}.get(field_name, field_name)
    source = source_field if source_field is not None else getattr(state, target_name, None)
    if source is None:
        source = state.thp if kind in (5, 7) else tendency
    u = source if field_name == "u" else tendency
    v = source if field_name == "v" else tendency
    w = source if field_name == "w" else tendency
    thp = source if field_name == "theta" else tendency
    php = source if field_name == "phi" else tendency
    scalar = source if kind in (5, 7) else tendency
    add_held = held is not None
    held = tendency if held is None else held
    mup = state.mup if source_mup is None else source_mup
    side_arrays = _metal_boundary_arrays(state, boundary)
    xp = state._array_module
    fcx = xp.asarray(fcx, dtype=np.float32)
    gcx = xp.asarray(gcx, dtype=np.float32)
    thb = state.thb
    c1h, c2h, c1f, c2f = state.c1h, state.c2h, state.c1f, state.c2f
    if kind == 4:
        thb, c1h, c2h, c1f, c2f = (x.view((1,)) for x in
                                     (thb, c1h, c2h, c1f, c2f))
    arrays = {"field_tend": tendency, "held": held, "mub2d": state.mub2d,
              "mup": mup, "u": u, "v": v, "w": w, "thp": thp,
              "thb": thb, "php": php, "scalar": scalar,
              "c1h": c1h, "c2h": c2h, "c1f": c1f, "c2f": c2f,
              "msft": state.msft, "msfu": state.msfu, "msfv": state.msfv,
              "west": side_arrays[0], "west_t": side_arrays[1],
              "east": side_arrays[2], "east_t": side_arrays[3],
              "south": side_arrays[4], "south_t": side_arrays[5],
              "north": side_arrays[6], "north_t": side_arrays[7],
              "fcx": fcx, "gcx": gcx}
    params = {"dtbc": float(dtbc), "width": int(width),
              "spec_zone": int(spec_zone), "relax_zone": int(relax_zone),
              "apply_relax": int(bool(apply_relax)),
              "clear_specified": int(bool(clear_specified)),
              "add_held": int(add_held),
              "divide_msf": int(bool(divide_msf)),
              "has_msf": int(bool(state.has_msf)),
              "thb_3d": int(state.thb.ndim == 3), "kind": kind,
              "nz": nz, "ny": ny, "nx": nx,
              "mny": int(state.mup.shape[0]), "mnx": int(state.mup.shape[1]),
              "relax_sides": int(relax_sides)}
    return state._array_module.context.submit_kernel(
        "state_specified_relaxation", arrays, params,
        exact_policy=False, grid_size=nz * ny * nx)


def submit_install_mu_boundary(state, boundary, *, dtbc: float,
                               spec_zone: int):
    """Install specified MU values and save its old perimeter in state scratch."""
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal MU boundary installation requires Metal state")
    if WRF_EXACT:
        raise NotImplementedError("Metal MU installation refuses exact mode")
    ny, nx = (int(n) for n in state.mup.shape)
    width = int(boundary.west.value.shape[-1])
    frame_count = _perimeter_count(ny, nx, int(spec_zone))
    old_frame = state.scratch((frame_count,), f"lbc_old_mup_frame_{spec_zone}")
    sides = _metal_boundary_arrays(state, boundary)
    arrays = {"mup": state.mup, "old_mup_frame": old_frame,
              "west": sides[0], "west_t": sides[1],
              "east": sides[2], "east_t": sides[3],
              "south": sides[4], "south_t": sides[5],
              "north": sides[6], "north_t": sides[7]}
    params = {"dtbc": float(dtbc), "width": width,
              "spec_zone": int(spec_zone), "ny": ny, "nx": nx,
              "frame_count": frame_count, "one": 1}
    state._array_module.context.submit_kernel(
        "install_mu_boundary", arrays, params, exact_policy=False,
        grid_size=frame_count)
    return old_frame


def submit_finalize_state_field(state, field_name: str, boundary,
                                old_mup_frame, *, dtbc: float,
                                spec_zone: int):
    """Fuse old-mass coupling, specified interpolation and current uncoupling."""
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal state finalization requires a Metal DomainState")
    if WRF_EXACT:
        raise NotImplementedError("Metal finalization refuses WRF exact mode")
    target_name = {"theta": "thp", "phi": "php"}.get(field_name, field_name)
    target = getattr(state, target_name, None)
    if target is None:
        raise ValueError(f"state has no finalization field {field_name!r}")
    nz, ny, nx = (int(n) for n in target.shape)
    kind = {"u": 0, "v": 1, "theta": 2, "phi": 3,
            "qv": 5, "w": 6}.get(field_name, 7)
    sides = _metal_boundary_arrays(state, boundary)
    arrays = {"target": target, "old_mup_frame": old_mup_frame,
              "mub2d": state.mub2d, "mup": state.mup, "thb": state.thb,
              "scalar_unused": target, "c1h": state.c1h, "c2h": state.c2h,
              "c1f": state.c1f, "c2f": state.c2f,
              "msft": state.msft, "msfu": state.msfu, "msfv": state.msfv,
              "west": sides[0], "west_t": sides[1],
              "east": sides[2], "east_t": sides[3],
              "south": sides[4], "south_t": sides[5],
              "north": sides[6], "north_t": sides[7]}
    params = {"dtbc": float(dtbc), "width": int(boundary.west.value.shape[-1]),
              "spec_zone": int(spec_zone), "has_msf": int(bool(state.has_msf)),
              "thb_3d": int(state.thb.ndim == 3), "kind": kind,
              "nz": nz, "ny": ny, "nx": nx,
              "mny": int(state.mup.shape[0]), "mnx": int(state.mup.shape[1]),
              "frame_count": _perimeter_count(*state.mup.shape, spec_zone)}
    return state._array_module.context.submit_kernel(
        "finalize_state_field", arrays, params, exact_policy=False,
        grid_size=nz * ny * nx)


def submit_uncouple_feedback_field(state, field_name: str, coupled, reg, *,
                                   spec_zone: int = 1):
    """Uncouple one already restricted feedback rectangle in place."""
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal feedback uncoupling requires a Metal DomainState")
    if WRF_EXACT:
        raise NotImplementedError("Metal feedback uncoupling refuses exact mode")
    if field_name == "mu":
        raise ValueError("feedback MU is restricted directly")
    from woof.core.nest_interp import feedback_parent_bounds
    target_name = {"t": "thp", "ph": "php"}.get(field_name, field_name)
    target = getattr(state, target_name, None)
    if target is None or tuple(coupled.shape) != tuple(target.shape):
        raise ValueError(f"invalid Metal feedback field {field_name!r}")
    i_lo, i_hi, j_lo, j_hi = feedback_parent_bounds(reg, spec_zone=spec_zone)
    ni, nj = max(0, int(i_hi)-int(i_lo)+1), max(0, int(j_hi)-int(j_lo)+1)
    if not ni or not nj:
        return target
    nz, ny, nx = (int(n) for n in target.shape)
    kind = {"u": 0, "v": 1, "t": 2, "ph": 3, "qv": 5, "w": 6}.get(field_name, 7)
    arrays = {"target": target, "coupled": coupled, "mub2d": state.mub2d,
              "mup": state.mup, "thb": state.thb,
              "c1h": state.c1h, "c2h": state.c2h,
              "c1f": state.c1f, "c2f": state.c2f,
              "msft": state.msft, "msfu": state.msfu, "msfv": state.msfv}
    params = {"has_msf": int(bool(state.has_msf)),
              "thb_3d": int(state.thb.ndim == 3), "kind": kind,
              "i_lo": int(i_lo), "i_hi": int(i_hi),
              "j_lo": int(j_lo), "j_hi": int(j_hi),
              "nz": nz, "ny": ny, "nx": nx,
              "mny": int(state.mup.shape[0]), "mnx": int(state.mup.shape[1])}
    state._array_module.context.submit_kernel(
        "uncouple_feedback_field", arrays, params, exact_policy=False,
        grid_size=nz * nj * ni)
    return target
