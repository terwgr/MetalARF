"""Metal launcher boundary for WRF terrain drag kernels.

The shaders (``shaders/terrain_drag_family.metal``) are the CUDA kernels of
``woof/core/kernels/terrain_drag.cu`` transcribed statement for statement;
the pinned glibc scalar math becomes Metal's precise functions.  Mass-level
heights are formed in the shader from the interface heights, as phy_prep
forms them.  The optional diagnostic outputs are not carried.
"""

from __future__ import annotations

import numpy as np

_KMAX = 128
_NSTAT = 4


def _context(value):
    return value.buffer.context


def _checked(name, value, shape):
    from .array import MetalArray

    if not isinstance(value, MetalArray):
        raise TypeError(f"terrain drag {name} must be a MetalArray")
    if value.dtype.name != "float32" or tuple(value.shape) != tuple(shape):
        raise ValueError(f"terrain drag {name} must be float32 {shape}, got "
                         f"{value.dtype} {value.shape}")
    return value


def _upload(xp, value, shape, dtype=np.float32):
    """A contiguous device copy of ``value`` (or ``value`` itself)."""
    from .array import MetalArray

    if isinstance(value, MetalArray):
        if (value.dtype == np.dtype(dtype) and value.flags.c_contiguous
                and tuple(value.shape) == tuple(shape)):
            return value
        value = value.to_numpy()
    host = np.ascontiguousarray(np.broadcast_to(
        np.asarray(value, dtype=dtype), shape))
    return xp.asarray(host, dtype=dtype)


def _stack4(xp, planes, shape):
    if getattr(planes, "shape", None) == (_NSTAT,) + tuple(shape):
        return planes  # already stacked on the device at build time
    host = np.empty((_NSTAT,) + tuple(shape), dtype=np.float32)
    for m, plane in enumerate(planes):
        host[m] = (plane.to_numpy() if hasattr(plane, "to_numpy")
                   else np.asarray(plane, dtype=np.float32))
    return xp.asarray(host, dtype=np.float32)


def submit_topo_wind_static(state, ht, var_sso, var2d, xland, *, topo_wind):
    """``(ctopo, ctopo2, lap_hgt)`` for topo_wind 0, 1 or 2 into scratch."""
    from .array import MetalArray

    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal terrain coefficients require a Metal DomainState")
    topo_wind = int(topo_wind)
    if topo_wind not in (0, 1, 2):
        raise ValueError("topo_wind must be 0, 1, or 2")
    if topo_wind == 1 and var_sso is None:
        raise ValueError("topo_wind=1 requires VAR_SSO data")
    if topo_wind == 2 and var2d is None:
        raise ValueError("topo_wind=2 requires VAR data")
    # The typed kernel keeps a fixed resource layout; these inputs are not
    # read by the selected option and use the terrain plane as a safe dummy.
    var_sso = ht if var_sso is None else var_sso
    var2d = ht if var2d is None else var2d
    arrays = (ht, var_sso, var2d, xland)
    if not all(isinstance(value, MetalArray) for value in arrays):
        raise TypeError("terrain coefficient inputs must be MetalArray values")
    if any(value.dtype.name != "float32" for value in arrays):
        raise TypeError("terrain coefficient inputs must be float32")
    if any(value.shape != ht.shape for value in arrays) or len(ht.shape) != 2:
        raise ValueError("terrain coefficient inputs must share (ny, nx) shape")
    ny, nx = ht.shape
    if not ny or not nx:
        raise ValueError("terrain coefficient inputs must be nonempty")
    lap = state.scratch((ny, nx), "terrain_drag_lap_hgt")
    ctopo = state.scratch((ny, nx), "terrain_drag_ctopo")
    ctopo2 = state.scratch((ny, nx), "terrain_drag_ctopo2")
    state._array_module.context.submit_kernel(
        "topo_wind_static",
        {"ht": ht, "var_sso": var_sso, "var2d": var2d,
         "xland": xland, "lap_hgt": lap, "ctopo": ctopo,
         "ctopo2": ctopo2},
        {"nx": nx, "ny": ny, "topo_wind": topo_wind},
        exact_policy=False, grid_size=ny * nx)
    return ctopo, ctopo2, lap


def submit_terrain_pbl_top(dz, pblh):
    """GSL's one-based KPBL from a diagnosed BL height (SASE), int32."""
    from .array import MetalArrayModule

    nz, ny, nx = dz.shape
    _checked("dz", dz, (nz, ny, nx))
    _checked("pblh", pblh, (ny, nx))
    context = _context(dz)
    kpbl = MetalArrayModule(context).empty((ny, nx), dtype=np.int32)
    context.submit_kernel(
        "terrain_pbl_top", {"dz": dz, "pblh": pblh, "kpbl": kpbl},
        {"nz": nz, "ny": ny, "nx": nx}, exact_policy=False,
        grid_size=ny * nx)
    return kpbl


def _columns(atmosphere, nz, ny, nx, names):
    shape = (nz, ny, nx)
    source = {"u": "u", "v": "v", "t": "temperature", "qv": "qv",
              "p": "pressure", "pi": "exner", "dz": "dz",
              "p_interface": "p_interface", "zi": "z_interface"}
    out = {}
    for key in names:
        level = (nz + 1, ny, nx) if key in ("p_interface", "zi") else shape
        out[key] = _checked(source[key], atmosphere[source[key]], level)
    return out


def submit_gwdo(atmosphere, du, dv, *, var, con, oa, ol, sina, cosa,
                dx, dt, constants):
    """gwd_opt = 1: add the KIM orographic drag to ``du``/``dv`` in place."""
    from .array import MetalArrayModule

    nz, ny, nx = du.shape
    if nz > _KMAX:
        raise ValueError(f"gwd_opt = 1 holds at most {_KMAX} levels, got {nz}")
    _checked("du", du, (nz, ny, nx))
    _checked("dv", dv, (nz, ny, nx))
    context = _context(du)
    xp = MetalArrayModule(context)
    shape2 = (ny, nx)
    arrays = _columns(atmosphere, nz, ny, nx,
                      ("u", "v", "t", "qv", "p", "p_interface", "pi", "zi"))
    arrays.update(
        var=_upload(xp, var, shape2), oc1=_upload(xp, con, shape2),
        oa4=_stack4(xp, oa, shape2), ol4=_stack4(xp, ol, shape2),
        sina=_upload(xp, sina, shape2), cosa=_upload(xp, cosa, shape2),
        du=du, dv=dv)
    g, cp_, rd, ep1, pi_ = constants
    context.submit_kernel(
        "gwdo_column", arrays,
        {"dxmeter": float(dx), "deltim": float(dt), "g_": float(g),
         "cp_": float(cp_), "rd_": float(rd), "fv_": float(ep1),
         "pi_": float(pi_), "nz": nz, "ny": ny, "nx": nx, "nstat": _NSTAT},
        exact_policy=False, grid_size=ny * nx)


def submit_gwdo_gsl(atmosphere, du, dv, *, ls, ss, sina, cosa, xland, br,
                    pblh, kpbl, kpblmax, dx, dt, constants):
    """gwd_opt = 3: add the GSL drag suite to ``du``/``dv`` in place."""
    from .array import MetalArrayModule

    nz, ny, nx = du.shape
    if nz > _KMAX:
        raise ValueError(f"gwd_opt = 3 holds at most {_KMAX} levels, got {nz}")
    _checked("du", du, (nz, ny, nx))
    _checked("dv", dv, (nz, ny, nx))
    context = _context(du)
    xp = MetalArrayModule(context)
    shape2 = (ny, nx)
    arrays = _columns(
        atmosphere, nz, ny, nx,
        ("u", "v", "t", "qv", "p", "p_interface", "pi", "zi", "dz"))
    arrays.update(
        var=_upload(xp, ls["var"], shape2), oc1=_upload(xp, ls["con"], shape2),
        oa4=_stack4(xp, ls["oa"], shape2), ol4=_stack4(xp, ls["ol"], shape2),
        varss=_upload(xp, ss["var"], shape2),
        oc1ss=_upload(xp, ss["con"], shape2),
        oa4ss=_stack4(xp, ss["oa"], shape2),
        ol4ss=_stack4(xp, ss["ol"], shape2),
        sina=_upload(xp, sina, shape2), cosa=_upload(xp, cosa, shape2),
        xland=_upload(xp, xland, shape2), br=_upload(xp, br, shape2),
        hpbl=_upload(xp, pblh, shape2),
        kpbl=_upload(xp, kpbl, shape2, dtype=np.int32), du=du, dv=dv)
    g, cp_, rd, ep1, pi_ = constants
    context.submit_kernel(
        "gwdo_gsl_column", arrays,
        {"dxmeter": float(dx), "deltim": float(dt), "g": float(g),
         "cp": float(cp_), "rd": float(rd), "fv": float(ep1),
         "pi_c": float(pi_), "kpblmax": int(kpblmax), "nz": nz, "ny": ny,
         "nx": nx, "nstat": _NSTAT},
        exact_policy=False, grid_size=ny * nx)
