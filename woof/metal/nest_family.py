"""Typed Metal launchers for float-only nesting operators.

Operators needing the current runtime's unavailable int32 resources or
binary64 arithmetic are rejected explicitly below; they are not routed
through a different numerical implementation.
"""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_THREADS = 128
_I32_MAX = np.iinfo(np.int32).max


def _i32(name, value, *, minimum=0):
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"Metal nest {name} must be an integer")
    value = int(value)
    if value < minimum or value > _I32_MAX:
        raise ValueError(f"Metal nest {name} must fit the supported int32 range")
    return value


def _arrays(context, arrays, shapes):
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous):
            raise TypeError(f"Metal nest {name} must be contiguous float32 storage")
        if array.shape != shapes[name]:
            raise ValueError(
                f"Metal nest {name} shape {array.shape} != {shapes[name]}")
        if array.buffer.context is not context:
            raise ValueError("Metal nest arrays must belong to one context")


def _submit(context, name, arrays, parameters, count):
    if count <= 0:
        return None
    if count > _I32_MAX:
        raise ValueError("Metal nest extent exceeds the int32 indexing ABI")
    return context.submit_kernel(
        name, arrays, parameters, exact_policy=True, grid_size=count,
        threads_per_group=_THREADS)


def submit_blend_terrain(context, *, interpolated, fine, nk, ny, nx,
                         spec_bdy_width, blend_width,
                         domain_ny=None, domain_nx=None, j0=0, i0=0):
    """Blend parent terrain with the child edge/interior weights."""
    nk = _i32("nk", nk, minimum=1)
    ny = _i32("ny", ny, minimum=1)
    nx = _i32("nx", nx, minimum=1)
    spec_bdy_width = _i32("spec_bdy_width", spec_bdy_width)
    blend_width = _i32("blend_width", blend_width)
    domain_ny = ny if domain_ny is None else _i32("domain_ny", domain_ny, minimum=1)
    domain_nx = nx if domain_nx is None else _i32("domain_nx", domain_nx, minimum=1)
    j0 = _i32("j0", j0)
    i0 = _i32("i0", i0)
    if j0 + ny > domain_ny or i0 + nx > domain_nx:
        raise ValueError("terrain window lies outside its domain extent")
    arrays = {"interpolated": interpolated, "fine": fine}
    shapes = {"interpolated": (nk, ny, nx), "fine": (nk, ny, nx)}
    _arrays(context, arrays, shapes)
    params = {"spec_bdy_width": spec_bdy_width,
              "blend_width": blend_width, "nk": nk, "ny": ny, "nx": nx,
              "domain_ny": domain_ny, "domain_nx": domain_nx,
              "j0": j0, "i0": i0}
    return _submit(context, "nest_blend_terrain", arrays, params, nk * ny * nx)


def submit_copy_fcn(context, *, parent, child, ipos, jpos, nri, nrj,
                    spec_zone, xstag, ystag, nz, loop_i0, loop_j0,
                    loop_ni, loop_nj, parent_i0=0, parent_j0=0,
                    child_i0=0, child_j0=0):
    """Restrict child float fields to a parent loop rectangle."""
    nz = _i32("nz", nz, minimum=1)
    if len(parent.shape) != 3 or len(child.shape) != 3:
        raise ValueError("Metal nest copy_fcn expects three-dimensional fields")
    nzc, nyp, nxp = parent.shape
    nzn, nyc, nxc = child.shape
    if nzc != nz or nzn != nz:
        raise ValueError("Metal nest copy_fcn level extents differ from nz")
    ipos, jpos = _i32("ipos", ipos), _i32("jpos", jpos)
    nri, nrj = _i32("nri", nri, minimum=1), _i32("nrj", nrj, minimum=1)
    spec_zone = _i32("spec_zone", spec_zone)
    xstag, ystag = _i32("xstag", xstag), _i32("ystag", ystag)
    if xstag not in (0, 1) or ystag not in (0, 1):
        raise ValueError("xstag and ystag must be 0 or 1")
    loop_ni = _i32("loop_ni", loop_ni)
    loop_nj = _i32("loop_nj", loop_nj)
    values = (parent_i0, parent_j0, child_i0, child_j0, loop_i0, loop_j0)
    names = ("parent_i0", "parent_j0", "child_i0", "child_j0", "loop_i0", "loop_j0")
    offsets = {name: _i32(name, value) for name, value in zip(names, values)}
    arrays = {"parent": parent, "child": child}
    shapes = {"parent": (nz, nyp, nxp), "child": (nz, nyc, nxc)}
    _arrays(context, arrays, shapes)
    params = {"ipos": ipos, "jpos": jpos, "nri": nri, "nrj": nrj,
              "spec_zone": spec_zone, "xstag": xstag, "ystag": ystag,
              "nz": nz, "nyp": nyp, "nxp": nxp, "nyc": nyc, "nxc": nxc,
              **offsets, "loop_ni": loop_ni, "loop_nj": loop_nj}
    return _submit(context, "nest_copy_fcn", arrays, params,
                   nz * loop_ni * loop_nj)


def submit_copy_fcn_theta(context, *, parent, child, parent_thb, child_thb,
                          transform_child, ipos, jpos, nri, nrj, spec_zone,
                          xstag, ystag, nz, loop_i0, loop_j0, loop_ni,
                          loop_nj, parent_i0=0, parent_j0=0, child_i0=0,
                          child_j0=0):
    """Restrict theta with the source-exact WRF base-state transforms."""
    nz = _i32("nz", nz, minimum=1)
    if len(parent.shape) != 3 or len(child.shape) != 3:
        raise ValueError("Metal nest copy_fcn_theta expects 3-D fields")
    nzc, nyp, nxp = parent.shape
    nzn, nyc, nxc = child.shape
    if nzc != nz or nzn != nz:
        raise ValueError("Metal nest copy_fcn_theta level extents differ")
    if parent_thb.shape not in ((nz,), (nz, nyp, nxp)):
        raise ValueError("Metal nest parent thb must be [nz] or [nz,nyp,nxp]")
    if child_thb.shape not in ((nz,), (nz, nyc, nxc)):
        raise ValueError("Metal nest child thb must be [nz] or [nz,nyc,nxc]")
    ipos, jpos = _i32("ipos", ipos), _i32("jpos", jpos)
    nri, nrj = _i32("nri", nri, minimum=1), _i32("nrj", nrj, minimum=1)
    spec_zone = _i32("spec_zone", spec_zone)
    xstag, ystag = _i32("xstag", xstag), _i32("ystag", ystag)
    if xstag not in (0, 1) or ystag not in (0, 1):
        raise ValueError("xstag and ystag must be 0 or 1")
    loop_i0, loop_j0 = _i32("loop_i0", loop_i0), _i32("loop_j0", loop_j0)
    loop_ni = _i32("loop_ni", loop_ni)
    loop_nj = _i32("loop_nj", loop_nj)
    offsets = {name: _i32(name, value) for name, value in (
        ("parent_i0", parent_i0), ("parent_j0", parent_j0),
        ("child_i0", child_i0), ("child_j0", child_j0))}
    arrays = {"parent": parent, "child": child,
              "parent_thb": parent_thb, "child_thb": child_thb}
    shapes = {"parent": (nz, nyp, nxp), "child": (nz, nyc, nxc)}
    _arrays(context, {"parent": parent, "child": child}, shapes)
    _arrays(context, {"parent_thb": parent_thb,
                      "child_thb": child_thb},
             {"parent_thb": tuple(parent_thb.shape),
              "child_thb": tuple(child_thb.shape)})
    params = {"ipos": ipos, "jpos": jpos, "nri": nri, "nrj": nrj,
              "spec_zone": spec_zone, "xstag": xstag, "ystag": ystag,
              "nz": nz, "nyp": nyp, "nxp": nxp, "nyc": nyc, "nxc": nxc,
              **offsets, "loop_i0": loop_i0, "loop_j0": loop_j0,
              "loop_ni": loop_ni, "loop_nj": loop_nj,
              "child_thb_3d": int(child_thb.ndim == 3),
              "parent_thb_3d": int(parent_thb.ndim == 3),
              "transform_child": int(bool(transform_child))}
    return _submit(context, "nest_copy_fcn_theta", arrays, params,
                   nz * loop_ni * loop_nj)


def _feedback_extent(nz, nyp, nxp, nyc, nxc, ipos, jpos, nri, nrj,
                     spec_zone, xstag, ystag):
    nide_span = nxc - (1 if xstag else 0)
    njde_span = nyc - (1 if ystag else 0)
    ni_cells = ipos + nide_span // nri - (0 if xstag else 1) - spec_zone - (
        ipos + spec_zone) + 1
    nj_cells = jpos + njde_span // nrj - (0 if ystag else 1) - spec_zone - (
        jpos + spec_zone) + 1
    return max(0, ni_cells) * max(0, nj_cells) * nz


def _feedback_common(context, name, parent, child, *, ipos, jpos, nri, nrj,
                     spec_zone, xstag, ystag):
    if len(parent.shape) != 3 or len(child.shape) != 3:
        raise ValueError(f"Metal nest {name} expects three-dimensional fields")
    nz, nyp, nxp = parent.shape
    nzn, nyc, nxc = child.shape
    if nz != nzn:
        raise ValueError(f"Metal nest {name} level extents differ")
    values = {"ipos": ipos, "jpos": jpos, "nri": nri, "nrj": nrj,
              "spec_zone": spec_zone, "xstag": xstag, "ystag": ystag}
    q = {key: _i32(key, val, minimum=(1 if key in ("nri", "nrj") else 0))
         for key, val in values.items()}
    if q["xstag"] not in (0, 1) or q["ystag"] not in (0, 1):
        raise ValueError("xstag and ystag must be 0 or 1")
    arrays = {"parent": parent, "child": child}
    _arrays(context, arrays, {"parent": (nz, nyp, nxp),
                              "child": (nz, nyc, nxc)})
    params = {**q, "nz": nz, "nyp": nyp, "nxp": nxp,
              "nyc": nyc, "nxc": nxc}
    count = _feedback_extent(nz, nyp, nxp, nyc, nxc, q["ipos"], q["jpos"],
                             q["nri"], q["nrj"], q["spec_zone"],
                             q["xstag"], q["ystag"])
    return arrays, params, count


def submit_copy_fcnm(context, *, parent, child, ipos, jpos, nri, nrj,
                     spec_zone, xstag, ystag):
    """Apply the WRF masked-float one-child-point restriction."""
    arrays, params, count = _feedback_common(
        context, "copy_fcnm", parent, child, ipos=ipos, jpos=jpos,
        nri=nri, nrj=nrj, spec_zone=spec_zone, xstag=xstag, ystag=ystag)
    return _submit(context, "nest_copy_fcnm", arrays, params, count)


def submit_smooth_j(context, *, parent, scratch, mode, xnu, i0, j0,
                    niw, njw, nz, nyp, nxp):
    """Run the first, J-axis feedback smoothing pass and copy its ring."""
    q = {name: _i32(name, value, minimum=(1 if name in ("nz", "nyp", "nxp") else 0))
         for name, value in {"mode": mode, "i0": i0, "j0": j0, "niw": niw,
                             "njw": njw, "nz": nz, "nyp": nyp,
                             "nxp": nxp}.items()}
    if q["mode"] not in (0, 1) or q["niw"] == 0 or q["njw"] == 0:
        raise ValueError("smoother mode must be 0/1 and its window nonempty")
    if (q["i0"] < 1 or q["j0"] < 1
            or q["i0"] + q["niw"] >= q["nxp"]
            or q["j0"] + q["njw"] >= q["nyp"]):
        raise ValueError("J-pass window and one-cell ring must fit parent extents")
    arrays = {"parent": parent, "scratch": scratch}
    shapes = {"parent": (q["nz"], q["nyp"], q["nxp"]),
              "scratch": (q["nz"], q["nyp"], q["nxp"])}
    _arrays(context, arrays, shapes)
    params = {"mode": q["mode"], "xnu": float(np.float32(xnu)),
              **{k: q[k] for k in ("i0", "j0", "niw", "njw", "nz", "nyp", "nxp")}}
    return _submit(context, "nest_smooth_j", arrays, params,
                   q["nz"] * (q["niw"] + 2) * (q["njw"] + 2))


def submit_smooth_i(context, *, parent, scratch, mode, xnu, i0, j0,
                    niw, njw, nz, nyp, nxp):
    """Run the second, I-axis feedback smoothing pass in place."""
    q = {name: _i32(name, value, minimum=(1 if name in ("nz", "nyp", "nxp") else 0))
         for name, value in {"mode": mode, "i0": i0, "j0": j0, "niw": niw,
                             "njw": njw, "nz": nz, "nyp": nyp,
                             "nxp": nxp}.items()}
    if q["mode"] not in (0, 1) or q["niw"] == 0 or q["njw"] == 0:
        raise ValueError("smoother mode must be 0/1 and its window nonempty")
    if (q["i0"] < 1 or q["j0"] < 0
            or q["i0"] + q["niw"] >= q["nxp"]
            or q["j0"] + q["njw"] > q["nyp"]):
        raise ValueError("I-pass window and neighbors must fit parent extents")
    arrays = {"parent": parent, "scratch": scratch}
    shapes = {"parent": (q["nz"], q["nyp"], q["nxp"]),
              "scratch": (q["nz"], q["nyp"], q["nxp"])}
    _arrays(context, arrays, shapes)
    params = {"mode": q["mode"], "xnu": float(np.float32(xnu)),
              **{k: q[k] for k in ("i0", "j0", "niw", "njw", "nz", "nyp", "nxp")}}
    return _submit(context, "nest_smooth_i", arrays, params,
                   q["nz"] * q["niw"] * q["njw"])


def _unsupported(name, reason):
    raise NotImplementedError(f"Metal nest {name} is unavailable: {reason}")


def _nest_geometry(context, reg):
    dev = reg._device
    if not dev or reg._binding != "manifest":
        raise RuntimeError(
            "Metal nest geometry must be bound through the domain manifest "
            "before interpolation; a lazy default upload is not supported")
    arrays = {"ci_map_a": dev["ci"], "ip_map_a": dev["ip"],
              "cj_map_a": dev["cj"], "jp_map_a": dev["jp"],
              "xig_a": dev["xig"], "xjg_a": dev["xjg"]}
    for name, array in arrays.items():
        dtype = np.dtype(np.int32 if "map" in name else np.float32)
        if (not isinstance(array, MetalArray) or array.dtype != dtype
                or not array.flags.c_contiguous
                or array.buffer.context is not context):
            raise TypeError(f"Metal nest geometry {name} must be contiguous "
                            f"{dtype} storage in the operand context")
    if (arrays["ci_map_a"].shape != (reg.nxc,)
            or arrays["ip_map_a"].shape != (reg.nxc,)
            or arrays["cj_map_a"].shape != (reg.nyc,)
            or arrays["jp_map_a"].shape != (reg.nyc,)
            or arrays["xig_a"].shape != (reg.nri,)
            or arrays["xjg_a"].shape != (reg.nrj,)):
        raise ValueError("Metal nest geometry tables do not match registration")
    return arrays


def submit_sint(context, *, parent, child, reg, nz):
    """Interpolate a parent field onto a registered child grid."""
    nz = _i32("nz", nz, minimum=1)
    if parent.ndim != 3 or child.ndim != 3:
        raise ValueError("Metal nest SINT expects three-dimensional fields")
    _, nyp, nxp = parent.shape
    arrays = {"parent_a": parent, "child_a": child,
              **_nest_geometry(context, reg)}
    shapes = {"parent_a": (nz, nyp, nxp),
              "child_a": (nz, reg.nyc, reg.nxc)}
    _arrays(context, {k: v for k, v in arrays.items()
                      if k in ("parent_a", "child_a")}, shapes)
    for name, arr in arrays.items():
        if name not in shapes and arr.buffer.context is not context:
            raise ValueError("Metal nest geometry belongs to another context")
    params = {"nz": nz, "nyc": reg.nyc, "nxc": reg.nxc,
              "nyp": nyp, "nxp": nxp}
    return _submit(context, "nest_sint_fragment", arrays, params,
                   nz * reg.nyc * reg.nxc)


def submit_bdy_interp1(context, *, parent, child, value, tendency, reg,
                       cdt, side, sz):
    """Interpolate one side's values and tendencies for the parent step."""
    side = _i32("side", side)
    if side > 3:
        raise ValueError("boundary side index must be in [0, 3]")
    nz = _i32("nz", parent.shape[0], minimum=1)
    sz = _i32("sz", sz, minimum=1)
    if not np.isfinite(cdt) or cdt <= 0:
        raise ValueError("boundary parent timestep must be positive")
    nlong = reg.nyc if side < 2 else reg.nxc
    arrays = {"parent_a": parent, "child_a": child,
              "value_a": value, "tendency_a": tendency,
              **_nest_geometry(context, reg)}
    table_shape = ((nz, reg.nyc, sz) if side < 2
                   else (nz, sz, reg.nxc))
    shapes = {"parent_a": (nz, reg.nyp, reg.nxp),
              "child_a": (nz, reg.nyc, reg.nxc),
              "value_a": table_shape,
              "tendency_a": table_shape}
    _arrays(context, {k: v for k, v in arrays.items() if k in shapes}, shapes)
    for name, arr in arrays.items():
        if name not in shapes and arr.buffer.context is not context:
            raise ValueError("Metal nest geometry belongs to another context")
    params = {"cdt": float(np.float32(cdt)), "side": side, "sz": sz,
              "nz": nz, "nyc": reg.nyc, "nxc": reg.nxc,
              "nyp": reg.nyp, "nxp": reg.nxp, "nlong": nlong}
    return _submit(context, "nest_bdy_interp1_fragment", arrays, params,
                   nz * sz * nlong)


def submit_bdy_interp1_all_sides(*args, **kwargs):
    """The fragment exposes one side per dispatch; callers loop sides."""
    _unsupported("bdy_interp1_all_sides", "the exact ABI exports one-side "
                 "dispatches; use submit_bdy_interp1 for each selected side")


def submit_rect_gather_f32(context, *, source, packed, j0, i0, nz, ny, nx):
    """Pack one rectangular region from a contiguous three-dimensional table."""
    dims = {name: _i32(name, value, minimum=1)
            for name, value in {"nz": nz, "ny": ny, "nx": nx}.items()}
    if len(source.shape) != 3:
        raise ValueError("rectangle gather source must be three-dimensional")
    _, source_ny, source_nx = source.shape
    j0, i0 = _i32("j0", j0), _i32("i0", i0)
    if j0 + dims["ny"] > source_ny or i0 + dims["nx"] > source_nx:
        raise ValueError(
            f"rectangle gather ({j0},{i0},{dims['ny']},{dims['nx']}) "
            f"lies outside source extents {(source_ny, source_nx)}")
    arrays = {"source_a": source, "packed_a": packed}
    _arrays(context, arrays, {"source_a": (dims["nz"], source_ny, source_nx),
                              "packed_a": (dims["nz"], dims["ny"], dims["nx"])})
    params = {"j0": j0, "i0": i0, **dims,
              "source_ny": source_ny, "source_nx": source_nx}
    return _submit(context, "nest_rect_gather_f32", arrays, params,
                   dims["nz"] * dims["ny"] * dims["nx"])


def submit_rect_scatter_f32(context, *, packed, target, j0, i0, nz, ny, nx):
    """Scatter one packed rectangle into a contiguous three-dimensional table."""
    dims = {name: _i32(name, value, minimum=1)
            for name, value in {"nz": nz, "ny": ny, "nx": nx}.items()}
    if len(target.shape) != 3:
        raise ValueError("rectangle scatter target must be three-dimensional")
    _, target_ny, target_nx = target.shape
    j0, i0 = _i32("j0", j0), _i32("i0", i0)
    if j0 + dims["ny"] > target_ny or i0 + dims["nx"] > target_nx:
        raise ValueError("rectangle scatter lies outside target extents")
    arrays = {"packed_a": packed, "target_a": target}
    _arrays(context, arrays, {"packed_a": (dims["nz"], dims["ny"], dims["nx"]),
                              "target_a": (dims["nz"], target_ny, target_nx)})
    params = {"j0": j0, "i0": i0, **dims,
              "target_ny": target_ny, "target_nx": target_nx}
    return _submit(context, "nest_rect_scatter_f32", arrays, params,
                   dims["nz"] * dims["ny"] * dims["nx"])


def submit_adjust_tempqv(*args, **kwargs):
    """Refuse the FP64 temperature/moisture correction."""
    _unsupported("adjust_tempqv", "the CUDA kernel evaluates pressure, "
                 "pow, exp, RH conservation and theta corrections in "
                 "binary64; Metal/runtime support for faithful binary64 "
                 "math and transcendentals is a prerequisite")


def submit_copy_fcni(*args, **kwargs):
    """Refuse integer feedback until the typed ABI supports int32 buffers."""
    _unsupported("copy_fcni", "both source and destination are int32 fields, "
                 "but the current typed Metal runtime accepts only float32 "
                 "resources")
