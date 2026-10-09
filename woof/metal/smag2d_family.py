"""Typed Metal launchers for the WRF Smagorinsky kernel family.

Launchers preserve device buffer ownership. CPU arrays and WRF exact mode are
refused; there is no numerical fallback in this module.
"""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from ..wrf_exact import ENABLED as WRF_EXACT


def _arrays(**arrays):
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous):
            raise TypeError(f"{name} must be a contiguous FP32 MetalArray")
    return arrays


def _shape(array, expected, name):
    if array.shape != expected:
        raise ValueError(f"{name} shape {array.shape} != expected {expected}")


def _submit(entrypoint, arrays, parameters, count):
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal smag2d flat-oracle kernels support ordinary FP32 only")
    if count <= 0:
        return None
    context = next(iter(arrays.values())).buffer.context
    return context.submit_kernel(entrypoint, arrays, parameters,
                                 exact_policy=False, grid_size=count)


def submit_zero_boundary_strips(buf, cfg, width, *, stag_high_extra=0):
    """Zero WRF-excluded lateral strips over the last two buffer axes.

    Leading dimensions are treated as independent planes, matching the
    ``buf[..., ...]`` assignments in dycore._zero_open_strips.
    """
    flat_buf = buf.view((buf.size,), byte_offset=buf.byte_offset)
    arrays = _arrays(buf=flat_buf)
    if buf.ndim < 2:
        raise ValueError("boundary-strip buffers must have at least two axes")
    if width < 0 or stag_high_extra < 0:
        raise ValueError("boundary-strip widths cannot be negative")
    if buf.size > np.iinfo(np.int32).max:
        raise OverflowError("boundary-strip buffer exceeds the int32 ABI")
    ny_buf, nx_buf = buf.shape[-2:]
    if nx_buf not in (cfg.nx, cfg.nx + 1):
        raise ValueError("boundary-strip x extent is not mass or u-staggered")
    if ny_buf not in (cfg.ny, cfg.ny + 1):
        raise ValueError("boundary-strip y extent is not mass or v-staggered")
    x_hi = int(width + (stag_high_extra if nx_buf == cfg.nx + 1 else 0))
    y_hi = int(width + (stag_high_extra if ny_buf == cfg.ny + 1 else 0))
    params = {
        "size": buf.size, "nx": nx_buf, "ny": ny_buf,
        "width": int(width), "x_hi": x_hi, "y_hi": y_hi,
        "boundary_x": int(bool(cfg.open_x or getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
        "boundary_y": int(bool(cfg.open_y or getattr(cfg, "specified", False)
                                or getattr(cfg, "nested", False))),
    }
    return _submit("smag_zero_boundary_strips", arrays, params, buf.size)


def submit_smag2d_km(u, v, xkmh, xkhh, *, rdx, rdy, mlen, c_s,
                      prandtl):
    nz, ny, nx = xkmh.shape
    arrays = _arrays(u=u, v=v, xkmh=xkmh, xkhh=xkhh)
    _shape(u, (nz, ny, nx + 1), "u")
    _shape(v, (nz, ny + 1, nx), "v")
    _shape(xkhh, (nz, ny, nx), "xkhh")
    params = {"rdx": float(np.float32(rdx)), "rdy": float(np.float32(rdy)),
              "mlen": float(np.float32(mlen)), "c_s": float(np.float32(c_s)),
              "prandtl": float(np.float32(prandtl)),
              "nz": nz, "ny": ny, "nx": nx}
    return _submit("smag2d_km", arrays, params, nz * ny * nx)


def _submit_hd(entrypoint, field_name, field, xk, mut, c1, c2, tend,
               *, rdx, rdy, open_axis=None):
    nz, ny, nx = xk.shape
    arrays = _arrays(**{field_name: field, "xk": xk, "mut": mut,
                        "c1": c1, "c2": c2, "tend": tend})
    _shape(xk, (nz, ny, nx), "xk")
    _shape(mut, (ny, nx), "mut")
    c_levels = nz + 1 if entrypoint == "smag_hd_w" else nz
    _shape(c1, (c_levels,), "c1")
    _shape(c2, (c_levels,), "c2")
    params = {"rdx": float(np.float32(rdx)), "rdy": float(np.float32(rdy)),
              "nz": nz, "ny": ny, "nx": nx}
    if open_axis is not None:
        key, enabled = open_axis
        params[key] = int(bool(enabled))
    return _submit(entrypoint, arrays, params, tend.size)


def submit_smag_hd_s(f, xk, mut, c1, c2, tend, *, rdx, rdy):
    nz, ny, nx = xk.shape
    _shape(f, (nz, ny, nx), "f")
    _shape(tend, (nz, ny, nx), "tend")
    return _submit_hd("smag_hd_s", "f", f, xk, mut, c1, c2, tend,
                      rdx=rdx, rdy=rdy)


def submit_smag_hd_u(u, xk, mut, c1, c2, tend, *, rdx, rdy,
                     open_x=False):
    nz, ny, nx = xk.shape
    _shape(u, (nz, ny, nx + 1), "u")
    _shape(tend, (nz, ny, nx + 1), "tend")
    return _submit_hd("smag_hd_u", "u", u, xk, mut, c1, c2, tend,
                      rdx=rdx, rdy=rdy, open_axis=("open_x", open_x))


def submit_smag_hd_v(v, xk, mut, c1, c2, tend, *, rdx, rdy,
                     open_y=False):
    nz, ny, nx = xk.shape
    _shape(v, (nz, ny + 1, nx), "v")
    _shape(tend, (nz, ny + 1, nx), "tend")
    return _submit_hd("smag_hd_v", "v", v, xk, mut, c1, c2, tend,
                      rdx=rdx, rdy=rdy, open_axis=("open_y", open_y))


def submit_smag_hd_w(w, xk, mut, c1, c2, tend, *, rdx, rdy):
    nz, ny, nx = xk.shape
    _shape(w, (nz + 1, ny, nx), "w")
    _shape(tend, (nz + 1, ny, nx), "tend")
    return _submit_hd("smag_hd_w", "w", w, xk, mut, c1, c2, tend,
                      rdx=rdx, rdy=rdy)


def _wrf_common(state, cfg, *, time_t):
    suffix = "0" if time_t else ""
    moist = state.qv is not None
    qv = getattr(state, "qv" + suffix) if moist else state.alt
    arrays = {
        "u": getattr(state, "u" + suffix),
        "v": getattr(state, "v" + suffix),
        "w": getattr(state, "w" + suffix),
        "php": getattr(state, "php" + suffix), "phb": state.phb,
        "alt": state.alt, "qv": qv, "msft": state.msft,
        "msfu": state.msfu, "msfv": state.msfv,
        "fnm": state.fnm, "fnp": state.fnp, "dn": state.dn,
        "dnw": state.dnw,
    }
    # The mass-grid dimensions come from ``alt``.  ``u`` is already
    # x-staggered, so using its last extent as mass ``nx`` would demand an
    # extra cell and reject the correct WRF shape (nz, ny, nx + 1).
    nz, ny, nx = arrays["alt"].shape
    shapes = {
        "u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
        "w": (nz + 1, ny, nx), "php": (nz + 1, ny, nx),
        "alt": (nz, ny, nx), "qv": (nz, ny, nx),
        "msft": (ny, nx), "msfu": (ny, nx + 1),
        "msfv": (ny + 1, nx), "fnm": (nz,), "fnp": (nz,),
        "dn": (nz,), "dnw": (nz,),
    }
    arrays = _arrays(**arrays)
    for name, shape in shapes.items():
        _shape(arrays[name], shape, name)
    phb3d = int(state.phb.ndim == 3)
    _shape(state.phb, (nz + 1, ny, nx) if phb3d else (nz + 1,), "phb")
    boundary_forced = bool(getattr(cfg, "specified", False)
                           or getattr(cfg, "nested", False))
    boundary_x = int(bool(cfg.open_x or boundary_forced))
    boundary_y = int(bool(cfg.open_y or boundary_forced))
    params = {
        "rdx": float(np.float32(1.0 / cfg.dx)),
        "rdy": float(np.float32(1.0 / cfg.dy)),
        "dx": float(np.float32(cfg.dx)), "dy": float(np.float32(cfg.dy)),
        "cf1": float(np.float32(state.cf1)),
        "cf2": float(np.float32(state.cf2)),
        "cf3": float(np.float32(state.cf3)), "moist": int(moist),
        "nz": nz, "ny": ny, "nx": nx, "phb3d": phb3d,
        "boundary_x": boundary_x, "boundary_y": boundary_y,
    }
    return arrays, params, (nz, ny, nx)


def submit_wrf_smag_deform(state, cfg, d11, d22, d12, *, time_t=False):
    """Submit metric-aware WRF D11/D22/D12 into caller-owned scratch."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal wrf_smag_deform exact round-to-nearest path is not ported")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    for name, out in (("d11", d11), ("d22", d22), ("d12", d12)):
        _arrays(**{name: out})
        _shape(out, (nz, ny, nx), name)
        arrays[name] = out
    if nz < 3:
        raise NotImplementedError("WRF deformation interpolation requires nz >= 3")
    return _submit("wrf_smag_deform", arrays, params, nz * ny * nx)


def submit_wrf_smag2d_km(state, cfg, d11, d22, d12, xkmh, xkhh, *,
                           c_s, prandtl=1.0 / 3.0, time_t=False):
    """Submit WRF's terrain-aware 2-D mixing-length reduction."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal wrf_smag2d_km exact round-to-nearest path is not ported")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    for name, array in (("d11", d11), ("d22", d22), ("d12", d12),
                        ("xkmh", xkmh), ("xkhh", xkhh)):
        _arrays(**{name: array})
        _shape(array, (nz, ny, nx), name)
        arrays[name] = array
    if nz < 3:
        raise NotImplementedError("WRF smag2d mixing length requires nz >= 3")
    params.update(c_s=float(np.float32(c_s)),
                  prandtl=float(np.float32(prandtl)))
    return _submit("wrf_smag2d_km", arrays, params, nz * ny * nx)


def submit_wrf_calc_n2(state, cfg, bn2, *, time_t=False):
    """Submit the WRF buoyancy-frequency input used by alternate closures.

    The Python route is ready for the corresponding source/signature
    fragment; until that is integrated the runtime reports the missing
    Metal function instead of falling through to the CUDA dispatcher.
    """
    if WRF_EXACT:
        raise NotImplementedError("Metal wrf_calc_n2 exact path is not ported")
    if state.qv is not None:
        raise NotImplementedError(
            "Metal wrf_calc_n2 source fragment currently admits dry profiles")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    suffix = "0" if time_t else ""
    arrays.update({
        "thp": getattr(state, "thp" + suffix), "thb": state.thb,
        "p": state.p, "bn2": bn2,
    })
    params.update(thb3d=int(state.thb.ndim == 3))
    return _submit("wrf_calc_n2", arrays, params, nz * ny * nx)


def submit_wrf_smag3d_km(state, cfg, d11, d22, d12, bn2,
                         xkmh, xkhh, xkmv, xkhv, *, time_t=False):
    """Submit the WRF full-deformation 3-D Smagorinsky coefficient stage."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal wrf_smag3d_km exact path is not ported")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    for name, array in (("d11", d11), ("d22", d22), ("d12", d12),
                        ("bn2", bn2), ("xkmh", xkmh), ("xkhh", xkhh),
                        ("xkmv", xkmv), ("xkhv", xkhv)):
        _arrays(**{name: array})
        _shape(array, (nz, ny, nx), name)
        arrays[name] = array
    params.update(c_s=float(np.float32(cfg.c_s)),
                  prandtl=float(np.float32(1.0 / 3.0)),
                  dt=float(np.float32(cfg.dt)),
                  mix_upper_bound=float(np.float32(cfg.mix_upper_bound)),
                  isotropic=int(cfg.mix_isotropic))
    return _submit("wrf_smag3d_km", arrays, params, nz * ny * nx)


def submit_wrf_tke_km(state, cfg, bn2, xkmh, xkhh, xkmv, xkhv,
                      *, seed, time_t=False):
    """Submit the WRF prognostic TKE exchange-coefficient stage."""
    if WRF_EXACT:
        raise NotImplementedError("Metal wrf_tke_km exact path is not ported")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    suffix = "0" if time_t else ""
    arrays.update({
        "thp": getattr(state, "thp" + suffix), "thb": state.thb,
        "p": state.p, "tke": getattr(state, "tke" + suffix), "bn2": bn2,
        "xkmh": xkmh, "xkhh": xkhh,
        "xkmv": xkmv, "xkhv": xkhv,
    })
    for name, array in arrays.items():
        _arrays(**{name: array})
    params.update(thb3d=int(state.thb.ndim == 3),
                  c_k=float(np.float32(cfg.c_k)),
                  prandtl=float(np.float32(1.0 / 3.0)),
                  dt=float(np.float32(cfg.dt)),
                  mix_upper_bound=float(np.float32(cfg.mix_upper_bound)),
                  isotropic=int(cfg.mix_isotropic),
                  seed=float(np.float32(seed)))
    return _submit("wrf_tke_km", arrays, params, nz * ny * nx)


def submit_wrf_tke_rhs(state, cfg, bn2, d11, d22, d12, xkmh, xkmv,
                       xkhv, mut, ustm, hfx, rtke, budget_terms, *,
                       has_ustm, has_hfx, budget_on, time_t=False):
    """Submit the WRF once-per-step TKE source and positivity limiter."""
    if WRF_EXACT:
        raise NotImplementedError("Metal wrf_tke_rhs exact path is not ported")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    suffix = "0" if time_t else ""
    arrays.update({
        "thp": getattr(state, "thp" + suffix), "thb": state.thb,
        "tke": getattr(state, "tke" + suffix), "bn2": bn2,
        "d11": d11, "d22": d22, "d12": d12,
        "xkmh": xkmh, "xkmv": xkmv, "xkhv": xkhv,
        "mut": mut, "c1": state.c1h, "c2": state.c2h,
        "ustm": ustm, "hfx": hfx, "rtke": rtke,
        "shear": budget_terms[0], "buoyancy": budget_terms[1],
        "dissipation": budget_terms[2], "limiter": budget_terms[3],
    })
    for name, array in arrays.items():
        _arrays(**{name: array})
    params.update(thb3d=int(state.thb.ndim == 3),
                  has_ustm=int(has_ustm), has_hfx=int(has_hfx),
                  c_k=float(np.float32(cfg.c_k)),
                  dt=float(np.float32(cfg.dt)),
                  drag=float(np.float32(cfg.tke_drag_coefficient)),
                  heat=float(np.float32(cfg.tke_heat_flux)),
                  isfflx=int(cfg.isfflx), budget_on=int(budget_on))
    return _submit("wrf_tke_rhs", arrays, params, nz * ny * nx)


def submit_wrf_smag_km_bc(xkmh, xkhh, *, boundary_x, boundary_y):
    """Preserve the cold-zero logical outer K row on physical boundaries."""
    if WRF_EXACT:
        raise NotImplementedError("Metal wrf_smag_km_bc has no exact policy")
    nz, ny, nx = xkmh.shape
    arrays = _arrays(xkmh=xkmh, xkhh=xkhh)
    _shape(xkhh, (nz, ny, nx), "xkhh")
    params = {"nz": nz, "ny": ny, "nx": nx,
              "boundary_x": int(bool(boundary_x)),
              "boundary_y": int(bool(boundary_y))}
    return _submit("wrf_smag_km_bc", arrays, params, nz * ny * nx)


def _submit_wrf_smag_hd(state, cfg, km, tend, *, time_t, stagger,
                        d11=None, d22=None, d12=None):
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal WRF stress exact round-to-nearest path is not ported")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    for name, array in (("km", km), ("tend", tend)):
        arrays[name] = _arrays(**{name: array})[name]
    _shape(km, (nz, ny, nx), "km")
    if stagger == "x":
        _shape(tend, (nz, ny, nx + 1), "tend")
        extras = {"d11": d11, "d12": d12}
        entry = "wrf_smag_hd_u"
    else:
        _shape(tend, (nz, ny + 1, nx), "tend")
        extras = {"d22": d22, "d12": d12}
        entry = "wrf_smag_hd_v"
    for name, array in extras.items():
        arrays[name] = _arrays(**{name: array})[name]
        _shape(array, (nz, ny, nx), name)
    if nz < 3:
        raise NotImplementedError("WRF stress interpolation requires nz >= 3")
    return _submit(entry, arrays, params, tend.size)


def submit_wrf_smag_hd_u(state, cfg, km, d11, d12, tend, *, time_t=False):
    """Submit the metric-aware u stress-divergence branch."""
    return _submit_wrf_smag_hd(state, cfg, km, tend, time_t=time_t,
                               stagger="x", d11=d11, d12=d12)


def submit_wrf_smag_hd_v(state, cfg, km, d22, d12, tend, *, time_t=False):
    """Submit the metric-aware v stress-divergence branch."""
    return _submit_wrf_smag_hd(state, cfg, km, tend, time_t=time_t,
                               stagger="y", d22=d22, d12=d12)


def submit_wrf_smag_vd(state, cfg, km, tend, *, stagger, time_t=False):
    """Submit one metric-aware vertical momentum diffusion component."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal WRF vertical stress exact arithmetic is not ported")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    arrays["km"] = _arrays(km=km)["km"]
    _shape(km, (nz, ny, nx), "km")
    if stagger == "u":
        name, shape = "wrf_smag_vd_u", (nz, ny, nx + 1)
    elif stagger == "v":
        name, shape = "wrf_smag_vd_v", (nz, ny + 1, nx)
    elif stagger == "w":
        name, shape = "wrf_smag_vd_w", (nz + 1, ny, nx)
    else:
        raise ValueError(f"unknown vertical Smagorinsky stagger {stagger!r}")
    arrays["tend"] = _arrays(tend=tend)["tend"]
    _shape(tend, shape, "tend")
    return _submit(name, arrays, params, tend.size)


def submit_wrf_smag_vd_scalar(state, cfg, field, khv, tend, *,
                              full_theta=False, time_t=False):
    """Submit the registered vertical scalar diffusion kernel."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal WRF vertical scalar diffusion has no exact policy")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    arrays.update(var=_arrays(var=field)["var"], thb=_arrays(thb=state.thb)["thb"],
                  khv=_arrays(khv=khv)["khv"], tend=_arrays(tend=tend)["tend"])
    _shape(field, (nz, ny, nx), "var")
    _shape(state.thb, (nz,) if state.thb.ndim == 1 else (nz, ny, nx), "thb")
    _shape(khv, (nz, ny, nx), "khv")
    _shape(tend, (nz, ny, nx), "tend")
    params.update(full_theta=int(bool(full_theta)), thb3d=int(state.thb.ndim == 3))
    return _submit("wrf_smag_vd_s", arrays, params, tend.size)


def submit_bound_tke(tke, upper_bound):
    flat = tke.view((tke.size,))
    arrays = _arrays(tke=flat)
    return _submit("bound_tke", arrays,
                   {"n": tke.size, "upper": float(np.float32(upper_bound))},
                   tke.size)


def submit_wrf_smag_surface(state, cfg, *, kind, time_t=False, tend=None,
                             ustm=None, active=False, cd0=0.0,
                             hfx=None, qfx=None, rth=None, rqv=None,
                             apply_heat=False, apply_moist=False,
                             heat_flux=0.0, write_hfx=False):
    """Submit one isfflx-selected WRF lower-boundary flux kernel."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal WRF Smagorinsky surface exact expression paths are not ported")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    if kind in ("u", "v"):
        arrays["ustm"] = _arrays(ustm=ustm)["ustm"]
        _shape(ustm, (ny, nx), "ustm")
        arrays["tend"] = _arrays(tend=tend)["tend"]
        _shape(tend, (nz, ny, nx + 1) if kind == "u" else (nz, ny + 1, nx), "tend")
        name = f"wrf_smag_surface_{kind}"
        params["active"] = int(bool(active))
    elif kind in ("u_cd0", "v_cd0"):
        arrays["tend"] = _arrays(tend=tend)["tend"]
        _shape(tend, (nz, ny, nx + 1) if kind == "u_cd0" else (nz, ny + 1, nx), "tend")
        name = f"wrf_smag_surface_{kind}"
        params["cd0"] = float(np.float32(cd0))
    elif kind == "scalars":
        for name, array, shape in (
                ("hfx", hfx, (ny, nx)), ("qfx", qfx, (ny, nx)),
                ("rth", rth, (nz, ny, nx)), ("rqv", rqv, (nz, ny, nx))):
            arrays[name] = _arrays(**{name: array})[name]
            _shape(array, shape, name)
        name = "wrf_smag_surface_scalars"
        params.update(apply_heat=int(bool(apply_heat)),
                      apply_moist=int(bool(apply_moist)))
    elif kind == "heat_const":
        for name, array, shape in (("hfx", hfx, (ny, nx)),
                                    ("rth", rth, (nz, ny, nx))):
            arrays[name] = _arrays(**{name: array})[name]
            _shape(array, shape, name)
        name = "wrf_smag_surface_heat_const"
        params.update(heat_flux=float(np.float32(heat_flux)),
                      write_hfx=int(bool(write_hfx)))
    else:
        raise ValueError(f"unknown WRF Smagorinsky surface mode {kind!r}")
    return _submit(name, arrays, params, ny * (nx + 1) if kind in ("u", "u_cd0")
                   else ((ny + 1) * nx if kind in ("v", "v_cd0") else ny * nx))


def submit_wrf_smag_flux_s(state, cfg, f, kh, fx, fy, *, full_theta=False,
                           time_t=False):
    """Build the terrain-aware scalar face fluxes into caller scratch."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal wrf_smag_flux_s exact scalar reconstruction is not ported")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    for name, array, shape in (
            ("f", f, (nz, ny, nx)), ("kh", kh, (nz, ny, nx)),
            ("fx", fx, (nz, ny, nx + 1)),
            ("fy", fy, (nz, ny + 1, nx))):
        arrays[name] = _arrays(**{name: array})[name]
        _shape(array, shape, name)
    thb = state.thb
    _arrays(thb=thb)
    _shape(thb, (nz, ny, nx) if thb.ndim == 3 else (nz,), "thb")
    arrays["thb"] = thb
    if nz < 3:
        raise NotImplementedError("WRF scalar face reconstruction requires nz >= 3")
    params.update(full_theta=int(bool(full_theta)), thb3d=int(thb.ndim == 3))
    return _submit("wrf_smag_flux_s", arrays, params,
                   nz * (ny + 1) * (nx + 1))


def submit_wrf_smag_hd_s(state, cfg, fx, fy, tend, *, time_t=False):
    """Add the metric-corrected scalar face-flux divergence to tendency."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal wrf_smag_hd_s exact divergence path is not ported")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    for name, array, shape in (
            ("fx", fx, (nz, ny, nx + 1)),
            ("fy", fy, (nz, ny + 1, nx)),
            ("tend", tend, (nz, ny, nx))):
        arrays[name] = _arrays(**{name: array})[name]
        _shape(array, shape, name)
    return _submit("wrf_smag_hd_s", arrays, params, nz * ny * nx)


def submit_wrf_smag_w_primitives(state, cfg, cached_what, cached_rdz,
                                  cached_zx, cached_zy, *, time_t=False):
    """Populate W-face wind, inverse-height, and slope scratch fields."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal WRF cached w primitives do not implement exact policy")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    work = {
        "cached_what": (cached_what, (nz + 1, ny, nx)),
        "cached_rdz": (cached_rdz, (nz, ny, nx)),
        "cached_zx": (cached_zx, (nz, ny, nx + 1)),
        "cached_zy": (cached_zy, (nz, ny + 1, nx)),
    }
    for name, (array, shape) in work.items():
        arrays[name] = _arrays(**{name: array})[name]
        _shape(array, shape, name)
    if nz < 3:
        raise NotImplementedError("WRF cached w metrics require nz >= 3")
    return _submit("wrf_smag_w_primitives", arrays, params,
                   (nz + 1) * (ny + 1) * (nx + 1))


def submit_wrf_smag_hd_w_cached(state, cfg, km, cached_what, cached_rdz,
                                cached_zx, cached_zy, tend, *, time_t=False):
    """Submit the cached terrain-aware vertical-w stress divergence."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal WRF cached w stress exact expression path is not ported")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    work = {
        "km": (km, (nz, ny, nx)),
        "cached_what": (cached_what, (nz + 1, ny, nx)),
        "cached_rdz": (cached_rdz, (nz, ny, nx)),
        "cached_zx": (cached_zx, (nz, ny, nx + 1)),
        "cached_zy": (cached_zy, (nz, ny + 1, nx)),
        "tend": (tend, (nz + 1, ny, nx)),
    }
    for name, (array, shape) in work.items():
        arrays[name] = _arrays(**{name: array})[name]
        _shape(array, shape, name)
    if nz < 3:
        raise NotImplementedError("WRF cached w stress requires nz >= 3")
    return _submit("wrf_smag_hd_w_cached", arrays, params,
                   (nz + 1) * ny * nx)


def submit_wrf_smag_w_stress(state, cfg, km, tx, ty, *, time_t=False):
    """Produce the interior tau13/tau23 face stresses for w diffusion."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal WRF w-stress exact expression policy is not ported")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    for name, array, shape in (("km", km, (nz, ny, nx)),
                               ("tx", tx, (nz, ny, nx + 1)),
                               ("ty", ty, (nz, ny + 1, nx))):
        arrays[name] = _arrays(**{name: array})[name]
        _shape(array, shape, name)
    return _submit("wrf_smag_w_stress", arrays, params,
                   nz * (ny + 1) * (nx + 1))


def submit_wrf_smag_hd_w_stress(state, cfg, tx, ty, tend, *, time_t=False):
    """Apply the separately materialized tau13/tau23 stresses to w."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal WRF w-stress exact divergence is not ported")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    for name, array, shape in (
            ("tx", tx, (nz, ny, nx + 1)),
            ("ty", ty, (nz, ny + 1, nx)),
            ("tend", tend, (nz + 1, ny, nx))):
        arrays[name] = _arrays(**{name: array})[name]
        _shape(array, shape, name)
    return _submit("wrf_smag_hd_w_stress", arrays, params,
                   (nz + 1) * ny * nx)


def submit_wrf_smag_hd_w(state, cfg, km, tend, *, time_t=False):
    """Submit the ordinary Metal W terrain-stress path using caller scratch.

    Metal does not expose a CUDA compute-capability label. The cached source
    route used by the non-sm_120 CUDA targets is the supported route here.
    """
    nz, ny, nx = km.shape
    what = state.scratch((nz + 1, ny, nx), "acoustic_a")
    rdz = state.scratch((nz, ny, nx), "acoustic_c2a")
    zx = state.scratch((nz, ny, nx + 1), "diff6_x")
    zy = state.scratch((nz, ny + 1, nx), "diff6_y")
    submit_wrf_smag_w_primitives(state, cfg, what, rdz, zx, zy,
                                 time_t=time_t)
    return submit_wrf_smag_hd_w_cached(state, cfg, km, what, rdz, zx, zy,
                                       tend, time_t=time_t)


def submit_wrf_smag_hd_w_direct(state, cfg, km, tend, *, time_t=False):
    """Submit CUDA's direct W reference form for ordinary diagnostics."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal direct W reference does not implement WRF exact arithmetic")
    arrays, params, (nz, ny, nx) = _wrf_common(state, cfg, time_t=time_t)
    arrays["km"] = _arrays(km=km)["km"]
    arrays["tend"] = _arrays(tend=tend)["tend"]
    _shape(km, (nz, ny, nx), "km")
    _shape(tend, (nz + 1, ny, nx), "tend")
    if nz < 3:
        raise NotImplementedError("WRF direct W stress requires nz >= 3")
    return _submit("wrf_smag_hd_w", arrays, params, (nz + 1) * ny * nx)


def submit_wrf_smag_map_factor(tendency, map_factor, *, nz, ny, nx,
                               stagger, multiply_reciprocal):
    """Divide one dry tendency by its stagger-matched WRF map factor.

    ``multiply_reciprocal`` selects the WRF big-step y-stagger expression:
    FP32 ``(1 / map_factor)`` followed by a separately rounded multiply.
    Ordinary paths use FP32 division of the tendency by the map factor.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal dry map-factor coupling has no WRF-exact policy")
    nz, ny, nx = int(nz), int(ny), int(nx)
    if min(nz, ny, nx) <= 0:
        raise ValueError("map-factor coupling requires positive mass dimensions")
    stagger = str(stagger)
    codes = {"": 0, "x": 1, "y": 2, "z": 3}
    if stagger not in codes:
        raise ValueError(f"unsupported dry tendency stagger {stagger!r}")
    multiply_reciprocal = bool(multiply_reciprocal)
    if multiply_reciprocal and stagger != "y":
        raise ValueError(
            "reciprocal-multiply map-factor coupling is only valid for y stagger")

    shape = {
        "": (nz, ny, nx), "x": (nz, ny, nx + 1),
        "y": (nz, ny + 1, nx), "z": (nz + 1, ny, nx),
    }[stagger]
    factor_shape = {
        "": (ny, nx), "x": (ny, nx + 1),
        "y": (ny + 1, nx), "z": (ny, nx),
    }[stagger]
    arrays = _arrays(tendency=tendency, map_factor=map_factor)
    _shape(tendency, shape, "tendency")
    _shape(map_factor, factor_shape, "map_factor")
    if tendency.size > np.iinfo(np.uint32).max:
        raise OverflowError("dry tendency exceeds the Metal uint32 grid ABI")
    if tendency.buffer.context is not map_factor.buffer.context:
        raise ValueError("tendency and map_factor belong to different Metal contexts")
    params = {"nz": nz, "ny": ny, "nx": nx,
              "stagger": codes[stagger],
              "multiply_reciprocal": int(multiply_reciprocal)}
    return _submit("smag2d_couple_map_factor", arrays, params, tendency.size)
