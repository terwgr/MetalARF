"""Typed Metal launchers for WRF's IEVA preparation and column solves.

The Thomas recurrences use the shared integer-only software-binary64 helpers
to preserve CUDA's FP64 solve arrays on Metal.
"""

from __future__ import annotations

import numpy as np

from .array import MetalArray


_THREADS = 128
_I32_MAX = np.iinfo(np.int32).max


def _integer(name, value, *, minimum=0):
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"Metal IEVA {name} must be an integer")
    value = int(value)
    if value < minimum or value > _I32_MAX:
        raise ValueError(f"Metal IEVA {name} must fit the supported int32 range")
    return value


def _arrays(context, arrays, shapes):
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous):
            raise TypeError(f"Metal IEVA {name} must be contiguous float32 storage")
        if array.shape != shapes[name]:
            raise ValueError(
                f"Metal IEVA {name} shape {array.shape} != {shapes[name]}")
        if array.buffer.context is not context:
            raise ValueError("Metal IEVA arrays must belong to one context")


def _tiered(kernel: str, nz: int) -> str:
    """The column-capacity entrypoint CUDA's IEVA_KMAX tier selects."""
    from woof.core.ieva_constants import level_tier

    tier = level_tier(nz)
    return kernel if tier == 65 else f"{kernel}_k{tier}"


def submit_ww_split(context, *, ww, u, v, mut, rdnw, c1f, c2f, ww_e,
                    ww_i, nz, ny, nx, rdx, rdy, dt, alpha_max, ceps,
                    cmnx_ratio, cutoff, r4cmx, msft, has_msf=0, legacy=0,
                    quarter=0.25, one=1.0, zero=0.0):
    """Split full vertical mass flux into explicit and implicit shares."""
    nz = _integer("nz", nz, minimum=1)
    ny = _integer("ny", ny, minimum=1)
    nx = _integer("nx", nx, minimum=1)
    st = (nz + 1, ny, nx)
    arrays = {"ww": ww, "u": u, "v": v, "mut": mut, "rdnw": rdnw,
              "c1f": c1f, "c2f": c2f, "ww_e": ww_e, "ww_i": ww_i,
              "msft": msft}
    shapes = {"ww": st, "u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
              "mut": (ny, nx), "rdnw": (nz,), "c1f": (nz,), "c2f": (nz,),
              "ww_e": st, "ww_i": st, "msft": (ny, nx)}
    _arrays(context, arrays, shapes)
    parameters = {
        "rdx": float(np.float32(rdx)), "rdy": float(np.float32(rdy)),
        "dt": float(np.float32(dt)), "alpha_max": float(np.float32(alpha_max)),
        "ceps": float(np.float32(ceps)),
        "cmnx_ratio": float(np.float32(cmnx_ratio)),
        "cutoff": float(np.float32(cutoff)),
        "r4cmx": float(np.float32(r4cmx)),
        "quarter": float(np.float32(quarter)), "one": float(np.float32(one)),
        "zero": float(np.float32(zero)),
        "has_msf": _integer("has_msf", has_msf),
        "legacy": _integer("legacy", legacy), "nz": nz, "ny": ny, "nx": nx,
    }
    count = (nz + 1) * ny * nx
    if count > _I32_MAX:
        raise ValueError("Metal IEVA split extent exceeds the int32 indexing ABI")
    return context.submit_kernel(
        "ieva_ww_split", arrays, parameters, exact_policy=True,
        grid_size=count, threads_per_group=_THREADS)


def submit_mut_new(context, *, u, v, muu, muv, c1h, c2h, rdnw,
                    mut_old, msft, msfu, msfv, mut_new, nz, ny, nx,
                    rdx, rdy, dt, has_msf, one=1.0):
    """Compute WRF CALC_MUT_NEW's bottom-up FP32 column mass update."""
    nz = _integer("nz", nz, minimum=1)
    ny = _integer("ny", ny, minimum=1)
    nx = _integer("nx", nx, minimum=1)
    has_msf = _integer("has_msf", has_msf)
    if has_msf not in (0, 1):
        raise ValueError("Metal IEVA has_msf must be 0 or 1")
    arrays = {"u": u, "v": v, "muu": muu, "muv": muv, "c1h": c1h,
              "c2h": c2h, "rdnw": rdnw, "mut_old": mut_old,
              "msft": msft, "msfu": msfu, "msfv": msfv,
              "mut_new": mut_new}
    shapes = {"u": (nz, ny, nx + 1), "v": (nz, ny + 1, nx),
              "muu": (ny, nx + 1), "muv": (ny + 1, nx),
              "c1h": (nz,), "c2h": (nz,), "rdnw": (nz,),
              "mut_old": (ny, nx), "msft": (ny, nx),
              "msfu": (ny, nx + 1), "msfv": (ny + 1, nx),
              "mut_new": (ny, nx)}
    _arrays(context, arrays, shapes)
    count = ny * nx
    if count > _I32_MAX:
        raise ValueError("Metal IEVA mass extent exceeds the int32 indexing ABI")
    parameters = {"rdx": float(np.float32(rdx)),
                  "rdy": float(np.float32(rdy)),
                  "dt": float(np.float32(dt)), "one": float(np.float32(one)),
                  "has_msf": has_msf, "nz": nz, "ny": ny, "nx": nx}
    return context.submit_kernel(
        "ieva_mut_new", arrays, parameters, exact_policy=True,
        grid_size=count, threads_per_group=_THREADS)


def submit_solve_s(context, *, tend, pert, base, base_mode, shift, rom,
                   c1h, c2h, mut_old, mut_new, rdnw, dt, nz, ny, nx,
                   one=1.0, zero=0.0):
    """Solve the theta or transported-scalar IEVA column equation."""
    nz = _integer("nz", nz, minimum=1)
    ny = _integer("ny", ny, minimum=1)
    nx = _integer("nx", nx, minimum=1)
    base_mode = _integer("base_mode", base_mode)
    if base_mode not in (0, 1, 2):
        raise ValueError("Metal IEVA base_mode must be 0, 1, or 2")
    st = (nz, ny, nx)
    arrays = {"tend": tend, "pert": pert, "base": base, "rom": rom,
              "c1h": c1h, "c2h": c2h, "mut_old": mut_old,
              "mut_new": mut_new, "rdnw": rdnw}
    shapes = {"tend": st, "pert": st,
              "base": ((nz,) if base_mode == 1 else st),
              "rom": (nz + 1, ny, nx), "c1h": (nz,), "c2h": (nz,),
              "mut_old": (ny, nx), "mut_new": (ny, nx), "rdnw": (nz,)}
    _arrays(context, arrays, shapes)
    count = ny * nx
    if count > _I32_MAX:
        raise ValueError("Metal IEVA scalar extent exceeds int32 indexing ABI")
    parameters = {"base_mode": base_mode,
                  "shift": float(np.float32(shift)),
                  "dt": float(np.float32(dt)),
                  "one": float(np.float32(one)),
                  "zero": float(np.float32(zero)),
                  "nz": nz, "ny": ny, "nx": nx}
    return context.submit_kernel(
        _tiered("ieva_solve_s", nz), arrays, parameters, exact_policy=True,
        grid_size=count, threads_per_group=_THREADS)


def _submit_wind_solve(name, context, *, tend, old, rom, c1h, c2h,
                       mut_old, mut_new, rdnw, msf, legacy_face, dt,
                       has_msf, periodic, f_lo, f_hi, nz, ny, nx, legacy=1):
    nz = _integer("nz", nz, minimum=1)
    ny = _integer("ny", ny, minimum=1)
    nx = _integer("nx", nx, minimum=1)
    has_msf = _integer("has_msf", has_msf)
    periodic = _integer("periodic", int(periodic))
    f_lo = _integer("f_lo", f_lo)
    f_hi = _integer("f_hi", f_hi)
    legacy = _integer("legacy", legacy)
    if has_msf not in (0, 1) or periodic not in (0, 1) or legacy not in (0, 1):
        raise ValueError("Metal IEVA has_msf, periodic and legacy must be 0 or 1")
    nfaces = f_hi - f_lo + 1
    if nfaces <= 0:
        raise ValueError("Metal IEVA wind solve has an empty face interval")
    st = ((nz, ny, nx + 1), (ny, nx + 1)) if name == "u" else (
        (nz, ny + 1, nx), (ny + 1, nx))
    arrays = {"tend": tend, f"{name}_old": old, "rom": rom, "c1h": c1h,
              "c2h": c2h, "mut_old": mut_old, "mut_new": mut_new,
              "rdnw": rdnw, f"msf{name}": msf,
              "legacy_face": legacy_face}
    shapes = {"tend": st[0], f"{name}_old": st[0],
              "rom": (nz + 1, ny, nx), "c1h": (nz,), "c2h": (nz,),
              "mut_old": (ny, nx), "mut_new": (ny, nx), "rdnw": (nz,),
              f"msf{name}": st[1], "legacy_face": st[1]}
    _arrays(context, arrays, shapes)
    grid_size = (ny if name == "u" else nx) * nfaces
    if grid_size > _I32_MAX:
        raise ValueError("Metal IEVA wind extent exceeds int32 indexing ABI")
    parameters = {"dt": float(np.float32(dt)), "one": 1.0,
                  "zero": 0.0, "half_val": 0.5,
                  "has_msf": has_msf, "periodic": periodic,
                  "f_lo": f_lo, "f_hi": f_hi, "legacy": legacy,
                  "nz": nz, "ny": ny, "nx": nx}
    return context.submit_kernel(
        _tiered(f"ieva_solve_{name}", nz), arrays, parameters, exact_policy=True,
        grid_size=grid_size, threads_per_group=_THREADS)


def submit_solve_u(context, **kwargs):
    """Solve the IEVA u-wind columns on their active x faces."""
    return _submit_wind_solve("u", context, **kwargs)


def submit_solve_v(context, **kwargs):
    """Solve the IEVA v-wind columns on their active y faces."""
    return _submit_wind_solve("v", context, **kwargs)


def submit_solve_ph(context, *, tend, ph_old, phb, rom, c1f, c2f, mut,
                    rdnw, msft, dt, has_msf, nz, ny, nx,
                    one=1.0, zero=0.0, half_val=0.5):
    """Solve the WRF legacy geopotential columns with software binary64."""
    nz = _integer("nz", nz, minimum=2)
    ny = _integer("ny", ny, minimum=1)
    nx = _integer("nx", nx, minimum=1)
    has_msf = _integer("has_msf", has_msf)
    if has_msf not in (0, 1):
        raise ValueError("Metal IEVA has_msf must be 0 or 1")
    st = (nz + 1, ny, nx)
    arrays = {"tend": tend, "ph_old": ph_old, "phb": phb,
              "rom": rom, "c1f": c1f, "c2f": c2f, "mut": mut,
              "rdnw": rdnw, "msft": msft}
    shapes = {"tend": (nz, ny, nx), "ph_old": st,
              "phb": tuple(phb.shape), "rom": st, "c1f": (nz,),
              "c2f": (nz,), "mut": (ny, nx), "rdnw": (nz,),
              "msft": (ny, nx)}
    if shapes["phb"] not in ((nz + 1,), st):
        raise ValueError("Metal IEVA phb must be one- or three-dimensional")
    _arrays(context, arrays, shapes)
    count = ny * nx
    if count > _I32_MAX:
        raise ValueError("Metal IEVA geopotential extent exceeds int32 ABI")
    parameters = {"base3d": int(len(shapes["phb"]) == 3),
                  "dt": float(np.float32(dt)),
                  "one": float(np.float32(one)),
                  "zero": float(np.float32(zero)),
                  "half_val": float(np.float32(half_val)),
                  "has_msf": has_msf, "nz": nz, "ny": ny, "nx": nx}
    return context.submit_kernel(
        _tiered("ieva_solve_ph", nz), arrays, parameters, exact_policy=True,
        grid_size=count, threads_per_group=_THREADS)


def submit_solve_w(context, *, tend, utend, vtend, mux, muy, c1h, c2h,
                   msfu, msfv, ht, rom, ph_new, ph_old, ph_tend, w_old,
                   c1f, c2f, mut, mut_old, mut_new, rdn, msft,
                   cf1, cf2, cf3, rdx, rdy, dt, g, has_msf, bx, by,
                   legacy, nz, ny, nx, one=1.0, zero=0.0, half_val=0.5):
    """Solve the coupled WRF legacy vertical-wind columns in software FP64."""
    nz = _integer("nz", nz, minimum=3)
    ny = _integer("ny", ny, minimum=1)
    nx = _integer("nx", nx, minimum=1)
    has_msf = _integer("has_msf", has_msf)
    bx = _integer("bx", bx)
    by = _integer("by", by)
    legacy = _integer("legacy", legacy)
    if (has_msf not in (0, 1) or bx not in (0, 1) or by not in (0, 1)
            or legacy not in (0, 1)):
        raise ValueError("Metal IEVA flags must be 0 or 1")
    arrays = {"tend": tend, "utend": utend, "vtend": vtend,
              "mux": mux, "muy": muy, "c1h": c1h, "c2h": c2h,
              "msfu": msfu, "msfv": msfv, "ht": ht, "rom": rom,
              "ph_new": ph_new, "ph_old": ph_old, "ph_tend": ph_tend,
              "w_old": w_old, "c1f": c1f, "c2f": c2f, "mut": mut,
              "mut_old": mut_old, "mut_new": mut_new, "rdn": rdn,
              "msft": msft}
    shapes = {"tend": (nz + 1, ny, nx), "utend": (nz, ny, nx + 1),
              "vtend": (nz, ny + 1, nx), "mux": (ny, nx + 1),
              "muy": (ny + 1, nx), "c1h": (nz,), "c2h": (nz,),
              "msfu": (ny, nx + 1), "msfv": (ny + 1, nx),
              "ht": (ny, nx), "rom": (nz + 1, ny, nx),
              "ph_new": (nz + 1, ny, nx), "ph_old": (nz + 1, ny, nx),
              "ph_tend": (nz + 1, ny, nx), "w_old": (nz + 1, ny, nx),
              "c1f": (nz,), "c2f": (nz,), "mut": (ny, nx),
              "mut_old": (ny, nx), "mut_new": (ny, nx), "rdn": (nz,),
              "msft": (ny, nx)}
    _arrays(context, arrays, shapes)
    count = ny * nx
    if count > _I32_MAX:
        raise ValueError("Metal IEVA w extent exceeds int32 ABI")
    parameters = {"cf1": float(np.float32(cf1)),
                  "cf2": float(np.float32(cf2)),
                  "cf3": float(np.float32(cf3)),
                  "rdx": float(np.float32(rdx)),
                  "rdy": float(np.float32(rdy)),
                  "dt": float(np.float32(dt)), "g": float(np.float32(g)),
                  "one": float(np.float32(one)), "zero": float(np.float32(zero)),
                  "half_val": float(np.float32(half_val)),
                  "has_msf": has_msf, "bx": bx, "by": by,
                  "legacy": legacy, "nz": nz, "ny": ny, "nx": nx}
    return context.submit_kernel(
        _tiered("ieva_solve_w", nz), arrays, parameters, exact_policy=True,
        grid_size=count, threads_per_group=_THREADS)
