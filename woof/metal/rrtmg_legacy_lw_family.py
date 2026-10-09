"""Typed Metal launches for legacy WRF RRTMG longwave stages."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray


def submit_inatm(plev, wkl, wx, coldry, wbrodl, pwvcm, *, grav, avogad):
    """Metal twin of ``rrtmg_lw_chain.cu::rlw_inatm``.

    ``wkl`` and ``wx`` retain the CUDA column/species slabs in C-contiguous
    ``(ncol, species, nl)`` buffers. The kernel writes those inputs in place,
    matching the CUDA entry point.
    """
    arrays = (plev, wkl, wx, coldry, wbrodl, pwvcm)
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError("legacy RRTMG LW inatm requires MetalArray resources")
    if any(value.dtype != np.dtype(np.float32)
           or not value.flags.c_contiguous for value in arrays):
        raise TypeError("legacy RRTMG LW inatm resources must be contiguous FP32")
    context = plev.buffer.context
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError("legacy RRTMG LW inatm resources must share a context")
    if plev.ndim != 2:
        raise ValueError("legacy RRTMG LW pressure levels must be rank two")
    ncol, nlev = plev.shape
    nl = nlev - 1
    if (nl < 1 or wkl.shape != (ncol, 35, nl)
            or wx.shape != (ncol, 4, nl)
            or coldry.shape != (ncol, nl) or wbrodl.shape != (ncol, nl)
            or pwvcm.shape != (ncol,)):
        raise ValueError("legacy RRTMG LW inatm resource shapes disagree")
    return context.submit_kernel(
        "rrtmg_legacy_lw_inatm",
        {"plev": plev, "wkl": wkl, "wx": wx, "coldry": coldry,
         "wbrodl": wbrodl, "pwvcm": pwvcm},
        {"ncol": ncol, "nl": nl, "grav": np.float32(grav),
         "avogad": np.float32(avogad)}, exact_policy=False,
        grid_size=ncol, threads_per_group=128)


def submit_planck(tavel, tz, tbound, sem, totplnk, totplk16,
                  planklay, planklev, plankbnd, *, istart=1):
    """Interpolate legacy RRTMG's 16-band Planck tables in SETCOEF."""
    arrays = (tavel, tz, tbound, sem, totplnk, totplk16,
              planklay, planklev, plankbnd)
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError("legacy RRTMG LW Planck stage requires MetalArray resources")
    context = tavel.buffer.context
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError("legacy RRTMG LW Planck resources must share a context")
    if any(not value.flags.c_contiguous for value in arrays):
        raise ValueError("legacy RRTMG LW Planck resources must be contiguous")
    if tavel.ndim != 2:
        raise ValueError("legacy RRTMG LW Planck temperature must be rank two")
    ncol, nl = tavel.shape
    if (ncol < 1 or nl < 1 or tz.shape != (ncol, nl + 1)
            or tbound.shape != (ncol,) or sem.shape != (ncol, 16)
            or totplnk.shape != (181 * 16,) or totplk16.shape != (181,)
            or planklay.shape != (ncol, nl, 16)
            or planklev.shape != (ncol, nl + 1, 16)
            or plankbnd.shape != (ncol, 16)):
        raise ValueError("legacy RRTMG LW Planck resource shapes disagree")
    if any(value.dtype != np.dtype(np.float32) for value in arrays):
        raise TypeError("legacy RRTMG LW Planck resources must be FP32")
    return context.submit_kernel(
        "rrtmg_legacy_lw_planck",
        {"tavel": tavel, "tz": tz, "tbound": tbound, "sem": sem,
         "totplnk": totplnk, "totplk16": totplk16,
         "planklay": planklay, "planklev": planklev,
         "plankbnd": plankbnd},
        {"ncol": ncol, "nl": nl, "istart": int(istart)},
        exact_policy=False, grid_size=ncol, threads_per_group=128)


def submit_taut(taug, taua, ngb, taut, *, nbnd=16):
    """Add band aerosol optical depth to each legacy RRTMG g-point."""
    arrays = (taug, taua, ngb, taut)
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError("legacy RRTMG LW taut requires MetalArray resources")
    context = taug.buffer.context
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError("legacy RRTMG LW taut resources must share a context")
    if (any(not value.flags.c_contiguous for value in arrays)
            or taug.dtype != np.dtype(np.float32)
            or taua.dtype != np.dtype(np.float32)
            or taut.dtype != np.dtype(np.float32)
            or ngb.dtype != np.dtype(np.int32)):
        raise TypeError("legacy RRTMG LW taut resource dtypes/layouts disagree")
    if taug.ndim != 3:
        raise ValueError("legacy RRTMG LW g-point optical depth must be rank three")
    ncol, nl, ngpt = taug.shape
    if (ngpt != 140 or taua.shape != (ncol, nl, nbnd)
            or ngb.shape != (ngpt,) or taut.shape != taug.shape):
        raise ValueError("legacy RRTMG LW taut resource shapes disagree")
    return context.submit_kernel(
        "rrtmg_legacy_lw_taut",
        {"taug": taug, "taua": taua, "ngb": ngb, "taut": taut},
        {"ncol": ncol, "nl": nl, "ngpt": ngpt, "nbnd": nbnd},
        exact_policy=False, grid_size=taug.size, threads_per_group=128)


def submit_setcoef_indices(pavel, tavel, preflog, tref, jp, jt, jt1,
                           fp, ft, ft1, laytrop):
    """Build RRTMG SETCOEF's pressure and temperature table coordinates."""
    arrays = (pavel, tavel, preflog, tref, jp, jt, jt1, fp, ft, ft1,
              laytrop)
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError("legacy RRTMG LW setcoef requires MetalArray resources")
    context = pavel.buffer.context
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError("legacy RRTMG LW setcoef resources must share a context")
    if any(not value.flags.c_contiguous for value in arrays):
        raise ValueError("legacy RRTMG LW setcoef resources must be contiguous")
    ncol, nl = pavel.shape if pavel.ndim == 2 else (0, 0)
    if (ncol < 1 or nl < 1 or tavel.shape != (ncol, nl)
            or preflog.shape != (59,) or tref.shape != (59,)
            or any(value.shape != (ncol, nl) for value in (jp, jt, jt1, fp, ft, ft1))
            or laytrop.shape != (ncol,)):
        raise ValueError("legacy RRTMG LW setcoef resource shapes disagree")
    if (any(value.dtype != np.dtype(np.float32)
            for value in (pavel, tavel, preflog, tref, fp, ft, ft1))
            or any(value.dtype != np.dtype(np.int32)
                   for value in (jp, jt, jt1, laytrop))):
        raise TypeError("legacy RRTMG LW setcoef resource dtypes disagree")
    return context.submit_kernel(
        "rrtmg_legacy_lw_setcoef_indices",
        {"pavel": pavel, "tavel": tavel, "preflog": preflog,
         "tref": tref, "jp": jp, "jt": jt, "jt1": jt1, "fp": fp,
         "ft": ft, "ft1": ft1, "laytrop": laytrop},
        {"ncol": ncol, "nl": nl}, exact_policy=False,
        grid_size=ncol, threads_per_group=128)


def submit_setcoef_gas(pavel, tavel, coldry, wkl, wbrodl, chi_mls,
                       jp, jt, jt1, fp, ft, ft1, fs, isv):
    """Pack the gas columns/factors from legacy RRTMG SETCOEF into FS/IS."""
    arrays = (pavel, tavel, coldry, wkl, wbrodl, chi_mls, jp, jt, jt1,
              fp, ft, ft1, fs, isv)
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError("legacy RRTMG LW setcoef gas requires MetalArray resources")
    context = pavel.buffer.context
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError("legacy RRTMG LW setcoef gas must share one context")
    if any(not value.flags.c_contiguous for value in arrays):
        raise ValueError("legacy RRTMG LW setcoef gas arrays must be contiguous")
    if pavel.ndim != 2:
        raise ValueError("legacy RRTMG LW setcoef pavel must be rank two")
    ncol, nl = pavel.shape
    if (ncol < 1 or nl < 1 or tavel.shape != (ncol, nl)
            or coldry.shape != (ncol, nl) or wbrodl.shape != (ncol, nl)
            or wkl.shape != (ncol, 35, nl) or chi_mls.shape != (413,)
            or any(value.shape != (ncol, nl)
                   for value in (jp, jt, jt1, fp, ft, ft1))
            or fs.shape != (33, ncol, nl)
            or isv.shape != (6, ncol, nl)):
        raise ValueError("legacy RRTMG LW setcoef gas resource shapes disagree")
    int_arrays = (jp, jt, jt1, isv)
    if (any(value.dtype != np.dtype(np.float32)
            for value in (pavel, tavel, coldry, wkl, wbrodl, chi_mls,
                          fp, ft, ft1, fs))
            or any(value.dtype != np.dtype(np.int32) for value in int_arrays)):
        raise TypeError("legacy RRTMG LW setcoef gas resource dtypes disagree")
    return context.submit_kernel(
        "rrtmg_legacy_lw_setcoef_gas",
        {"pavel": pavel, "tavel": tavel, "coldry": coldry, "wkl": wkl,
         "wbrodl": wbrodl, "chi_mls": chi_mls, "jp": jp, "jt": jt,
         "jt1": jt1, "fp": fp, "ft": ft, "ft1": ft1,
         "fs": fs, "isv": isv},
        {"ncol": ncol, "nl": nl}, exact_policy=False,
        grid_size=ncol * nl, threads_per_group=128)



def submit_taugb2(laytrop, fs, isv, absa, absb, selfref, forref,
                  fracrefa, fracrefb, taug, fracs):
    """Launch legacy RRTMG longwave band 2 optical-depth calculation."""
    arrays = (laytrop, fs, isv, absa, absb, selfref, forref,
              fracrefa, fracrefb, taug, fracs)
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError("legacy RRTMG LW taugb2 requires MetalArray resources")
    context = laytrop.buffer.context
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError("legacy RRTMG LW taugb2 resources must share a context")
    if any(not value.flags.c_contiguous for value in arrays):
        raise ValueError("legacy RRTMG LW taugb2 resources must be contiguous")
    if laytrop.ndim != 1 or fs.ndim != 3 or isv.ndim != 3:
        raise ValueError("legacy RRTMG LW taugb2 state ranks disagree")
    ncol = int(laytrop.shape[0])
    if fs.shape[0] != 33 or fs.shape[1] != ncol or isv.shape != (6, ncol, fs.shape[2]):
        raise ValueError("legacy RRTMG LW taugb2 packed state shapes disagree")
    nl = int(fs.shape[2])
    if (laytrop.dtype != np.dtype(np.int32) or isv.dtype != np.dtype(np.int32)
            or fs.dtype != np.dtype(np.float32)
            or any(value.dtype != np.dtype(np.float32)
                   for value in (absa, absb, selfref, forref, fracrefa,
                                 fracrefb, taug, fracs))):
        raise TypeError("legacy RRTMG LW taugb2 requires FP32 and INT32 buffers")
    if (absa.shape != (65 * 12,) or absb.shape != (235 * 12,)
            or selfref.shape != (10 * 12,) or forref.shape != (4 * 12,)
            or fracrefa.shape != (12,) or fracrefb.shape != (12,)
            or taug.shape != (ncol, nl, 140) or fracs.shape != (ncol, nl, 140)):
        raise ValueError("legacy RRTMG LW taugb2 coefficient/output shapes disagree")
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb2",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "absa": absa,
         "absb": absb, "selfref": selfref, "forref": forref,
         "fracrefa": fracrefa, "fracrefb": fracrefb,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl}, exact_policy=False,
        grid_size=ncol * nl, threads_per_group=128)



def _taugb_state(laytrop, fs, isv, taug, fracs, band):
    state = (laytrop, fs, isv, taug, fracs)
    if any(not isinstance(value, MetalArray) for value in state):
        raise TypeError(f"legacy RRTMG LW taugb{band} requires MetalArray state")
    context = laytrop.buffer.context
    if any(value.buffer.context is not context for value in state):
        raise ValueError(f"legacy RRTMG LW taugb{band} state must share a context")
    if any(not value.flags.c_contiguous for value in state):
        raise ValueError(f"legacy RRTMG LW taugb{band} state must be contiguous")
    if laytrop.ndim != 1 or fs.ndim != 3 or isv.ndim != 3:
        raise ValueError(f"legacy RRTMG LW taugb{band} state ranks disagree")
    ncol = int(laytrop.shape[0])
    nl = int(fs.shape[2])
    if (fs.shape[:2] != (33, ncol) or isv.shape != (6, ncol, nl)
            or taug.shape != (ncol, nl, 140)
            or fracs.shape != (ncol, nl, 140)):
        raise ValueError(f"legacy RRTMG LW taugb{band} state shapes disagree")
    if (laytrop.dtype != np.dtype(np.int32) or isv.dtype != np.dtype(np.int32)
            or any(value.dtype != np.dtype(np.float32)
                   for value in (fs, taug, fracs))):
        raise TypeError(f"legacy RRTMG LW taugb{band} state dtypes disagree")
    return context, ncol, nl


def _taugb_coefficients(context, arrays, shapes, band):
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError(f"legacy RRTMG LW taugb{band} requires MetalArray tables")
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError(f"legacy RRTMG LW taugb{band} tables must share a context")
    if any(not value.flags.c_contiguous or value.dtype != np.dtype(np.float32)
           for value in arrays):
        raise TypeError(f"legacy RRTMG LW taugb{band} tables must be contiguous FP32")
    if any(value.shape != shape for value, shape in zip(arrays, shapes)):
        raise ValueError(f"legacy RRTMG LW taugb{band} table shapes disagree")


def submit_taugb10(laytrop, fs, isv, absa, absb, selfref, forref,
                   fracrefa, fracrefb, taug, fracs):
    """Launch legacy RRTMG longwave band 10 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 10)
    coeff = (absa, absb, selfref, forref, fracrefa, fracrefb)
    _taugb_coefficients(context, coeff, ((390,), (1410,), (60,), (24,), (6,), (6,)), 10)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb10",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "absa": absa,
         "absb": absb, "selfref": selfref, "forref": forref,
         "fracrefa": fracrefa, "fracrefb": fracrefb,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl}, exact_policy=False,
        grid_size=ncol * nl, threads_per_group=128)


def submit_taugb11(laytrop, fs, isv, absa, absb, selfref, forref,
                   ka_mo2, kb_mo2, fracrefa, fracrefb, taug, fracs):
    """Launch legacy RRTMG longwave band 11 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 11)
    coeff = (absa, absb, selfref, forref, ka_mo2, kb_mo2,
             fracrefa, fracrefb)
    _taugb_coefficients(context, coeff,
                        ((520,), (1880,), (80,), (32,), (152,), (152,), (8,), (8,)),
                        11)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb11",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "absa": absa,
         "absb": absb, "selfref": selfref, "forref": forref,
         "ka_mo2": ka_mo2, "kb_mo2": kb_mo2,
         "fracrefa": fracrefa, "fracrefb": fracrefb,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl}, exact_policy=False,
        grid_size=ncol * nl, threads_per_group=128)



def submit_taugb12(laytrop, fs, isv, chi_mls, absa, selfref, forref,
                   fracrefa, taug, fracs, *, oneminus):
    """Launch legacy RRTMG longwave band 12 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 12)
    coeff = (chi_mls, absa, selfref, forref, fracrefa)
    _taugb_coefficients(context, coeff,
                        ((413,), (4680,), (80,), (32,), (72,)), 12)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb12",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "chi_mls": chi_mls,
         "absa": absa, "selfref": selfref, "forref": forref,
         "fracrefa": fracrefa, "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl, "oneminus": np.float32(oneminus)},
        exact_policy=False, grid_size=ncol * nl, threads_per_group=128)



def submit_taugb1(laytrop, fs, isv, absa, absb, selfref, forref,
                  ka_mn2, kb_mn2, fracrefa, fracrefb, taug, fracs):
    """Launch legacy RRTMG longwave band 1 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 1)
    coeff = (absa, absb, selfref, forref, ka_mn2, kb_mn2, fracrefa, fracrefb)
    _taugb_coefficients(context, coeff,
                        ((650,), (2350,), (100,), (40,), (190,), (190,), (10,), (10,)),
                        1)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb1",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "absa": absa,
         "absb": absb, "selfref": selfref, "forref": forref,
         "ka_mn2": ka_mn2, "kb_mn2": kb_mn2,
         "fracrefa": fracrefa, "fracrefb": fracrefb,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl}, exact_policy=False,
        grid_size=ncol * nl, threads_per_group=128)



def submit_taugb4(laytrop, fs, isv, chi_mls, absa, absb, selfref, forref,
                  fracrefa, fracrefb, taug, fracs, *, oneminus):
    """Launch legacy RRTMG longwave band 4 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 4)
    coeff = (chi_mls, absa, absb, selfref, forref, fracrefa, fracrefb)
    _taugb_coefficients(context, coeff,
                        ((413,), (8190,), (16450,), (140,), (56,), (126,), (70,)),
                        4)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb4",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "chi_mls": chi_mls,
         "absa": absa, "absb": absb, "selfref": selfref, "forref": forref,
         "fracrefa": fracrefa, "fracrefb": fracrefb,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl, "oneminus": np.float32(oneminus)},
        exact_policy=False, grid_size=ncol * nl, threads_per_group=128)



def submit_taugb3(laytrop, fs, isv, chi_mls, absa, absb, selfref, forref,
                  ka_mn2o, kb_mn2o, fracrefa, fracrefb, taug, fracs, *, oneminus):
    """Launch legacy RRTMG longwave band 3 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 3)
    coeff = (chi_mls, absa, absb, selfref, forref, ka_mn2o, kb_mn2o,
             fracrefa, fracrefb)
    _taugb_coefficients(context, coeff,
                        ((413,), (9360,), (18800,), (160,), (64,),
                         (2736,), (1520,), (144,), (80,)), 3)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb3",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "chi_mls": chi_mls,
         "absa": absa, "absb": absb, "selfref": selfref, "forref": forref,
         "ka_mn2o": ka_mn2o, "kb_mn2o": kb_mn2o,
         "fracrefa": fracrefa, "fracrefb": fracrefb,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl, "oneminus": np.float32(oneminus)},
        exact_policy=False, grid_size=ncol * nl, threads_per_group=128)



def submit_taugb5(laytrop, fs, isv, wx, chi_mls, absa, absb, selfref, forref,
                  ka_mo3, ccl4, fracrefa, fracrefb, taug, fracs, *, oneminus):
    """Launch legacy RRTMG longwave band 5 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 5)
    coeff = (wx, chi_mls, absa, absb, selfref, forref, ka_mo3,
             ccl4, fracrefa, fracrefb)
    _taugb_coefficients(context, coeff,
                        ((ncol, 4, nl), (413,), (9360,), (18800,),
                         (160,), (64,), (2736,), (16,), (144,), (80,)), 5)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb5",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "wx": wx,
         "chi_mls": chi_mls, "absa": absa, "absb": absb,
         "selfref": selfref, "forref": forref, "ka_mo3": ka_mo3,
         "ccl4": ccl4, "fracrefa": fracrefa, "fracrefb": fracrefb,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl, "oneminus": np.float32(oneminus)},
        exact_policy=False, grid_size=ncol * nl, threads_per_group=128)



def submit_taugb6(laytrop, fs, isv, wx, chi_mls, absa, selfref, forref,
                  ka_mco2, cfc11adj, cfc12, fracrefa, taug, fracs):
    """Launch legacy RRTMG longwave band 6 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 6)
    coeff = (wx, chi_mls, absa, selfref, forref, ka_mco2,
             cfc11adj, cfc12, fracrefa)
    _taugb_coefficients(context, coeff,
                        ((ncol, 4, nl), (413,), (520,), (80,), (32,),
                         (152,), (8,), (8,), (8,)), 6)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb6",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "wx": wx,
         "chi_mls": chi_mls, "absa": absa, "selfref": selfref,
         "forref": forref, "ka_mco2": ka_mco2, "cfc11adj": cfc11adj,
         "cfc12": cfc12, "fracrefa": fracrefa, "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl}, exact_policy=False,
        grid_size=ncol * nl, threads_per_group=128)



def submit_taugb7(laytrop, fs, isv, chi_mls, absa, absb, selfref, forref,
                  ka_mco2, kb_mco2, fracrefa, fracrefb, taug, fracs, *, oneminus):
    """Launch legacy RRTMG longwave band 7 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 7)
    coeff = (chi_mls, absa, absb, selfref, forref, ka_mco2,
             kb_mco2, fracrefa, fracrefb)
    _taugb_coefficients(context, coeff,
                        ((413,), (7020,), (2820,), (120,), (48,),
                         (2052,), (228,), (108,), (12,)), 7)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb7",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "chi_mls": chi_mls,
         "absa": absa, "absb": absb, "selfref": selfref, "forref": forref,
         "ka_mco2": ka_mco2, "kb_mco2": kb_mco2,
         "fracrefa": fracrefa, "fracrefb": fracrefb,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl, "oneminus": np.float32(oneminus)},
        exact_policy=False, grid_size=ncol * nl, threads_per_group=128)



def submit_taugb8(laytrop, fs, isv, wx, chi_mls, absa, absb, selfref, forref,
                  ka_mco2, kb_mco2, ka_mn2o, kb_mn2o, ka_mo3, cfc12, cfc22adj,
                  fracrefa, fracrefb, taug, fracs):
    """Launch legacy RRTMG longwave band 8 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 8)
    coeff = (wx, chi_mls, absa, absb, selfref, forref, ka_mco2, kb_mco2,
             ka_mn2o, kb_mn2o, ka_mo3, cfc12, cfc22adj, fracrefa, fracrefb)
    _taugb_coefficients(context, coeff,
        ((ncol, 4, nl), (413,), (520,), (1880,), (80,), (32,),
                         (152,), (152,), (152,), (152,), (152,), (8,), (8,),
                         (8,), (8,)), 8)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb8",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "wx": wx,
         "chi_mls": chi_mls, "absa": absa, "absb": absb,
         "selfref": selfref, "forref": forref, "ka_mco2": ka_mco2,
         "kb_mco2": kb_mco2, "ka_mn2o": ka_mn2o, "kb_mn2o": kb_mn2o,
         "ka_mo3": ka_mo3, "cfc12": cfc12, "cfc22adj": cfc22adj,
         "fracrefa": fracrefa, "fracrefb": fracrefb,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl}, exact_policy=False,
        grid_size=ncol * nl, threads_per_group=128)



def submit_taugb9(laytrop, fs, isv, chi_mls, absa, absb, selfref, forref,
                  ka_mn2o, kb_mn2o, fracrefa, fracrefb, taug, fracs, *, oneminus):
    """Launch legacy RRTMG longwave band 9 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 9)
    coeff = (chi_mls, absa, absb, selfref, forref, ka_mn2o, kb_mn2o,
             fracrefa, fracrefb)
    _taugb_coefficients(context, coeff,
                        ((413,), (7020,), (2820,), (120,), (48,),
                         (2052,), (228,), (108,), (12,)), 9)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb9",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "chi_mls": chi_mls,
         "absa": absa, "absb": absb, "selfref": selfref, "forref": forref,
         "ka_mn2o": ka_mn2o, "kb_mn2o": kb_mn2o,
         "fracrefa": fracrefa, "fracrefb": fracrefb,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl, "oneminus": np.float32(oneminus)},
        exact_policy=False, grid_size=ncol * nl, threads_per_group=128)



def submit_taugb14(laytrop, fs, isv, absa, absb, selfref, forref,
                   fracrefa, fracrefb, taug, fracs):
    """Launch legacy RRTMG longwave band 14 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 14)
    coeff = (absa, absb, selfref, forref, fracrefa, fracrefb)
    _taugb_coefficients(context, coeff,
                        ((130,), (470,), (20,), (8,), (2,), (2,)), 14)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb14",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "absa": absa,
         "absb": absb, "selfref": selfref, "forref": forref,
         "fracrefa": fracrefa, "fracrefb": fracrefb,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl}, exact_policy=False,
        grid_size=ncol * nl, threads_per_group=128)



def submit_taugb15(laytrop, fs, isv, chi_mls, absa, selfref, forref,
                   ka_mn2, fracrefa, taug, fracs, *, oneminus):
    """Launch legacy RRTMG longwave band 15 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 15)
    coeff = (chi_mls, absa, selfref, forref, ka_mn2, fracrefa)
    _taugb_coefficients(context, coeff,
                        ((413,), (1170,), (20,), (8,), (342,), (18,)), 15)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb15",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "chi_mls": chi_mls,
         "absa": absa, "selfref": selfref, "forref": forref,
         "ka_mn2": ka_mn2, "fracrefa": fracrefa,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl, "oneminus": np.float32(oneminus)},
        exact_policy=False, grid_size=ncol * nl, threads_per_group=128)


def submit_taugb13(laytrop, fs, isv, chi_mls, absa, selfref, forref,
                   ka_mco2, ka_mco, kb_mo3, fracrefa, fracrefb,
                   taug, fracs, *, oneminus):
    """Launch legacy RRTMG longwave band 13 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 13)
    coeff = (chi_mls, absa, selfref, forref, ka_mco2, ka_mco,
             kb_mo3, fracrefa, fracrefb)
    _taugb_coefficients(context, coeff,
                        ((413,), (2340,), (40,), (16,), (684,),
                         (684,), (76,), (36,), (4,)), 13)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb13",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "chi_mls": chi_mls,
         "absa": absa, "selfref": selfref, "forref": forref,
         "ka_mco2": ka_mco2, "ka_mco": ka_mco, "kb_mo3": kb_mo3,
         "fracrefa": fracrefa, "fracrefb": fracrefb,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl, "oneminus": np.float32(oneminus)},
        exact_policy=False, grid_size=ncol * nl, threads_per_group=128)


def submit_taugb16(laytrop, fs, isv, chi_mls, absa, absb, selfref,
                   forref, fracrefa, fracrefb, taug, fracs, *, oneminus):
    """Launch legacy RRTMG longwave band 16 optical-depth calculation."""
    context, ncol, nl = _taugb_state(laytrop, fs, isv, taug, fracs, 16)
    coeff = (chi_mls, absa, absb, selfref, forref, fracrefa, fracrefb)
    _taugb_coefficients(context, coeff,
                        ((413,), (1170,), (470,), (20,), (8,), (18,), (2,)), 16)
    return context.submit_kernel(
        "rrtmg_legacy_lw_taugb16",
        {"laytrop": laytrop, "fs": fs, "isv": isv, "chi_mls": chi_mls,
         "absa": absa, "absb": absb, "selfref": selfref,
         "forref": forref, "fracrefa": fracrefa, "fracrefb": fracrefb,
         "taug": taug, "fracs": fracs},
        {"ncol": ncol, "nl": nl, "oneminus": np.float32(oneminus)},
        exact_policy=False, grid_size=ncol * nl, threads_per_group=128)


def submit_rtrn_secdiff(pwvcm, a0, a1, a2, secdiff):
    """Compute the legacy RRTMG diffusivity factor for each band."""
    arrays = (pwvcm, a0, a1, a2, secdiff)
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError("legacy RRTMG LW secdiff requires MetalArray resources")
    context = pwvcm.buffer.context
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError("legacy RRTMG LW secdiff resources must share a context")
    if any(not value.flags.c_contiguous for value in arrays):
        raise ValueError("legacy RRTMG LW secdiff resources must be contiguous")
    ncol = int(pwvcm.shape[0]) if pwvcm.ndim == 1 else 0
    if (ncol < 1 or any(value.shape != (16,) for value in (a0, a1, a2))
            or secdiff.shape != (ncol, 16)):
        raise ValueError("legacy RRTMG LW secdiff resource shapes disagree")
    if any(value.dtype != np.dtype(np.float32) for value in arrays):
        raise TypeError("legacy RRTMG LW secdiff resources must be FP32")
    return context.submit_kernel(
        "rrtmg_legacy_lw_rtrn_secdiff",
        {"pwvcm": pwvcm, "a0": a0, "a1": a1, "a2": a2,
         "secdiff": secdiff},
        {"ncol": ncol}, exact_policy=False,
        grid_size=ncol, threads_per_group=128)


def submit_rtrn_prol(cldfmc, taucmc, secdiff, ngb, odcld, efclfrac,
                     icldlyr):
    """Prepare cloud optical depth and layer cloudy flags per g-point."""
    arrays = (cldfmc, taucmc, secdiff, ngb, odcld, efclfrac, icldlyr)
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError("legacy RRTMG LW cloud prep requires MetalArray resources")
    context = cldfmc.buffer.context
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError("legacy RRTMG LW cloud prep resources must share a context")
    if any(not value.flags.c_contiguous for value in arrays):
        raise ValueError("legacy RRTMG LW cloud prep resources must be contiguous")
    ncol, ngpt, nl = cldfmc.shape if cldfmc.ndim == 3 else (0, 0, 0)
    if (ncol < 1 or ngpt != 140 or nl < 1 or taucmc.shape != cldfmc.shape
            or odcld.shape != cldfmc.shape or efclfrac.shape != cldfmc.shape
            or secdiff.shape != (ncol, 16) or ngb.shape != (140,)
            or icldlyr.shape != (ncol, nl)):
        raise ValueError("legacy RRTMG LW cloud prep resource shapes disagree")
    if any(value.dtype != np.dtype(np.float32)
           for value in (cldfmc, taucmc, secdiff, odcld, efclfrac)):
        raise TypeError("legacy RRTMG LW cloud prep float resources must be FP32")
    if ngb.dtype != np.dtype(np.int32) or icldlyr.dtype != np.dtype(np.int32):
        raise TypeError("legacy RRTMG LW cloud prep integer resources must be INT32")
    return context.submit_kernel(
        "rrtmg_legacy_lw_rtrn_prol",
        {"cldfmc": cldfmc, "taucmc": taucmc, "secdiff": secdiff,
         "ngb": ngb, "odcld": odcld, "efclfrac": efclfrac,
         "icldlyr": icldlyr},
        {"ncol": ncol, "nl": nl}, exact_policy=False,
        grid_size=ncol * nl, threads_per_group=128)


def submit_rtrn_accum(radld_p, radclrd_p, radlu_p, radclru_p, iclddn_p,
                      radlu_sfc, radclru_sfc, ngs, delwave, totuflux,
                      totdflux, totuclfl, totdclfl, *, wtdiff):
    """Accumulate the 140 g-point radiances into 16 bands."""
    arrays = (radld_p, radclrd_p, radlu_p, radclru_p, iclddn_p,
              radlu_sfc, radclru_sfc, ngs, delwave,
              totuflux, totdflux, totuclfl, totdclfl)
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError("legacy RRTMG LW accumulation requires MetalArray resources")
    context = radld_p.buffer.context
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError("legacy RRTMG LW accumulation resources must share a context")
    if any(not value.flags.c_contiguous for value in arrays):
        raise ValueError("legacy RRTMG LW accumulation resources must be contiguous")
    ncol, ngpt, nl = radld_p.shape if radld_p.ndim == 3 else (0, 0, 0)
    shape = (ncol, ngpt, nl)
    outshape = (ncol, nl + 1)
    if (ncol < 1 or ngpt != 140 or nl < 1
            or any(value.shape != shape for value in (radclrd_p, radlu_p, radclru_p, iclddn_p))
            or radlu_sfc.shape != (ncol, 140) or radclru_sfc.shape != (ncol, 140)
            or ngs.shape != (16,) or delwave.shape != (16,)
            or any(value.shape != outshape for value in (totuflux, totdflux, totuclfl, totdclfl))):
        raise ValueError("legacy RRTMG LW accumulation resource shapes disagree")
    if any(value.dtype != np.dtype(np.float32) for value in
           (radld_p, radclrd_p, radlu_p, radclru_p, radlu_sfc, radclru_sfc, delwave,
            totuflux, totdflux, totuclfl, totdclfl)):
        raise TypeError("legacy RRTMG LW accumulation float resources must be FP32")
    if any(value.dtype != np.dtype(np.int32) for value in (iclddn_p, ngs)):
        raise TypeError("legacy RRTMG LW accumulation integer resources must be INT32")
    return context.submit_kernel(
        "rrtmg_legacy_lw_rtrn_accum",
        {"radld_p": radld_p, "radclrd_p": radclrd_p, "radlu_p": radlu_p,
         "radclru_p": radclru_p, "iclddn_p": iclddn_p,
         "radlu_sfc": radlu_sfc, "radclru_sfc": radclru_sfc,
         "ngs": ngs, "delwave": delwave,
         "totuflux": totuflux, "totdflux": totdflux,
         "totuclfl": totuclfl, "totdclfl": totdclfl},
        {"ncol": ncol, "nl": nl, "wtdiff": np.float32(wtdiff)},
        exact_policy=False, grid_size=ncol * (nl + 1), threads_per_group=128)


def submit_rtrn_final(pz, totuflux, totdflux, totuclfl, totdclfl,
                      fnet, fnetc, htr, htrc, *, fluxfac, heatfac):
    """Scale RRTMG LW fluxes and compute net fluxes and heating rates."""
    arrays = (pz, totuflux, totdflux, totuclfl, totdclfl, fnet, fnetc, htr, htrc)
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError("legacy RRTMG LW finalization requires MetalArray resources")
    context = pz.buffer.context
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError("legacy RRTMG LW finalization resources must share a context")
    if any(not value.flags.c_contiguous for value in arrays):
        raise ValueError("legacy RRTMG LW finalization resources must be contiguous")
    ncol, nlev = pz.shape if pz.ndim == 2 else (0, 0)
    if (ncol < 1 or nlev < 2
            or any(value.shape != (ncol, nlev) for value in arrays[1:])):
        raise ValueError("legacy RRTMG LW finalization resource shapes disagree")
    if any(value.dtype != np.dtype(np.float32) for value in arrays):
        raise TypeError("legacy RRTMG LW finalization resources must be FP32")
    return context.submit_kernel(
        "rrtmg_legacy_lw_rtrn_final",
        {"pz": pz, "totuflux": totuflux, "totdflux": totdflux,
         "totuclfl": totuclfl, "totdclfl": totdclfl, "fnet": fnet,
         "fnetc": fnetc, "htr": htr, "htrc": htrc},
        {"ncol": ncol, "nl": nlev - 1,
         "fluxfac": np.float32(fluxfac), "heatfac": np.float32(heatfac)},
        exact_policy=False, grid_size=ncol, threads_per_group=128)


def submit_rtrn_march(cldfmc, odcld, efclfrac, icldlyr, secdiff, semiss,
                      planklay, planklev, plankbnd, fracs, taut, tau_tbl,
                      exp_tbl, tfn_tbl, ngb, radld_p, radclrd_p, radlu_p,
                      radclru_p, iclddn_p, radlu_sfc, radclru_sfc, *, bpade):
    """Run the per-g-point upward/downward RRTMG longwave flux march."""
    arrays = (cldfmc, odcld, efclfrac, icldlyr, secdiff, semiss, planklay,
              planklev, plankbnd, fracs, taut, tau_tbl, exp_tbl, tfn_tbl,
              ngb, radld_p, radclrd_p, radlu_p, radclru_p, iclddn_p,
              radlu_sfc, radclru_sfc)
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError("legacy RRTMG LW march requires MetalArray resources")
    context = cldfmc.buffer.context
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError("legacy RRTMG LW march resources must share a context")
    if any(not value.flags.c_contiguous for value in arrays):
        raise ValueError("legacy RRTMG LW march resources must be contiguous")
    ncol, ngpt, nl = cldfmc.shape if cldfmc.ndim == 3 else (0, 0, 0)
    profile_shape = (ncol, ngpt, nl)
    if (ncol < 1 or ngpt != 140 or nl < 1 or nl > 128
            or any(value.shape != profile_shape for value in
                   (odcld, efclfrac, radld_p, radclrd_p, radlu_p, radclru_p, iclddn_p))
            or icldlyr.shape != (ncol, nl) or secdiff.shape != (ncol, 16)
            or semiss.shape != (ncol, 16) or planklay.shape != (ncol, nl, 16)
            or planklev.shape != (ncol, nl + 1, 16) or plankbnd.shape != (ncol, 16)
            or fracs.shape != (ncol, nl, 140) or taut.shape != fracs.shape
            or any(value.shape != (10001,) for value in (tau_tbl, exp_tbl, tfn_tbl))
            or ngb.shape != (140,) or radlu_sfc.shape != (ncol, 140)
            or radclru_sfc.shape != (ncol, 140)):
        raise ValueError("legacy RRTMG LW march resource shapes disagree")
    floats = (cldfmc, odcld, efclfrac, secdiff, semiss, planklay, planklev,
              plankbnd, fracs, taut, tau_tbl, exp_tbl, tfn_tbl, radld_p,
              radclrd_p, radlu_p, radclru_p, radlu_sfc, radclru_sfc)
    if any(value.dtype != np.dtype(np.float32) for value in floats):
        raise TypeError("legacy RRTMG LW march float resources must be FP32")
    if any(value.dtype != np.dtype(np.int32) for value in
           (icldlyr, ngb, iclddn_p)):
        raise TypeError("legacy RRTMG LW march integer resources must be INT32")
    return context.submit_kernel(
        "rrtmg_legacy_lw_rtrn_march",
        {"cldfmc": cldfmc, "odcld": odcld, "efclfrac": efclfrac,
         "icldlyr": icldlyr, "secdiff": secdiff, "semiss": semiss,
         "planklay": planklay, "planklev": planklev, "plankbnd": plankbnd,
         "fracs": fracs, "taut": taut, "tau_tbl": tau_tbl,
         "exp_tbl": exp_tbl, "tfn_tbl": tfn_tbl, "ngb": ngb,
         "radld_p": radld_p, "radclrd_p": radclrd_p,
         "radlu_p": radlu_p, "radclru_p": radclru_p,
         "iclddn_p": iclddn_p, "radlu_sfc": radlu_sfc,
         "radclru_sfc": radclru_sfc},
        {"ncol": ncol, "nl": nl, "bpade": np.float32(bpade)},
        exact_policy=False, grid_size=ncol * 140, threads_per_group=128)


def submit_cldprmc(cldfmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc,
                   resnmc, absice1, absice2, absice3, absice0, absliq1,
                   ngb, taucmc, ncb_flag, err_flag, *, inflag, iceflag,
                   liqflag, absliq0):
    """Apply legacy RRTMG ice/liquid/snow optical tables to McICA paths."""
    arrays = (cldfmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc, resnmc,
              absice1, absice2, absice3, absice0, absliq1, ngb, taucmc,
              ncb_flag, err_flag)
    if any(not isinstance(value, MetalArray) for value in arrays):
        raise TypeError("legacy RRTMG LW cloud optics requires MetalArray resources")
    context = cldfmc.buffer.context
    if any(value.buffer.context is not context for value in arrays):
        raise ValueError("legacy RRTMG LW cloud optics must share a context")
    if any(not value.flags.c_contiguous for value in arrays):
        raise ValueError("legacy RRTMG LW cloud optics resources must be contiguous")
    ncol, ngpt, nl = cldfmc.shape if cldfmc.ndim == 3 else (0, 0, 0)
    profile = (ncol, ngpt, nl)
    if (ncol < 1 or ngpt != 140 or nl < 1
            or any(value.shape != profile for value in (ciwpmc, clwpmc, cswpmc, taucmc))
            or any(value.shape != (ncol, nl) for value in (reicmc, relqmc, resnmc))
            or absice1.shape != (10,) or absice2.shape != (688,)
            or absice3.shape != (736,) or absice0.shape != (2,)
            or absliq1.shape != (928,) or ngb.shape != (140,)
            or ncb_flag.shape != (ncol,) or err_flag.shape != (ncol,)):
        raise ValueError("legacy RRTMG LW cloud optics resource shapes disagree")
    floats = (cldfmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc, resnmc,
              absice1, absice2, absice3, absice0, absliq1, taucmc)
    if any(value.dtype != np.dtype(np.float32) for value in floats):
        raise TypeError("legacy RRTMG LW cloud optics float resources must be FP32")
    if any(value.dtype != np.dtype(np.int32) for value in (ngb, ncb_flag, err_flag)):
        raise TypeError("legacy RRTMG LW cloud optics integer resources must be INT32")
    return context.submit_kernel(
        "rrtmg_legacy_lw_cldprmc",
        {"cldfmc": cldfmc, "ciwpmc": ciwpmc, "clwpmc": clwpmc,
         "cswpmc": cswpmc, "reicmc": reicmc, "relqmc": relqmc,
         "resnmc": resnmc, "absice1": absice1, "absice2": absice2,
         "absice3": absice3, "absice0": absice0, "absliq1": absliq1,
         "ngb": ngb, "taucmc": taucmc, "ncb_flag": ncb_flag,
         "err_flag": err_flag},
        {"ncol": ncol, "nl": nl, "inflag": int(inflag),
         "iceflag": int(iceflag), "liqflag": int(liqflag),
         "absliq0": np.float32(absliq0)},
        exact_policy=False, grid_size=ncol, threads_per_group=128)
