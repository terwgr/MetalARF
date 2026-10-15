"""Metal orchestration for the legacy RRTMG longwave column solver.

The caller supplies the already prepared WRF columns produced by
``rrtmg_legacy_prep.lwrad_prep_batch``.  Cloud-path preparation remains on
the host, as it does in the CUDA implementation; the optical-depth and
radiative-transfer chain runs on the owning Metal context.
"""

from __future__ import annotations

import inspect
import numpy as np

from woof.metal.array import MetalArrayModule
from woof.metal import rrtmg_legacy_lw_family as _lw_metal
from woof.core import rrtmg_lw as _lw


def run_lw_prepared(context, prepared, coefficients):
    """Run one prepared batch through the legacy LW Metal kernels.

    Arrays returned by ``lwrad_prep_batch`` have leading column dimensions;
    the McICA cloud slabs are transposed into the frozen CUDA-compatible
    ``(ncol, 140, nl)`` layout before upload.
    """
    xp = MetalArrayModule(context)
    ncol = int(prepared["ncol"])
    nl = int(prepared["nlay"])
    if ncol < 1 or nl < 1 or nl > 128:
        raise ValueError("legacy RRTMG LW Metal batch dimensions are unsupported")

    def upload(value, dtype=np.float32):
        return xp.asarray(np.ascontiguousarray(np.asarray(value, dtype=dtype)))

    def coeff(key, *, fortran=False, dtype=np.float32):
        value = np.asarray(coefficients[key], dtype=dtype)
        if fortran:
            value = value.ravel(order="F")
        return upload(value, dtype)

    def ctab(band, name):
        return coeff("kg%02d/%s" % (band, name), fortran=True)

    plev = upload(prepared["plev"])
    play = upload(prepared["play"])
    tlay = upload(prepared["tlay"])
    tlev = upload(prepared["tlev"])
    tsfc = upload(prepared["tsfc"])
    emis = upload(prepared["emis"])
    cloud = {}
    for name in ("cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl"):
        value = np.asarray(prepared[name], dtype=np.float32)
        if value.shape == (140, ncol, nl):
            value = value.transpose(1, 0, 2)
        elif value.shape == (ncol, nl, 140):
            value = value.transpose(0, 2, 1)
        if value.shape != (ncol, 140, nl):
            raise ValueError(f"prepared {name} must have shape (ncol, nl, 140)")
        cloud[name] = upload(value)
    reic = upload(prepared["reicmcl"])
    relq = upload(prepared["relqmcl"])
    resn = upload(prepared["resnmcl"])
    ngb = coeff("wvn/ngb", dtype=np.int32)

    # Molecular slabs match RRTMG's fixed slot order. Unused species and
    # cross-section slots remain explicit positive zero.
    wkl_h = np.zeros((ncol, 35, nl), dtype=np.float32)
    for slot, key in ((1, "h2ovmr"), (2, "co2vmr"), (3, "o3vmr"),
                      (4, "n2ovmr"), (6, "ch4vmr"), (7, "o2vmr")):
        wkl_h[:, slot - 1, :] = prepared[key]
    wx_h = np.zeros((ncol, _lw.MAXXSEC, nl), dtype=np.float32)
    for slot, key in ((1, "ccl4vmr"), (2, "cfc11vmr"),
                      (3, "cfc12vmr"), (4, "cfc22vmr")):
        wx_h[:, slot - 1, :] = prepared[key]
    wkl, wx = upload(wkl_h), upload(wx_h)
    coldry = xp.empty((ncol, nl), np.float32)
    wbrodl = xp.empty((ncol, nl), np.float32)
    pwvcm = xp.empty((ncol,), np.float32)
    _lw_metal.submit_inatm(plev, wkl, wx, coldry, wbrodl, pwvcm,
                           grav=coefficients["con/grav"],
                           avogad=coefficients["con/avogad"])

    # SETCOEF pressure/temperature indices, fractions, gas factors, and
    # Planck emission tables. FS/IS keep the solver's fixed 33/6 slot axes.
    fs = xp.zeros((len(_lw.GPU_FSLOTS), ncol, nl), np.float32)
    isv = xp.zeros((len(_lw.GPU_ISLOTS), ncol, nl), np.int32)
    jp, jt, jt1 = (xp.zeros((ncol, nl), np.int32) for _ in range(3))
    fp, ft, ft1 = (xp.empty((ncol, nl), np.float32) for _ in range(3))
    laytrop = xp.empty((ncol,), np.int32)
    _lw_metal.submit_setcoef_indices(
        play, tlay, coeff("ref/preflog"), coeff("ref/tref"),
        jp, jt, jt1, fp, ft, ft1, laytrop)
    _lw_metal.submit_setcoef_gas(
        play, tlay, coldry, wkl, wbrodl, coeff("ref/chi_mls", fortran=True),
        jp, jt, jt1, fp, ft, ft1, fs, isv)
    planklay = xp.empty((ncol, nl, 16), np.float32)
    planklev = xp.empty((ncol, nl + 1, 16), np.float32)
    plankbnd = xp.empty((ncol, 16), np.float32)
    _lw_metal.submit_planck(
        tlay, tlev, tsfc, emis, coeff("wvn/totplnk", fortran=True),
        coeff("wvn/totplk16"), planklay, planklev, plankbnd)

    taug = xp.empty((ncol, nl, 140), np.float32)
    fracs = xp.empty((ncol, nl, 140), np.float32)
    for band in range(1, 17):
        function = getattr(_lw_metal, f"submit_taugb{band}")
        kwargs = {}
        for parameter in inspect.signature(function).parameters:
            if parameter in {"laytrop", "fs", "isv", "taug", "fracs"}:
                kwargs[parameter] = locals()[parameter]
            elif parameter == "wx":
                # Column amounts scaled by INATM, not the prepared ratios.
                kwargs[parameter] = wx
            elif parameter == "chi_mls":
                kwargs[parameter] = coeff("ref/chi_mls", fortran=True)
            elif parameter == "oneminus":
                kwargs[parameter] = coefficients["con/oneminus"]
            elif parameter == "*":
                continue
            elif parameter not in {"self"}:
                kwargs[parameter] = ctab(band, parameter)
        function(**kwargs)

    taua = xp.zeros((ncol, nl, 16), np.float32)
    taut = xp.empty((ncol, nl, 140), np.float32)
    _lw_metal.submit_taut(taug, taua, ngb, taut)

    cld_tau = cloud["taucmcl"]
    ncb_flag = xp.zeros((ncol,), np.int32)
    err_flag = xp.zeros((ncol,), np.int32)
    _lw_metal.submit_cldprmc(
        cloud["cldfmcl"], cloud["ciwpmcl"], cloud["clwpmcl"],
        cloud["cswpmcl"], reic, relq, resn,
        coeff("cld/absice1", fortran=True), coeff("cld/absice2", fortran=True),
        coeff("cld/absice3", fortran=True), coeff("cld/absice0"),
        coeff("cld/absliq1", fortran=True), ngb, cld_tau, ncb_flag,
        err_flag, inflag=prepared["inflglw"], iceflag=prepared["iceflglw"],
        liqflag=prepared["liqflglw"], absliq0=coefficients["cld/absliq0"])

    secdiff = xp.empty((ncol, 16), np.float32)
    _lw_metal.submit_rtrn_secdiff(pwvcm, upload(_lw.A0), upload(_lw.A1),
                                  upload(_lw.A2), secdiff)
    odcld = xp.empty((ncol, 140, nl), np.float32)
    efclfrac = xp.empty((ncol, 140, nl), np.float32)
    icldlyr = xp.zeros((ncol, nl), np.int32)
    _lw_metal.submit_rtrn_prol(cloud["cldfmcl"], cld_tau, secdiff, ngb,
                               odcld, efclfrac, icldlyr)
    radiance = [xp.empty((ncol, 140, nl), np.float32) for _ in range(4)]
    iclddn = xp.empty((ncol, 140, nl), np.int32)
    radlu_sfc = xp.empty((ncol, 140), np.float32)
    radclru_sfc = xp.empty((ncol, 140), np.float32)
    _lw_metal.submit_rtrn_march(
        cloud["cldfmcl"], odcld, efclfrac, icldlyr, secdiff, emis,
        planklay, planklev, plankbnd, fracs, taut,
        coeff("tbl/tau_tbl"), coeff("tbl/exp_tbl"), coeff("tbl/tfn_tbl"),
        ngb, *radiance, iclddn, radlu_sfc, radclru_sfc,
        bpade=coefficients["tbl/bpade"])
    fluxes = [xp.empty((ncol, nl + 1), np.float32) for _ in range(4)]
    _lw_metal.submit_rtrn_accum(
        *radiance, iclddn, radlu_sfc, radclru_sfc,
        upload(_lw.NGS, dtype=np.int32), coeff("wvn/delwave"), *fluxes,
        wtdiff=_lw.WTDIFF)
    fnet = xp.empty((ncol, nl + 1), np.float32)
    fnetc = xp.empty((ncol, nl + 1), np.float32)
    htr = xp.empty((ncol, nl + 1), np.float32)
    htrc = xp.empty((ncol, nl + 1), np.float32)
    _lw_metal.submit_rtrn_final(
        plev, *fluxes, fnet, fnetc, htr, htrc,
        fluxfac=coefficients["con/fluxfac"],
        heatfac=coefficients["con/heatfac"])
    errors = err_flag.to_numpy()
    failed = np.flatnonzero(errors)
    if failed.size:
        raise ValueError(f"rlw_cldprmc Metal abort, code {int(errors[failed[0]])}")
    return {"uflx": fluxes[0], "dflx": fluxes[1], "uflxc": fluxes[2],
            "dflxc": fluxes[3], "hr": htr, "hrc": htrc}
