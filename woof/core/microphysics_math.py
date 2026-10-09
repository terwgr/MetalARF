"""Shared host particle-size limits and reflectivity calculations."""

from __future__ import annotations
import math
import numpy as np
from woof.core.morrison_constants import rimed_ice_constants
import math


_MORR_PI = np.pi


_MORR_RHOW = 997.0


_MORR_RHOI = 500.0


_MORR_RHOS = 100.0


_MORR_QSMALL = 1.0e-14


_MORR_DCS = 125.0e-6


_MORR_CI = _MORR_RHOI * _MORR_PI / 6.0


_MORR_CS = _MORR_RHOS * _MORR_PI / 6.0


def _np_morrison_slopes(q, n, rho, temperature, *, pressure=None,
                        reset_cloud_number=True, morr_rimed_ice=1):
    """Return bounded lambda/number fields for the five species.

    This is the repeated PSD limiter at WRF source lines 1525-1638,
    2122-2267, and 3855-4000.  Number concentrations are per kg dry air.
    """
    q = {name: np.asarray(value, np.float64) for name, value in q.items()}
    n = {name: np.maximum(np.asarray(value, np.float64), 0.0)
         for name, value in n.items()}
    nz = np.asarray(temperature).size
    lam = {name: np.zeros(nz, np.float64)
           for name in ("c", "r", "i", "s", "g")}
    pgam = np.full(nz, 2.0, np.float64)

    # INUM=1 is imposed once on entry to the warm/cold branch.  A LAMC
    # rebound then alters NC3D at fixed QC, and that altered concentration
    # feeds PRC/MNUCCC/PGAM for the rest of the process section
    # (WRF 1493-1586, 2108-2215).  Callers doing sedimentation/final bounds
    # must not silently restore 250 cm-3 here.
    if reset_cloud_number:
        n["c"] = 250.0e6 / rho
    active = q["c"] >= _MORR_QSMALL
    if np.any(active):
        dens = (rho[active] if pressure is None else
                np.asarray(pressure, np.float64)[active]
                / (287.15 * np.asarray(temperature, np.float64)[active]))
        pg = 0.0005714 * (n["c"][active] / 1.0e6 * dens) + 0.2714
        pgam[active] = np.clip(1.0 / (pg * pg) - 1.0, 2.0, 10.0)
        raw = (_MORR_PI / 6.0 * _MORR_RHOW * n["c"][active]
               * np.vectorize(math.gamma)(pgam[active] + 4.0)
               / (q["c"][active]
                  * np.vectorize(math.gamma)(pgam[active] + 1.0))) \
            ** (1.0 / 3.0)
        lo = (pgam[active] + 1.0) / 60.0e-6
        hi = (pgam[active] + 1.0) / 1.0e-6
        bounded = np.clip(raw, lo, hi)
        lam["c"][active] = bounded
        rebounded = raw != bounded
        if np.any(rebounded):
            aa = np.flatnonzero(active)[rebounded]
            n["c"][aa] = (bounded[rebounded] ** 3 * q["c"][aa]
                            * np.vectorize(math.gamma)(pgam[aa] + 1.0)
                            / (np.vectorize(math.gamma)(pgam[aa] + 4.0)
                               * (_MORR_PI / 6.0 * _MORR_RHOW)))

    rimed = rimed_ice_constants(morr_rimed_ice)
    specs = {
        "r": (_MORR_PI * _MORR_RHOW, 1.0 / 2800.0e-6,
              1.0 / 20.0e-6),
        "i": (6.0 * _MORR_CI, 1.0 / (2.0 * _MORR_DCS + 100.0e-6),
              1.0 / 1.0e-6),
        "s": (6.0 * _MORR_CS, 1.0 / 2000.0e-6, 1.0 / 10.0e-6),
        "g": (6.0 * rimed.cg, 1.0 / 2000.0e-6, 1.0 / 20.0e-6),
    }
    for name, (six_c, lo, hi) in specs.items():
        active = q[name] >= _MORR_QSMALL
        raw = np.zeros(nz)
        raw[active] = np.cbrt(six_c * n[name][active] / q[name][active])
        lam[name][active] = np.clip(raw[active], lo, hi)
        # q = Gamma(4)*C*N/lambda^3 = 6*C*N/lambda^3.
        n[name][active] = q[name][active] * lam[name][active] ** 3 / six_c
        n[name][~active] = 0.0
    return lam, pgam, n


def _np_refl_maxwellgarnett(vol1, vol2, vol3, m1, m2, m3):
    """Maxwell-Garnett effective refractive index, spheroidal inclusions.

    Mirror of ``m_complex_maxwellgarnett`` (module_mp_radar.F:544-590) on
    the only inclusion string radar_init configures ('spheroidal',
    :106-118, betas :574-576); the volume-closure check is the
    Fortran's 1e-6 guard (:558).
    """
    if abs(vol1 + vol2 + vol3 - 1.0) > 1.0e-6:
        raise ValueError("partial volume fractions must sum to 1")
    m1t = m1 * m1
    m2t = m2 * m2
    m3t = m3 * m3
    beta2 = (2.0 * m1t / (m2t - m1t)
             * (m2t / (m2t - m1t) * np.log(m2t / m1t) - 1.0))
    beta3 = (2.0 * m1t / (m3t - m1t)
             * (m3t / (m3t - m1t) * np.log(m3t / m1t) - 1.0))
    return np.sqrt(((1.0 - vol2 - vol3) * m1t + vol2 * beta2 * m2t
                    + vol3 * beta3 * m3t)
                   / (1.0 - vol2 - vol3 + vol2 * beta2 + vol3 * beta3))


def _np_refl_m_mix_nested(m_a, m_i, m_w, volair, volice, volwater):
    """``get_m_mix_nested`` on radar_init's Morrison string set.

    host='air', matrix='water', inclusion='spheroidal',
    hostmatrix='icewater', hostinclusion='spheroidal'
    (module_mp_radar.F:106-118), which selects exactly two nested
    Maxwell-Garnett evaluations along the host='air' branch (:384-413): the ice-in-water inclusion
    mix, then that mixture as 'ice' inclusions in the air host.
    """
    vol1 = volice / max(volice + volwater, 1.0e-10)     # :391
    vol2 = 1.0 - vol1
    # get_m_mix(..., 0.0, vol1, vol2, 'water', 'spheroidal') resolves to
    # m_complex_maxwellgarnett(volwater, volair, volice, m_w, m_a, m_i)
    # (:517-519).
    mtmp = _np_refl_maxwellgarnett(vol2, 0.0, vol1, m_w, m_a, m_i)
    # hostmatrix='icewater' -> get_m_mix(m_a, mtmp, 2*m_a, volair,
    # 1-volair, 0.0, 'ice', 'spheroidal') (:402-406) resolves to
    # m_complex_maxwellgarnett(volice, volair, volwater, m_i, m_a, m_w)
    # (:514-516) with the mixture as the ice member.
    return _np_refl_maxwellgarnett(1.0 - volair, volair, 0.0,
                                   mtmp, m_a, 2.0 * m_a)


def _np_refl_rayleigh_soak_wetgraupel(x_g, a_geo, b_geo, fmelt,
                                      meltratio_outside, m_w, m_i, rc):
    """Mirror of ``rayleigh_soak_wetgraupel`` (module_mp_radar.F:265-358).

    Backscattering cross section (m2) of one melting snow/graupel
    particle of ice mass ``x_g`` (kg): meltwater fraction ``fmelt`` sits
    90% on the surface (``meltratio_outside``), converging linearly to a
    water drop as fm -> 1; the ice-air-water mixture index comes from the
    nested Maxwell-Garnett path above.  ``rc`` is :func:`radar_init`.
    """
    fm = min(max(fmelt, 0.0), 1.0)                       # :290
    mra = min(max(meltratio_outside, 0.0), 1.0)          # :292
    mra = mra + (1.0 - mra) * fm                         # :299
    x_w = x_g * fm                                       # :301
    d_g = a_geo * x_g ** b_geo                           # :303
    if d_g < 1.0e-12:                                    # :305/:354-356
        return 0.0
    pix = 3.1415926535897932384626434                    # :283
    vg = pix / 6.0 * d_g ** 3                            # :307
    rhog = min(max(x_g / vg, 10.0), 900.0)               # :308
    vg = x_g / rhog                                      # :309
    meltratio_outside_grenz = 1.0 - rhog / 1000.0        # :311
    if mra <= meltratio_outside_grenz:                   # :313-317
        volg = vg * (1.0 - mra * fm)
    else:                                                # :319-331
        fmgrenz = ((900.0 - rhog)
                   / (mra * 900.0 - rhog + 900.0 * rhog / 1000.0))
        if fm <= fmgrenz:
            volg = (1.0 - mra * fm) * vg
        else:
            volg = (x_g - x_w) / 900.0 + x_w / 1000.0
    d_large = (6.0 / pix * volg) ** (1.0 / 3.0)          # :335
    volice = (x_g - x_w) / (volg * 900.0)                # :336
    volwater = x_w / (1000.0 * volg)                     # :337
    volair = 1.0 - volice - volwater                     # :338
    m_core = _np_refl_m_mix_nested(complex(1.0, 0.0), m_i, m_w,
                                   volair, volice, volwater)
    return (abs((m_core ** 2 - 1.0) / (m_core ** 2 + 2.0)) ** 2
            * rc.pi5 * d_large ** 6 / rc.lamda4)         # :351-352


def np_refl10cm_morrison_column(qv, qr, nr, qs, ns, qg, ng, t, p, *,
                                morr_rimed_ice=1):
    """Float64 one-column mirror of Morrison's radar reflectivity.

    ``refl10cm_hm`` (module_mp_morr_two_moment.F:4502-4675) with the
    wrapper's floor ``MAX(-35., dBZ)`` applied (:913-917): Rayleigh
    10 cm equivalent reflectivity (dBZ, shape ``(nz,)``) from rain,
    snow, and graupel using the scheme's own m(D) parameters and
    PROGNOSTIC number concentrations, with the Blahak water-coated
    melting treatment below the highest melting level.  Inputs are
    ``(nz,)`` bottom-to-top columns: mixing ratios kg/kg, numbers kg-1,
    air temperature K, full pressure Pa.  Setup-time constants come from
    :func:`woof.core.refl.radar_init` (WRF computes them once at
    scheme init; the per-column math here is independent of the kernel).
    ``morr_rimed_ice`` selects the same hail/graupel CG used by the scheme.
    """
    from woof.core.refl import (MELT_OUTSIDE_G, MELT_OUTSIDE_S, NRBINS,
                                 radar_init)

    rc = radar_init(morr_rimed_ice)
    names = ("qv", "qr", "nr", "qs", "ns", "qg", "ng", "t", "p")
    a = {name: np.asarray(value, np.float64).copy()
         for name, value in zip(names, (qv, qr, nr, qs, ns, qg, ng, t, p))}
    nz = a["t"].size
    if any(value.shape != (nz,) for value in a.values()):
        raise ValueError("all reflectivity column inputs must be (nz,)")

    # WRF gates activity only on mass before using the number moment in a
    # fractional-power slope (:4544-4584).  Its production scheme supplies
    # positive bounded moments (:1528-1635); explicitly retain non-finite
    # output for invalid active pairs so a corrupt input cannot masquerade
    # as the wrapper's meteorological -35 dBZ clear-air floor.
    invalid_moment = np.zeros(nz, bool)
    for mass, number in (("qr", "nr"), ("qs", "ns"), ("qg", "ng")):
        invalid_moment |= ((a[mass] > 1.0e-9)
                           & ((a[number] <= 0.0)
                              | ~np.isfinite(a[number])))

    temp = a["t"]
    qvc = np.maximum(1.0e-10, a["qv"])                   # :4540
    pres = a["p"]
    # :4542 -- the routine's own moist density (literal 0.622, not EP2).
    rho = 0.622 * pres / (287.0 * temp * (qvc + 0.622))

    rr = np.full(nz, 1.0e-12)
    rs = np.full(nz, 1.0e-12)
    rg = np.full(nz, 1.0e-12)
    ilamr = np.zeros(nz)
    ilams = np.zeros(nz)
    ilamg = np.zeros(nz)
    n0_r = np.zeros(nz)
    n0_s = np.zeros(nz)
    n0_g = np.zeros(nz)
    l_qr = np.zeros(nz, bool)
    l_qs = np.zeros(nz, bool)
    l_qg = np.zeros(nz, bool)
    for k in range(nz):
        if a["qr"][k] > 1.0e-9:                          # :4544-4556
            rr[k] = a["qr"][k] * rho[k]
            nrk = a["nr"][k] * rho[k]
            lamr = (rc.xam_r * rc.xcrg[2] * rc.xorg2
                    * nrk / rr[k]) ** rc.xobmr
            ilamr[k] = 1.0 / lamr
            n0_r[k] = nrk * rc.xorg2 * lamr ** rc.xcre[1]
            l_qr[k] = True
        if a["qs"][k] > 1.0e-9:                          # :4558-4570
            rs[k] = a["qs"][k] * rho[k]
            nsk = a["ns"][k] * rho[k]
            lams = (rc.xam_s * rc.xcsg[2] * rc.xosg2
                    * nsk / rs[k]) ** rc.xobms
            ilams[k] = 1.0 / lams
            n0_s[k] = nsk * rc.xosg2 * lams ** rc.xcse[1]
            l_qs[k] = True
        if a["qg"][k] > 1.0e-9:                          # :4572-4584
            rg[k] = a["qg"][k] * rho[k]
            ngk = a["ng"][k] * rho[k]
            lamg = (rc.xam_g * rc.xcgg[2] * rc.xogg2
                    * ngk / rg[k]) ** rc.xobmg
            ilamg[k] = 1.0 / lamg
            n0_g[k] = ngk * rc.xogg2 * lamg ** rc.xcge[1]
            l_qg[k] = True

    # :4586-4599 -- highest warm rainy level with frozen species above;
    # k_0 is the level just above it.
    melti = False
    k_0 = 0
    for k in range(nz - 2, -1, -1):
        if temp[k] > 273.15 and l_qr[k] and (l_qs[k + 1] or l_qg[k + 1]):
            k_0 = max(k + 1, k_0)
            melti = True
            break

    # :4601-4620 -- Rayleigh integrals; dry snow/graupel carry the
    # (0.176/0.93)*(6/pi)^2*(am/900)^2 density/dielectric adjustment.
    pi_m = 3.1415926535897932384626434
    ze_rain = np.full(nz, 1.0e-22)
    ze_snow = np.full(nz, 1.0e-22)
    ze_graupel = np.full(nz, 1.0e-22)
    for k in range(nz):
        if l_qr[k]:
            ze_rain[k] = n0_r[k] * rc.xcrg[3] * ilamr[k] ** rc.xcre[3]
        if l_qs[k]:
            ze_snow[k] = ((0.176 / 0.93) * (6.0 / pi_m) * (6.0 / pi_m)
                          * (rc.xam_s / 900.0) * (rc.xam_s / 900.0)
                          * n0_s[k] * rc.xcsg[3] * ilams[k] ** rc.xcse[3])
        if l_qg[k]:
            ze_graupel[k] = ((0.176 / 0.93) * (6.0 / pi_m) * (6.0 / pi_m)
                             * (rc.xam_g / 900.0) * (rc.xam_g / 900.0)
                             * n0_g[k] * rc.xcgg[3] * ilamg[k] ** rc.xcge[3])

    # :4622-4671 -- melting snow/graupel: 50-bin Simpson integration of
    # the soaked-particle backscatter, meltwater fraction from the mass
    # ratio against the k_0 reference level.
    if melti and k_0 >= 1:
        for k in range(k_0 - 1, -1, -1):
            if l_qs[k] and l_qs[k_0]:
                fmelt_s = max(0.005, min(1.0 - rs[k] / rs[k_0], 0.99))
                eta = 0.0
                lams = 1.0 / ilams[k]
                for n in range(NRBINS):
                    x = rc.xam_s * rc.xxds[n] ** 3.0     # xbm_s
                    cback = _np_refl_rayleigh_soak_wetgraupel(
                        x, rc.xocms, rc.xobms, fmelt_s, MELT_OUTSIDE_S,
                        rc.m_w_0, rc.m_i_0, rc)
                    f_d = n0_s[k] * np.exp(-lams * rc.xxds[n])  # xmu_s = 0
                    eta += f_d * cback * rc.simpson[n] * rc.xdts[n]
                ze_snow[k] = rc.lamda4 / (rc.pi5 * rc.k_w) * eta
            if l_qg[k] and l_qg[k_0]:
                fmelt_g = max(0.005, min(1.0 - rg[k] / rg[k_0], 0.99))
                eta = 0.0
                lamg = 1.0 / ilamg[k]
                for n in range(NRBINS):
                    x = rc.xam_g * rc.xxdg[n] ** 3.0     # xbm_g
                    cback = _np_refl_rayleigh_soak_wetgraupel(
                        x, rc.xocmg, rc.xobmg, fmelt_g, MELT_OUTSIDE_G,
                        rc.m_w_0, rc.m_i_0, rc)
                    f_d = n0_g[k] * np.exp(-lamg * rc.xxdg[n])  # xmu_g = 0
                    eta += f_d * cback * rc.simpson[n] * rc.xdtg[n]
                ze_graupel[k] = rc.lamda4 / (rc.pi5 * rc.k_w) * eta

    dbz = 10.0 * np.log10((ze_rain + ze_snow + ze_graupel)
                          * 1.0e18)                      # :4670-4672
    out = np.maximum(-35.0, dbz)                         # wrapper :916
    out[invalid_moment] = np.nan
    return out


def np_refl10cm_wsm6_column(qv, qr, qs, qg, t, p, *, hail_opt=0):
    """Float64 one-column mirror of WRF v4.6.1 ``refl10cm_wsm6``.

    WSM6 diagnoses fixed-intercept exponential rain/snow/rimed-ice PSDs,
    then applies the same 50-bin Blahak melting-particle calculation used by
    Morrison.  Inputs are bottom-to-top ``(nz,)`` arrays.  ``hail_opt``
    selects WRF's graupel (0) or hail (1) density and intercept.
    """
    from woof.core.refl import (MELT_OUTSIDE_G, MELT_OUTSIDE_S, NRBINS,
                                 radar_init_wsm6)
    from woof.core.wsm6_constants import rimed_ice_constants

    rc = radar_init_wsm6(hail_opt)
    rimed = rimed_ice_constants(hail_opt)
    names = ("qv", "qr", "qs", "qg", "t", "p")
    a = {name: np.asarray(value, np.float64).copy()
         for name, value in zip(names, (qv, qr, qs, qg, t, p))}
    nz = a["t"].size
    if any(value.shape != (nz,) for value in a.values()):
        raise ValueError("all reflectivity column inputs must be (nz,)")

    temp = a["t"]
    qvc = np.maximum(1.0e-10, a["qv"])
    rho = 0.622 * a["p"] / (287.0 * temp * (qvc + 0.622))
    rr = np.full(nz, 1.0e-12)
    rs = np.full(nz, 1.0e-12)
    rg = np.full(nz, 1.0e-12)
    ilamr = np.zeros(nz)
    ilams = np.zeros(nz)
    ilamg = np.zeros(nz)
    # WSM6's snow intercept increases exponentially below freezing.
    temp_c = np.minimum(-0.001, temp - 273.15)
    n0_s = np.minimum(1.0e11, 2.0e6 * np.exp(-0.12 * temp_c))
    n0_r = 8.0e6
    n0_g = float(rimed.n0g)
    l_qr = a["qr"] > 1.0e-9
    l_qs = a["qs"] > 1.0e-9
    l_qg = a["qg"] > 1.0e-9
    for k in range(nz):
        if l_qr[k]:
            rr[k] = a["qr"][k] * rho[k]
            # Fixed-intercept WSM6 solves the third mass moment, hence
            # lambda exponent 1/(1+xbm)=1/4 (not Morrison's 1/xbm).
            lamr = (rc.xam_r * rc.xcrg[2] * n0_r / rr[k]) ** 0.25
            ilamr[k] = 1.0 / lamr
        if l_qs[k]:
            rs[k] = a["qs"][k] * rho[k]
            lams = (rc.xam_s * rc.xcsg[2] * n0_s[k] / rs[k]) ** 0.25
            ilams[k] = 1.0 / lams
        if l_qg[k]:
            rg[k] = a["qg"][k] * rho[k]
            lamg = (rc.xam_g * rc.xcgg[2] * n0_g / rg[k]) ** 0.25
            ilamg[k] = 1.0 / lamg

    melti = False
    k_0 = 0
    for k in range(nz - 2, -1, -1):
        if temp[k] > 273.15 and l_qr[k] and (l_qs[k + 1] or l_qg[k + 1]):
            k_0 = max(k + 1, k_0)
            melti = True
            break

    pi_m = 3.1415926535897932384626434
    ze_rain = np.full(nz, 1.0e-22)
    ze_snow = np.full(nz, 1.0e-22)
    ze_graupel = np.full(nz, 1.0e-22)
    for k in range(nz):
        if l_qr[k]:
            ze_rain[k] = n0_r * rc.xcrg[3] * ilamr[k] ** rc.xcre[3]
        if l_qs[k]:
            ze_snow[k] = ((0.176 / 0.93) * (6.0 / pi_m) ** 2
                          * (rc.xam_s / 900.0) ** 2 * n0_s[k]
                          * rc.xcsg[3] * ilams[k] ** rc.xcse[3])
        if l_qg[k]:
            ze_graupel[k] = ((0.176 / 0.93) * (6.0 / pi_m) ** 2
                             * (rc.xam_g / 900.0) ** 2 * n0_g
                             * rc.xcgg[3] * ilamg[k] ** rc.xcge[3])

    if melti and k_0 >= 1:
        for k in range(k_0 - 1, -1, -1):
            if l_qs[k] and l_qs[k_0]:
                fmelt = max(0.005, min(1.0 - rs[k] / rs[k_0], 0.99))
                eta = 0.0
                lams = 1.0 / ilams[k]
                for n in range(NRBINS):
                    x = rc.xam_s * rc.xxds[n] ** 3.0
                    cback = _np_refl_rayleigh_soak_wetgraupel(
                        x, rc.xocms, rc.xobms, fmelt, MELT_OUTSIDE_S,
                        rc.m_w_0, rc.m_i_0, rc)
                    eta += (n0_s[k] * np.exp(-lams * rc.xxds[n]) * cback
                            * rc.simpson[n] * rc.xdts[n])
                ze_snow[k] = rc.lamda4 / (rc.pi5 * rc.k_w) * eta
            if l_qg[k] and l_qg[k_0]:
                fmelt = max(0.005, min(1.0 - rg[k] / rg[k_0], 0.99))
                eta = 0.0
                lamg = 1.0 / ilamg[k]
                for n in range(NRBINS):
                    x = rc.xam_g * rc.xxdg[n] ** 3.0
                    cback = _np_refl_rayleigh_soak_wetgraupel(
                        x, rc.xocmg, rc.xobmg, fmelt, MELT_OUTSIDE_G,
                        rc.m_w_0, rc.m_i_0, rc)
                    eta += (n0_g * np.exp(-lamg * rc.xxdg[n]) * cback
                            * rc.simpson[n] * rc.xdtg[n])
                ze_graupel[k] = rc.lamda4 / (rc.pi5 * rc.k_w) * eta

    dbz = 10.0 * np.log10((ze_rain + ze_snow + ze_graupel) * 1.0e18)
    return np.maximum(-35.0, dbz)


def np_refl10cm_kessler_column(qv, qr, t, p):
    """Float64 mirror of the Kessler rain-only reflectivity fallback.

    WRF's Kessler has no refl_10cm diagnostic; this is the Smith et al.
    (1975) rain form for an exponential PSD with the fixed Marshall-Palmer
    intercept N0r = 8e6 m-4 and rho_w = 1000 kg m-3 that Kessler's own
    fall-speed closure assumes (derivation in woof/core/refl.py):
    Ze = Gamma(7)*N0r*lambda**-7 with lambda = (pi*rho_w*N0r/(rho*qr))**0.25.
    Density diagnosis, the 1e-9 kg/kg threshold, the 1e-22 ze floor, and
    the -35 dBZ output floor follow ``refl10cm_hm`` so both schemes'
    outputs share one convention.
    """
    names = ("qv", "qr", "t", "p")
    a = {name: np.asarray(value, np.float64).copy()
         for name, value in zip(names, (qv, qr, t, p))}
    nz = a["t"].size
    if any(value.shape != (nz,) for value in a.values()):
        raise ValueError("all reflectivity column inputs must be (nz,)")
    qvc = np.maximum(1.0e-10, a["qv"])
    rho = 0.622 * a["p"] / (287.0 * a["t"] * (qvc + 0.622))
    pi_m = 3.1415926535897932384626434
    n0r = 8.0e6
    rhow = 1000.0
    ze = np.full(nz, 1.0e-22)
    active = a["qr"] > 1.0e-9
    lam = (pi_m * rhow * n0r / (rho[active] * a["qr"][active])) ** 0.25
    ze[active] = math.gamma(7.0) * n0r * lam ** -7.0
    return np.maximum(-35.0, 10.0 * np.log10(ze * 1.0e18))
