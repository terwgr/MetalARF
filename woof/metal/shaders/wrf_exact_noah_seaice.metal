#include <metal_stdlib>
#include "exact_gfk_math.metalh"
#include "wrf_exact_noah_seaice_bindings.metalh"
using namespace metal;
static inline bool operator<(Exact32 a, Exact32 b) { return soft32_less(a.bits,b.bits); }
static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_less_equal(a.bits,b.bits); }
static inline bool operator==(Exact32 a, Exact32 b) { return soft32_equal(a.bits,b.bits); }
static inline bool operator!=(Exact32 a, Exact32 b) { return !soft32_equal(a.bits,b.bits); }
 static Exact32 nglac_csnow(Exact32 dsnow)
{
    Exact32 c = exact_mul(0.328f, exact_gfk_pow(10.0f, exact_mul(2.25f, dsnow)));
    return exact_mul(exact_mul(2.0f, 0.11631f), c);
}

 static void nglac_snow_new(Exact32 temp, Exact32 newsn, thread Exact32& snowh,
                                      thread Exact32& sndens)
{
    Exact32 tempc = exact_sub(temp, 273.15f);
    Exact32 dsnew;
    if (tempc <= -15.0f) dsnew = 0.05f;
    else dsnew = exact_add(0.05f, exact_mul(0.0017f,
                     exact_gfk_pow(exact_add(tempc, 15.0f), 1.5f)));
    Exact32 snowhc = exact_mul(snowh, 100.0f);
    Exact32 newsnc = exact_mul(newsn, 100.0f);
    Exact32 hnewc = exact_div(newsnc, dsnew);
    if (exact_add(snowhc, hnewc) < 1.0e-3f) {
        sndens = exact_max(dsnew, sndens);
    } else {
        sndens = exact_div(exact_add(exact_mul(snowhc, sndens),
                                     exact_mul(hnewc, dsnew)),
                           exact_add(snowhc, hnewc));
    }
    snowhc = exact_add(snowhc, hnewc);
    snowh = exact_mul(snowhc, 0.01f);
}

 static void nice_hstep(thread Exact32* stc, Exact32 dt, thread Exact32* rhsts, thread Exact32* ai,
                                  thread Exact32* bi, thread Exact32* ci)
{
    for (int k = 0; k < 4; ++k) {
        rhsts[k] = exact_mul(rhsts[k], dt);
        ai[k] = exact_mul(ai[k], dt);
        bi[k] = exact_add(1.0f, exact_mul(bi[k], dt));
        ci[k] = exact_mul(ci[k], dt);
    }
    Exact32 p[4], delta[4], c[4];
    for (int k = 0; k < 4; ++k) c[k] = ci[k];
    c[4 - 1] = 0.0f;
    p[0] = exact_div(-c[0], bi[0]);
    delta[0] = exact_div(rhsts[0], bi[0]);
    for (int k = 1; k < 4; ++k) {
        Exact32 rb = exact_div(1.0f, exact_add(bi[k], exact_mul(ai[k], p[k - 1])));
        p[k] = exact_mul(-c[k], rb);
        delta[k] = exact_mul(exact_sub(rhsts[k], exact_mul(ai[k], delta[k - 1])),
                             rb);
    }
    p[4 - 1] = delta[4 - 1];
    for (int k = 1; k < 4; ++k) {
        int kk = 4 - k - 1;
        p[kk] = exact_add(exact_mul(p[kk], p[kk + 1]), delta[kk]);
    }
    for (int k = 0; k < 4; ++k) stc[k] = exact_add(stc[k], p[k]);
}

 static void nglac_snowpack(Exact32 sneqv, Exact32 dtsec, thread Exact32& snowh,
                                      thread Exact32& sndens, Exact32 tsnow, Exact32 tsoil)
{
    Exact32 esdc = exact_mul(sneqv, 100.0f);
    Exact32 dthr = exact_div(dtsec, 3600.0f);
    Exact32 tsnowc = exact_sub(tsnow, 273.15f);
    Exact32 tsoilc = exact_sub(tsoil, 273.15f);
    Exact32 tavgc = exact_mul(0.5f, exact_add(tsnowc, tsoilc));
    Exact32 esdcx = (esdc > 1.0e-2f) ? esdc : 1.0e-2f;
    Exact32 bfac = exact_mul(exact_mul(dthr, 0.01f),
                          exact_gfk_exp(exact_sub(exact_mul(0.08f, tavgc),
                                            exact_mul(21.0f, sndens))));
    Exact32 pexp = 0.0f;
    for (int j = 4; j >= 1; --j)
        pexp = exact_div(exact_mul(exact_mul(exact_add(1.0f, pexp), bfac),
                                   esdcx), (Exact32)(j + 1));
    pexp = exact_add(pexp, 1.0f);
    Exact32 dsx = exact_mul(sndens, pexp);
    if (dsx > 0.40f) dsx = 0.40f;
    if (dsx < 0.05f) dsx = 0.05f;
    sndens = dsx;
    if (tsnowc >= 0.0f) {
        Exact32 dw = exact_div(exact_mul(0.13f, dthr), 24.0f);
        sndens = exact_add(exact_mul(sndens, exact_sub(1.0f, dw)), dw);
        if (sndens >= 0.40f) sndens = 0.40f;
    }
    snowh = exact_mul(exact_div(esdc, sndens), 0.01f);
}

 // Retain this boundary for source-word agreement; the inline divergence is unresolved.
static __attribute__((noinline)) void nsice_hrtice(thread Exact32* rhsts, thread const Exact32* stc, Exact32 tbot,
                                    thread const Exact32* zsoil, Exact32 yy, Exact32 zz1,
                                    Exact32 df1, thread Exact32* ai, thread Exact32* bi, thread Exact32* ci)
{
    const Exact32 hcpct = 1.72396e+6f;
    Exact32 zbot = zsoil[4 - 1];
    Exact32 ddz = exact_div(1.0f, exact_mul(-0.5f, zsoil[1]));
    ai[0] = 0.0f;
    ci[0] = exact_div(exact_mul(df1, ddz), exact_mul(zsoil[0], hcpct));
    bi[0] = exact_add(-ci[0], exact_div(df1,
                exact_mul(exact_mul(exact_mul(exact_mul(0.5f, zsoil[0]),
                                              zsoil[0]), hcpct), zz1)));
    Exact32 dtsdz = exact_div(exact_sub(stc[0], stc[1]),
                           exact_mul(-0.5f, zsoil[1]));
    Exact32 ssoil = exact_div(exact_mul(df1, exact_sub(stc[0], yy)),
                           exact_mul(exact_mul(0.5f, zsoil[0]), zz1));
    rhsts[0] = exact_div(exact_sub(exact_mul(df1, dtsdz), ssoil),
                         exact_mul(zsoil[0], hcpct));
    Exact32 ddz2 = 0.0f;
    Exact32 df1k = df1, df1n = df1;
    for (int k = 1; k < 4; ++k) {
        Exact32 dtsdz2;
        if (k != 4 - 1) {
            Exact32 denom = exact_mul(0.5f, exact_sub(zsoil[k - 1], zsoil[k + 1]));
            dtsdz2 = exact_div(exact_sub(stc[k], stc[k + 1]), denom);
            ddz2 = exact_div(2.0f, exact_sub(zsoil[k - 1], zsoil[k + 1]));
            ci[k] = -exact_div(exact_mul(df1n, ddz2),
                               exact_mul(exact_sub(zsoil[k - 1], zsoil[k]),
                                         hcpct));
        } else {
            dtsdz2 = exact_div(exact_sub(stc[k], tbot),
                               exact_sub(exact_mul(0.5f, exact_add(zsoil[k - 1],
                                                                   zsoil[k])),
                                         zbot));
            ci[k] = 0.0f;
        }
        Exact32 denom = exact_mul(exact_sub(zsoil[k], zsoil[k - 1]), hcpct);
        rhsts[k] = exact_div(exact_sub(exact_mul(df1n, dtsdz2),
                                       exact_mul(df1k, dtsdz)), denom);
        ai[k] = -exact_div(exact_mul(df1k, ddz),
                           exact_mul(exact_sub(zsoil[k - 1], zsoil[k]), hcpct));
        bi[k] = -exact_add(ai[k], ci[k]);
        df1k = df1n;
        dtsdz = dtsdz2;
        ddz = ddz2;
    }
}

  void nsice_clamp_depth(thread Exact32& sneqv, thread Exact32& snowh,
                                                  Exact32 dmin, Exact32 dmax)
{
    Exact32 sndens = exact_div(sneqv, snowh);
    snowh = exact_max(dmin, exact_min(snowh, dmax));
    sneqv = exact_mul(snowh, sndens);
}


void noah_seaice_column_body(device const Exact32*  xice_a,
                        device const Exact32*  psfc_a,
                        device const Exact32*  sfcprs_a,
                        device const Exact32*  sfctmp_a,
                        device const Exact32*  qv1_a,
                        device const Exact32*  qgh_a,
                        device const Exact32*  glw_a,
                        device const Exact32*  swdown_a,
                        device const Exact32*  rainbl_a,
                        device const Exact32*  sr_a,
                        device const Exact32*  rib_a,
                        device const Exact32*  chs_a,
                        device Exact32*  cqs2_a,
                        device Exact32*  chs2_a,
                        device Exact32*  tsk_a,
                        device Exact32*  emiss_a,
                        device Exact32*  albedo_a,
                        device Exact32*  z0_a,
                        device Exact32*  snow_a,
                        device Exact32*  snowc_a,
                        device Exact32*  snowh_a,
                        device Exact32*  znt_a,
                        device Exact32*  lh_a,
                        device Exact32*  hfx_a,
                        device Exact32*  qfx_a,
                        device Exact32*  potevp_a,
                        device Exact32*  grdflx_a,
                        device Exact32*  qsfc_a,
                        device Exact32*  acsnow_a,
                        device Exact32*  acsnom_a,
                        device Exact32*  snopcx_a,
                        device Exact32*  sfcrunoff_a,
                        device Exact32*  noahres_a,
                        device Exact32*  tslb_a,
                        device int*  fatal,
                        Exact32 dt, Exact32 xice_threshold, int frpcpn,
                        Exact32 albedo_default, Exact32 sithick,
                        Exact32 snowdepth_min, Exact32 snowdepth_max,
                        int ny, int nx, int col)
{

    if (col >= ny * nx) return;
    ulong idx = (ulong)col;
    ulong plane = (ulong)ny * nx;
    if (xice_a[idx] < xice_threshold) return;
    const Exact32 a2 = 17.67f, a3 = 273.15f, a4 = 29.65f;
    const Exact32 a23m4 = exact_mul(a2, exact_sub(a3, a4));
    Exact32 sfctmp = sfctmp_a[idx];
    Exact32 t1 = tsk_a[idx];
    Exact32 emissi = emiss_a[idx];
    Exact32 lwdn = exact_mul(glw_a[idx], emissi);
    Exact32 sneqv = exact_mul(snow_a[idx], 0.001f);
    Exact32 snowh = snowh_a[idx];
    Exact32 sncovr = snowc_a[idx];
    Exact32 solnet = exact_mul(swdown_a[idx], exact_sub(1.0f, albedo_a[idx]));
    Exact32 sfcprs = sfcprs_a[idx];
    Exact32 psfc = psfc_a[idx];
    Exact32 q2 = exact_div(qv1_a[idx], exact_add(1.0f, qv1_a[idx]));
    Exact32 apes = exact_gfk_pow(exact_div(1.0e5f, psfc), Exact32::from_bits(0x3e924925u));
    Exact32 apelm = exact_gfk_pow(exact_div(1.0e5f, sfcprs), Exact32::from_bits(0x3e924925u));
    Exact32 th2 = exact_div(exact_mul(sfctmp, apelm), apes);
    Exact32 q2sat = exact_div(qgh_a[idx], exact_add(1.0f, qgh_a[idx]));
    Exact32 sfcma4 = exact_sub(sfctmp, a4);
    Exact32 dqsdt2 = exact_div(exact_mul(q2sat, a23m4), exact_mul(sfcma4, sfcma4));
    if (soft32_greater((snow_a[idx]).bits,0u)) {
        Exact32 sfctsno = sfctmp;
        Exact32 e2sat = exact_mul(611.2f, exact_gfk_exp(exact_mul(6174.0f,
                         exact_sub(exact_div(1.0f, 273.15f),
                                   exact_div(1.0f, sfctsno)))));
        Exact32 q2sati = exact_div(exact_mul(0.622f, e2sat),
                                exact_sub(sfcprs, e2sat));
        q2sati = exact_div(q2sati, exact_add(1.0f, q2sati));
        Exact32 snowc = snowc_a[idx];
        Exact32 dqi = exact_div(exact_mul(q2sati, 6174.0f),
                             exact_mul(sfctsno, sfctsno));
        if (t1 > 273.14f) {
            q2sat = exact_add(exact_mul(q2sat, exact_sub(1.0f, snowc)),
                              exact_mul(q2sati, snowc));
            dqsdt2 = exact_add(exact_mul(dqsdt2, exact_sub(1.0f, snowc)),
                               exact_mul(dqi, snowc));
        } else {
            q2sat = q2sati;
            dqsdt2 = dqi;
        }
        if (t1 > 273.0f && soft32_greater((snowc).bits,0u))
            dqsdt2 = exact_mul(dqsdt2, exact_sub(1.0f, snowc));
    }
    Exact32 prcp = exact_div(rainbl_a[idx], dt);
    Exact32 ffrozp;
    if (frpcpn) ffrozp = sr_a[idx];
    else ffrozp = (sfctmp <= 273.15f) ? 1.0f : 0.0f;
    const Exact32 tbot = 271.36f;
    Exact32 stc[4];
    for (int k = 0; k < 4; ++k) stc[k] = tslb_a[k * plane + idx];
    Exact32 ch = chs_a[idx];
    Exact32 ribb = rib_a[idx];
    Exact32 snomlt = 0.0f;
    Exact32 zsoil[4];
    for (int k = 0; k < 4; ++k)
        zsoil[k] = -exact_div(exact_mul(sithick, (Exact32)(k + 1)), (Exact32)4);
    Exact32 z0brd = 0.001f;
    bool snowng = false, frzgra = false;
    if (sneqv < 0.01f) {
        sneqv = 0.01f;
        snowh = 0.05f;
    }
    nsice_clamp_depth(sneqv, snowh, snowdepth_min, snowdepth_max);
    Exact32 sndens = exact_div(sneqv, snowh);
    if (sndens > 1.0f) atomic_exchange_explicit(reinterpret_cast<device atomic_int *>(fatal),1,memory_order_relaxed);
    Exact32 sncond = nglac_csnow(sndens);
    if (prcp > 0.0f) {
        if (ffrozp > 0.5f) snowng = true;
        else if (t1 <= 273.15f) frzgra = true;
    }
    if (snowng || frzgra) {
        Exact32 sn_new = exact_mul(exact_mul(prcp, dt), 0.001f);
        sneqv = exact_add(sneqv, sn_new);
        nglac_snow_new(sfctmp, sn_new, snowh, sndens);
        if (sncovr > 0.99f) {
            if (stc[0] < exact_sub(273.15f, 5.0f)) sndens = 0.2f;
            if (snowng && t1 < 273.0f && sfctmp < 273.0f) sndens = 0.2f;
        }
        sncond = nglac_csnow(sndens);
    }
    sncovr = 1.0f;
    emissi = 0.98f;
    Exact32 albedo = albedo_default;
    Exact32 df1 = 2.2f;
    Exact32 dsoil = -exact_mul(0.5f, zsoil[0]);
    Exact32 dtot = exact_add(snowh, dsoil);
    Exact32 frcsno = exact_div(snowh, dtot);
    Exact32 frcsoi = exact_div(dsoil, dtot);
    Exact32 df1a = exact_add(exact_mul(frcsno, sncond), exact_mul(frcsoi, df1));
    df1 = exact_add(exact_mul(df1a, sncovr),
                    exact_mul(df1, exact_sub(1.0f, sncovr)));
    Exact32 ssoil = exact_div(exact_mul(df1, exact_sub(t1, stc[0])), dtot);
    Exact32 burial = exact_sub(exact_mul(7.0f, z0brd), snowh);
    Exact32 z0eff = (burial <= 0.0007f) ? 0.001f : exact_div(burial, 7.0f);
    Exact32 z0 = exact_add(exact_mul(exact_sub(1.0f, sncovr), z0brd),
                        exact_mul(sncovr, z0eff));
    Exact32 fdown = exact_add(solnet, lwdn);
    Exact32 t2v = exact_mul(sfctmp, exact_add(1.0f, exact_mul(0.61f, q2)));
    Exact32 t24 = exact_mul(exact_mul(exact_mul(sfctmp, sfctmp), sfctmp), sfctmp);
    Exact32 rho = exact_div(sfcprs, exact_mul(287.04f, t2v));
    Exact32 rch = exact_mul(exact_mul(rho, 1004.6f), ch);
    Exact32 elcp1, lvs;
    if (t1 > 273.15f) {
        elcp1 = 2.4888e+3f;
        lvs = 2.501e+6f;
    } else {
        elcp1 = exact_div(exact_mul(2.4888e+3f, 2.83e+6f), 2.501e+6f);
        lvs = 2.83e+6f;
    }
    Exact32 flx2 = 0.0f;
    Exact32 delta = exact_mul(elcp1, dqsdt2);
    Exact32 rr = exact_add(exact_div(exact_mul(exact_mul(emissi, t24), 6.48e-8f),
                                  exact_mul(sfcprs, ch)), 1.0f);
    if (prcp > 0.0f) {
        if (!snowng) rr = exact_add(rr, exact_div(exact_mul(4.218e+3f, prcp), rch));
        else rr = exact_add(rr, exact_div(exact_mul(2.106e+3f, prcp), rch));
    }
    Exact32 fnet = exact_sub(exact_sub(fdown,
                    exact_mul(exact_mul(emissi, 5.67e-8f), t24)), ssoil);
    if (frzgra) {
        flx2 = exact_mul(-3.335e+5f, prcp);
        fnet = exact_sub(fnet, flx2);
    }
    Exact32 rad = exact_sub(exact_add(exact_div(fnet, rch), th2), sfctmp);
    Exact32 a = exact_mul(elcp1, exact_sub(q2sat, q2));
    Exact32 epsca = exact_div(exact_add(exact_mul(a, rr), exact_mul(rad, delta)),
                           exact_add(delta, rr));
    Exact32 etp = exact_div(exact_mul(epsca, rch), lvs);
    const Exact32 esdmin = 1.0e-6f;
    Exact32 esnow = 0.0f, esnow2 = 0.0f, etanrg;
    if (etp <= 0.0f) {
        if (ribb >= 0.1f && fdown > 150.0f) {
            etp = exact_div(exact_add(
                      exact_div(exact_mul(exact_min(exact_mul(etp,
                                                          exact_sub(1.0f, ribb)),
                                                0.0f), sncovr), 0.980f),
                      exact_mul(etp, exact_sub(0.980f, sncovr))), 0.980f);
        }
        Exact32 etp1 = exact_mul(etp, 0.001f);
        esnow2 = exact_mul(etp1, dt);
        etanrg = exact_mul(etp, exact_add(exact_mul(exact_sub(1.0f, sncovr),
                                                    2.501e+6f),
                                          exact_mul(sncovr, 2.83e+6f)));
    } else {
        esnow = exact_mul(etp, sncovr);
        Exact32 esnow1 = exact_mul(esnow, 0.001f);
        esnow2 = exact_mul(esnow1, dt);
        etanrg = exact_mul(esnow, 2.83e+6f);
    }
    Exact32 flx1 = 0.0f;
    if (snowng) flx1 = exact_mul(exact_mul(2.106e+3f, prcp), exact_sub(t1, sfctmp));
    else if (prcp > 0.0f)
        flx1 = exact_mul(exact_mul(4.218e+3f, prcp), exact_sub(t1, sfctmp));
    dsoil = -exact_mul(0.5f, zsoil[0]);
    dtot = exact_add(snowh, dsoil);
    Exact32 denom = exact_add(1.0f, exact_div(df1, exact_mul(exact_mul(dtot, rr), rch)));
    Exact32 t12a = exact_div(exact_sub(exact_sub(exact_add(exact_div(
                    exact_sub(exact_sub(exact_sub(fdown, flx1), flx2),
                              exact_mul(exact_mul(emissi, 5.67e-8f), t24)), rch),
                    th2), sfctmp), exact_div(etanrg, rch)), rr);
    Exact32 t12b = exact_div(exact_mul(df1, stc[0]),
                          exact_mul(exact_mul(dtot, rr), rch));
    Exact32 t12 = exact_div(exact_add(exact_add(sfctmp, t12a), t12b), denom);
    Exact32 esd = sneqv;
    Exact32 flx3 = 0.0f;
    if (t12 <= 273.15f) {
        t1 = t12;
        ssoil = exact_div(exact_mul(df1, exact_sub(t1, stc[0])), dtot);
        esd = exact_max(0.0f, exact_sub(esd, esnow2));
    } else {
        t1 = 273.15f;
        ssoil = exact_div(exact_mul(df1, exact_sub(t1, stc[0])), dtot);
        if (exact_sub(esd, esnow2) <= esdmin) {
            esd = 0.0f;
        } else {
            esd = exact_sub(esd, esnow2);
            Exact32 seh = exact_mul(rch, exact_sub(t1, th2));
            Exact32 t14 = exact_mul(exact_mul(t1, t1), exact_mul(t1, t1));
            flx3 = exact_sub(exact_sub(exact_sub(exact_sub(exact_sub(
                       exact_sub(fdown, flx1), flx2),
                       exact_mul(exact_mul(emissi, 5.67e-8f), t14)), ssoil),
                       seh), etanrg);
            if (flx3 <= 0.0f) flx3 = 0.0f;
            Exact32 ex = exact_div(exact_mul(flx3, 0.001f), 3.335e+5f);
            snomlt = exact_mul(ex, dt);
            if (exact_sub(esd, snomlt) >= esdmin) {
                esd = exact_sub(esd, snomlt);
            } else {
                ex = exact_div(esd, dt);
                flx3 = exact_mul(exact_mul(ex, 1000.0f), 3.335e+5f);
                snomlt = esd;
                esd = 0.0f;
            }
        }
    }
    Exact32 yy = exact_sub(stc[0], exact_div(exact_mul(exact_mul(
                  exact_mul(0.5f, ssoil), zsoil[0]), 1.0f), df1));
    {
        Exact32 rhsts[4], ai[4], bi[4], ci[4];
        nsice_hrtice(rhsts, stc, tbot, zsoil, yy, 1.0f, df1, ai, bi, ci);
        nice_hstep(stc, dt, rhsts, ai, bi, ci);
    }
    if (esd >= 0.01f) {
        nglac_snowpack(esd, dt, snowh, sndens, t1, yy);
    } else {
        esd = 0.01f;
        snowh = 0.05f;
        sncovr = 1.0f;
    }
    sneqv = esd;
    Exact32 eta_kin = etp;
    nsice_clamp_depth(sneqv, snowh, snowdepth_min, snowdepth_max);
    Exact32 q1 = exact_add(q2, exact_div(exact_mul(eta_kin, 1004.5f), rch));
    Exact32 sheat = exact_mul(exact_div(-exact_mul(exact_mul(ch, 1004.5f), sfcprs),
                                     exact_mul(287.04f, t2v)),
                           exact_sub(th2, t1));
    esnow = exact_mul(esnow, 2.83e+6f);
    etp = exact_mul(etp, exact_add(exact_mul(exact_sub(1.0f, sncovr), 2.501e+6f),
                                   exact_mul(sncovr, 2.83e+6f)));
    Exact32 eta = (etp > 0.0f) ? esnow : etp;
    ssoil = exact_mul(-1.0f, ssoil);
    Exact32 runoff1 = exact_div(snomlt, dt);
    albedo_a[idx] = albedo;
    emiss_a[idx] = emissi;
    tsk_a[idx] = t1;
    z0_a[idx] = z0brd;
    snowh_a[idx] = snowh;
    snowc_a[idx] = sncovr;
    snow_a[idx] = exact_mul(sneqv, 1000.0f);
    for (int k = 0; k < 4; ++k) tslb_a[k * plane + idx] = stc[k];
    znt_a[idx] = z0;
    lh_a[idx] = eta;
    hfx_a[idx] = sheat;
    qfx_a[idx] = eta_kin;
    potevp_a[idx] = exact_add(potevp_a[idx], exact_mul(etp,
                        exact_div(dt, exact_mul(2500000.0f, 1000.0f))));
    grdflx_a[idx] = ssoil;
    chs2_a[idx] = cqs2_a[idx];
    if (q1 > qsfc_a[idx]) cqs2_a[idx] = chs_a[idx];
    qsfc_a[idx] = exact_div(q1, exact_sub(1.0f, q1));
    if (ffrozp > 0.5f)
        acsnow_a[idx] = exact_add(acsnow_a[idx], exact_mul(prcp, dt));
    acsnom_a[idx] = exact_add(acsnom_a[idx], exact_mul(snomlt, 1000.0f));
    snopcx_a[idx] = exact_sub(snopcx_a[idx], exact_div(snomlt,
                        exact_div(dt, exact_mul(1000.0f, 3.50e+5f))));
    sfcrunoff_a[idx] = exact_add(sfcrunoff_a[idx],
                                 exact_mul(exact_mul(runoff1, dt), 1000.0f));
    Exact32 t14 = exact_mul(exact_mul(t1, t1), exact_mul(t1, t1));
    noahres_a[idx] = exact_sub(exact_sub(exact_sub(exact_sub(exact_sub(
                         exact_add(exact_sub(exact_add(solnet, lwdn), sheat),
                                   ssoil), eta),
                         exact_mul(exact_mul(emissi, 5.67051e-8f), t14)),
                         flx1), flx2), flx3);
}
kernel void wrf_exact_noah_seaice(constant WrfExactNoahSeaiceResources &r [[buffer(0)]], constant WrfExactNoahSeaiceParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(p.ny)*uint(p.nx)) return;
  noah_seaice_column_body(reinterpret_cast<device const Exact32*>(r.xice), reinterpret_cast<device const Exact32*>(r.psfc), reinterpret_cast<device const Exact32*>(r.sfcprs), reinterpret_cast<device const Exact32*>(r.sfctmp), reinterpret_cast<device const Exact32*>(r.qv1), reinterpret_cast<device const Exact32*>(r.qgh), reinterpret_cast<device const Exact32*>(r.glw), reinterpret_cast<device const Exact32*>(r.swdown), reinterpret_cast<device const Exact32*>(r.rainbl), reinterpret_cast<device const Exact32*>(r.sr), reinterpret_cast<device const Exact32*>(r.rib), reinterpret_cast<device const Exact32*>(r.chs), reinterpret_cast<device Exact32*>(r.cqs2), reinterpret_cast<device Exact32*>(r.chs2), reinterpret_cast<device Exact32*>(r.tsk), reinterpret_cast<device Exact32*>(r.emiss), reinterpret_cast<device Exact32*>(r.albedo), reinterpret_cast<device Exact32*>(r.z0), reinterpret_cast<device Exact32*>(r.snow), reinterpret_cast<device Exact32*>(r.snowc), reinterpret_cast<device Exact32*>(r.snowh), reinterpret_cast<device Exact32*>(r.znt), reinterpret_cast<device Exact32*>(r.lh), reinterpret_cast<device Exact32*>(r.hfx), reinterpret_cast<device Exact32*>(r.qfx), reinterpret_cast<device Exact32*>(r.potevp), reinterpret_cast<device Exact32*>(r.grdflx), reinterpret_cast<device Exact32*>(r.qsfc), reinterpret_cast<device Exact32*>(r.acsnow), reinterpret_cast<device Exact32*>(r.acsnom), reinterpret_cast<device Exact32*>(r.snopcx), reinterpret_cast<device Exact32*>(r.sfcrunoff), reinterpret_cast<device Exact32*>(r.noahres), reinterpret_cast<device Exact32*>(r.tslb), r.fatal, p.dt, p.xice_threshold, p.frpcpn, p.albedo_default, p.sithick, p.snowdepth_min, p.snowdepth_max, p.ny, p.nx, int(gid));
}
