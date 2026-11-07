#include <metal_stdlib>
#include "exact_gfk_math.metalh"
#include "thompson_aerosol_soft8.metalh"
#include "wrf_exact_noah_column_bindings.metalh"
#include "wrf_exact_noah_column_urban_bindings.metalh"
using namespace metal;
static inline bool operator<(Exact32 a, Exact32 b) { return soft32_less(a.bits,b.bits); }
static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_less_equal(a.bits,b.bits); }
static inline bool operator==(Exact32 a, Exact32 b) { return soft32_equal(a.bits,b.bits); }
static inline bool operator!=(Exact32 a, Exact32 b) { return !soft32_equal(a.bits,b.bits); }
static inline Soft8 widen(Exact32 x) { return Soft8::fromBits(soft64_from_f32_bits(x.bits)); }
static inline Exact32 narrow(Soft8 x) { return Exact32::from_bits(soft64_to_f32_bits(x.bits)); }
static inline int as_int(Exact32 x) { return soft64_to_i32_rz(soft64_from_f32_bits(x.bits)).value; }
  Exact32 noah_log10f(Exact32 x)
{
    unsigned int hx = x.bits;
    if ((hx & 0x7fffffffu) == 0u) return Exact32::from_bits(0xff800000u);
    if (hx >= 0x7f800000u) return (hx > 0x7f800000u || (hx >> 31))
                               ? Exact32::from_bits(0x7fc00000u) : x;
    int k = 0;
    if (hx < 0x00800000u) {
        x = exact_mul(x, 33554432.0f);
        hx = x.bits;
        k -= 25;
    }
    k += (int)(hx >> 23) - 127;
    int i = (int)(((unsigned int)k & 0x80000000u) >> 31);
    hx = (hx & 0x007fffffu) | ((unsigned int)(0x7f - i) << 23);
    Exact32 y = (Exact32)(k + i);
    Exact32 z = exact_add(exact_mul(y, Exact32::from_bits(0x355427dbu)),
                        exact_mul(Exact32::from_bits(0x3ede5bd9u),
                                  exact_gfk_log(Exact32::from_bits(hx))));
    return exact_add(z, exact_mul(y, Exact32::from_bits(0x3e9a2080u)));
}

  Exact32 noah_powi(Exact32 x, int m)
{
    unsigned int n = m < 0 ? (unsigned int)(-m) : (unsigned int)m;
    Exact32 y = (n % 2u) ? x : 1.0f;
    while (n >>= 1) {
        x = exact_mul(x, x);
        if (n % 2u) y = exact_mul(y, x);
    }
    return m < 0 ? exact_div(1.0f, y) : y;
}

 static Exact32 noah_frh2o(Exact32 tkelv, Exact32 smc, Exact32 sh2o,
                                  Exact32 smcmax, Exact32 bexp, Exact32 psis)
{
    const Exact32 ck = 8.0f, blim = 5.5f, error = 0.005f;
    const Exact32 hlice = 3.335e5f, gs = 9.81f, t0 = 273.15f;
    Exact32 bx = (bexp <= blim) ? bexp : blim;
    int nlog = 0, kcount = 0;
    if (tkelv > (t0 - 1.0e-3f)) return smc;
    Exact32 swl = smc - sh2o;
    if (swl > (smc - 0.02f)) swl = smc - 0.02f;
    if (swl < 0.0f) swl = 0.0f;
    while (nlog < 10 && kcount == 0) {
        nlog += 1;
        Exact32 df = exact_gfk_log(exact_div(psis * gs, hlice)
                       * exact_gfk_pow(1.0f + ck * swl, 2.0f)
                       * exact_gfk_pow(smcmax / (smc - swl), bx))
                  - exact_gfk_log(-(tkelv - t0) / tkelv);
        Exact32 denom = 2.0f * ck / (1.0f + ck * swl) + bx / (smc - swl);
        Exact32 swlk = swl - df / denom;
        if (swlk > (smc - 0.02f)) swlk = smc - 0.02f;
        if (swlk < 0.0f) swlk = 0.0f;
        Exact32 dswl = exact_abs(swlk - swl);
        swl = swlk;
        if (dswl <= error) kcount += 1;
    }
    Exact32 freew = smc - swl;
    if (kcount == 0) {
        Exact32 fk = exact_gfk_pow((hlice / (gs * (-psis)))
                       * ((tkelv - t0) / tkelv), -1.0f / bx) * smcmax;
        if (fk < 0.02f) fk = 0.02f;
        freew = exact_min(fk, smc);
    }
    return freew;
}

 static Exact32 noah_csnow(Exact32 dsnow)
{
    return 2.0f * 0.11631f * (0.328f * exact_gfk_pow(10.0f, 2.25f * dsnow));
}

 static void noah_snow_new(Exact32 temp, Exact32 newsn, thread Exact32& snowh,
                                     thread Exact32& sndens)
{
    Exact32 snowhc = snowh * 100.0f;
    Exact32 newsnc = newsn * 100.0f;
    Exact32 tempc = temp - 273.15f;
    Exact32 dsnew;
    if (tempc <= -15.0f) dsnew = 0.05f;
    else dsnew = 0.05f + 0.0017f * exact_gfk_pow(tempc + 15.0f, 1.5f);
    Exact32 hnewc = newsnc / dsnew;
    if (snowhc + hnewc < 1.0e-3f) sndens = exact_max(dsnew, sndens);
    else sndens = (snowhc * sndens + hnewc * dsnew) / (snowhc + hnewc);
    snowhc = snowhc + hnewc;
    snowh = snowhc * 0.01f;
}

 static Exact32 noah_snfrac(Exact32 sneqv, Exact32 snup, Exact32 salp)
{
    if (sneqv < snup) {
        Exact32 rsnow = sneqv / snup;
        return 1.0f - (exact_gfk_exp(-salp * rsnow) - rsnow * exact_gfk_exp(-salp));
    }
    return 1.0f;
}

 static void noah_alcalc(Exact32 alb, Exact32 snoalb, Exact32 embrd,
                                   Exact32 sncovr, Exact32 dt, bool snowng,
                                   thread Exact32& snotime1, Exact32 lvcoef,
                                   thread Exact32& albedo, thread Exact32& emissi)
{
    const Exact32 snacca = 0.94f, snaccb = 0.58f;
    albedo = alb + sncovr * (snoalb - alb);
    emissi = embrd + sncovr * (0.95f - embrd);
    Exact32 snoalb1 = snoalb + lvcoef * (0.85f - snoalb);
    Exact32 snoalb2 = snoalb1;
    if (snowng) {
        snotime1 = 0.0f;
    } else {
        snotime1 = snotime1 + dt;
        snoalb2 = snoalb1 * exact_gfk_pow(snacca,
                                 exact_gfk_pow(exact_div(snotime1, 86400.0f), snaccb));
    }
    snoalb2 = exact_max(snoalb2, alb);
    albedo = alb + sncovr * (snoalb2 - alb);
    if (albedo > snoalb2) albedo = snoalb2;
}

 static Exact32 noah_tdfcnd(Exact32 smc, Exact32 qz, Exact32 smcmax,
                                   Exact32 sh2o, Exact32 bexp, Exact32 psisat,
                                   int soiltyp, int opt_thcnd)
{
    if (opt_thcnd == 1 || (opt_thcnd == 2 && soiltyp != 4
                           && soiltyp != 3)) {
        Exact32 satratio = smc / smcmax;
        const Exact32 thkice = 2.2f, thkw = 0.57f, thko = 2.0f,
                   thkqtz = 7.7f;
        Exact32 thks = exact_gfk_pow(thkqtz, qz) * exact_gfk_pow(thko, 1.0f - qz);
        Exact32 xunfroz = sh2o / smc;
        Exact32 xu = xunfroz * smcmax;
        Exact32 thksat = exact_gfk_pow(thks, 1.0f - smcmax)
                      * exact_gfk_pow(thkice, smcmax - xu) * exact_gfk_pow(thkw, xu);
        Exact32 gammd = (1.0f - smcmax) * 2700.0f;
        Exact32 thkdry = (0.135f * gammd + 64.7f)
                      / (2700.0f - 0.947f * gammd);
        Exact32 akei = satratio;
        Exact32 akel;
        if (satratio > 0.1f) akel = noah_log10f(satratio) + 1.0f;
        else akel = 0.0f;
        Exact32 ake = ((smc - sh2o) * akei + sh2o * akel) / smc;
        return ake * (thksat - thkdry) + thkdry;
    }
    Exact32 psif = psisat * 100.0f * exact_gfk_pow(smcmax / smc, bexp);
    Exact32 pf = noah_log10f(exact_abs(psif));
    if (pf <= 5.1f) return 420.0f * exact_gfk_exp(-(pf + 2.7f));
    return 0.1744f;
}

 static Exact32 noah_snowz0(Exact32 sncovr, Exact32 z0brd, Exact32 snowh)
{
    const Exact32 z0s = 0.001f;
    Exact32 burial = 7.0f * z0brd - snowh;
    Exact32 z0eff;
    if (burial <= 0.0007f) z0eff = z0s;
    else z0eff = exact_div(burial, 7.0f);
    return (1.0f - sncovr) * z0brd + sncovr * z0eff;
}

 static void noah_penman(Exact32 sfctmp, Exact32 sfcprs, Exact32 ch,
                                   Exact32 t2v, Exact32 th2, Exact32 prcp,
                                   Exact32 fdown, Exact32 ssoil, Exact32 q2,
                                   Exact32 q2sat, Exact32 dqsdt2, bool snowng,
                                   bool frzgra, Exact32 emissi_in,
                                   Exact32 sncovr, thread Exact32& etp, thread Exact32& rch,
                                   thread Exact32& rr, thread Exact32& epsca, thread Exact32& t24,
                                   thread Exact32& flx2)
{
    Exact32 emissi = emissi_in;
    Exact32 elcp1 = (1.0f - sncovr) * 2.4888e+3f
                 + exact_div(sncovr * 2.4888e+3f * 2.83e+6f, 2.501e+6f);
    Exact32 lvs = (1.0f - sncovr) * 2.501e+6f + sncovr * 2.83e+6f;
    flx2 = 0.0f;
    Exact32 delta = elcp1 * dqsdt2;
    t24 = sfctmp * sfctmp * sfctmp * sfctmp;
    rr = emissi * t24 * 6.48e-8f / (sfcprs * ch) + 1.0f;
    Exact32 rho = sfcprs / (287.04f * t2v);
    rch = rho * 1004.6f * ch;
    if (!snowng) {
        if (prcp > 0.0f) rr = rr + 4.218e+3f * prcp / rch;
    } else {
        rr = rr + 2.106e+3f * prcp / rch;
    }
    Exact32 fnet = fdown - emissi * 5.67e-8f * t24 - ssoil;
    if (frzgra) {
        flx2 = -3.335e+5f * prcp;
        fnet = fnet - flx2;
    }
    Exact32 rad = fnet / rch + th2 - sfctmp;
    Exact32 a = elcp1 * (q2sat - q2);
    epsca = (a * rr + rad * delta) / (delta + rr);
    etp = epsca * rch / lvs;
}

 static void noah_canres(Exact32 solar, Exact32 ch, Exact32 sfctmp,
                                   Exact32 q2, thread const Exact32* sh2o,
                                   thread const Exact32* zsoil, Exact32 smcwlt,
                                   Exact32 smcref, Exact32 rsmin, int nroot,
                                   Exact32 q2sat, Exact32 dqsdt2, Exact32 topt,
                                   Exact32 rsmax, Exact32 rgl, Exact32 hs,
                                   Exact32 xlai, Exact32 sfcprs, Exact32 emissi,
                                   thread Exact32& rc, thread Exact32& pc)
{
    const Exact32 slv = 2.501000e6f;
    Exact32 ff = 0.55f * 2.0f * solar / (rgl * xlai);
    Exact32 rcs = (ff + rsmin / rsmax) / (1.0f + ff);
    rcs = exact_max(rcs, 0.0001f);
    Exact32 temperature_delta = exact_sub(topt, sfctmp);
    Exact32 rct = exact_sub(1.0f, exact_mul(0.0016f,
                                    exact_mul(temperature_delta, temperature_delta)));
    rct = exact_max(rct, 0.0001f);
    Exact32 rcq = 1.0f / (1.0f + hs * (q2sat - q2));
    rcq = exact_max(rcq, 0.01f);
    Exact32 rcsoil = 0.0f;
    Exact32 gx = (sh2o[0] - smcwlt) / (smcref - smcwlt);
    gx = exact_min(exact_max(gx, 0.0f), 1.0f);
    Exact32 part[4];
    part[0] = (zsoil[0] / zsoil[nroot - 1]) * gx;
    for (int k = 1; k < nroot; ++k) {
        gx = (sh2o[k] - smcwlt) / (smcref - smcwlt);
        gx = exact_min(exact_max(gx, 0.0f), 1.0f);
        part[k] = ((zsoil[k] - zsoil[k - 1]) / zsoil[nroot - 1]) * gx;
    }
    for (int k = 0; k < nroot; ++k) rcsoil = rcsoil + part[k];
    rcsoil = exact_max(rcsoil, 0.0001f);
    rc = rsmin / (xlai * rcs * rct * rcq * rcsoil);
    Exact32 rr2 = (exact_div(4.0f * emissi * 5.67e-8f * 287.04f, 1004.5f))
               * exact_gfk_pow(sfctmp, 4.0f) / (sfcprs * ch) + 1.0f;
    Exact32 delta = (slv / 1004.5f) * dqsdt2;
    pc = (rr2 + delta) / (rr2 * (1.0f + rc * ch) + delta);
}

 static Exact32 noah_devap(Exact32 etp1, Exact32 smc, Exact32 shdfac,
                                  Exact32 smcmax, Exact32 smcdry, Exact32 fxexp)
{
    Exact32 sratio = (smc - smcdry) / (smcmax - smcdry);
    Exact32 fx;
    if (sratio > 0.0f) {
        fx = exact_gfk_pow(sratio, fxexp);
        fx = exact_max(exact_min(fx, 1.0f), 0.0f);
    } else {
        fx = 0.0f;
    }
    return fx * (1.0f - shdfac) * etp1;
}

 static void noah_transp(thread Exact32* et, Exact32 etp1,
                                   thread const Exact32* sh2o, Exact32 cmc,
                                   Exact32 shdfac, Exact32 smcwlt,
                                   Exact32 cmcmax, Exact32 pc, Exact32 cfactr,
                                   Exact32 smcref, int nroot,
                                   thread const Exact32* rtdis)
{
    for (int k = 0; k < 4; ++k) et[k] = 0.0f;
    Exact32 etp1a;
    if (cmc != 0.0f)
        etp1a = shdfac * pc * etp1
                * (1.0f - exact_gfk_pow(cmc / cmcmax, cfactr));
    else
        etp1a = shdfac * pc * etp1;
    Exact32 gx[4];
    Exact32 sgx = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        gx[i] = (sh2o[i] - smcwlt) / (smcref - smcwlt);
        gx[i] = exact_max(exact_min(gx[i], 1.0f), 0.0f);
        sgx = sgx + gx[i];
    }
    sgx = sgx / (Exact32)nroot;
    Exact32 denom = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        Exact32 rtx = rtdis[i] + gx[i] - sgx;
        gx[i] = gx[i] * exact_max(rtx, 0.0f);
        denom = denom + gx[i];
    }
    if (denom <= 0.0f) denom = 1.0f;
    for (int i = 0; i < nroot; ++i) et[i] = etp1a * gx[i] / denom;
}

 static void noah_evapo(thread Exact32& eta1, thread const Exact32* smc, Exact32 cmc,
                                  Exact32 etp1, Exact32 dt, thread const Exact32* sh2o,
                                  Exact32 smcmax, Exact32 pc, Exact32 smcwlt,
                                  Exact32 smcref, Exact32 shdfac, Exact32 cmcmax,
                                  Exact32 smcdry, Exact32 cfactr, int nroot,
                                  thread const Exact32* rtdis, Exact32 fxexp,
                                  thread Exact32& edir, thread Exact32& ec, thread Exact32* et,
                                  thread Exact32& ett)
{
    edir = 0.0f;
    ec = 0.0f;
    ett = 0.0f;
    for (int k = 0; k < 4; ++k) et[k] = 0.0f;
    if (etp1 > 0.0f) {
        if (shdfac < 1.0f)
            edir = noah_devap(etp1, smc[0], shdfac, smcmax, smcdry,
                              fxexp);
        if (shdfac > 0.0f) {
            noah_transp(et, etp1, sh2o, cmc, shdfac, smcwlt, cmcmax,
                        pc, cfactr, smcref, nroot, rtdis);
            for (int k = 0; k < 4; ++k) ett = ett + et[k];
            if (cmc > 0.0f)
                ec = shdfac * exact_gfk_pow(cmc / cmcmax, cfactr) * etp1;
            else
                ec = 0.0f;
            Exact32 cmc2ms = cmc / dt;
            ec = exact_min(cmc2ms, ec);
        }
    }
    eta1 = edir + ett + ec;
}

 static Exact32 noah_fac2mit(Exact32 smcmax)
{
    Exact32 flimit = 0.90f;
    if (smcmax == 0.395f) flimit = 0.59f;
    else if (smcmax == 0.434f || smcmax == 0.404f) flimit = 0.85f;
    else if (smcmax == 0.465f || smcmax == 0.406f) flimit = 0.86f;
    else if (smcmax == 0.476f || smcmax == 0.439f) flimit = 0.74f;
    else if (smcmax == 0.200f || smcmax == 0.464f) flimit = 0.80f;
    return flimit;
}

 static void noah_wdfcnd(thread Exact32& wdf, thread Exact32& wcnd, Exact32 smc,
                                   Exact32 smcmax, Exact32 bexp, Exact32 dksat,
                                   Exact32 dwsat, Exact32 sicemax)
{
    Exact32 factr1 = 0.05f / smcmax;
    Exact32 factr2 = smc / smcmax;
    factr1 = exact_min(factr1, factr2);
    Exact32 expon = bexp + 2.0f;
    wdf = dwsat * exact_gfk_pow(factr2, expon);
    if (sicemax > 0.0f) {
        Exact32 vkwgt = 1.0f / (1.0f + exact_gfk_pow(500.0f * sicemax, 3.0f));
        wdf = vkwgt * wdf + (1.0f - vkwgt) * dwsat
              * exact_gfk_pow(factr1, expon);
    }
    expon = 2.0f * bexp + 3.0f;
    wcnd = dksat * exact_gfk_pow(factr2, expon);
}

 static void noah_srt(thread Exact32* rhstt, Exact32 edir, thread const Exact32* et,
                                thread const Exact32* sh2o, thread const Exact32* sh2oa,
                                Exact32 pcpdrp, thread const Exact32* zsoil,
                                Exact32 dwsat, Exact32 dksat, Exact32 smcmax,
                                Exact32 bexp, Exact32 dt, Exact32 smcwlt,
                                Exact32 slope, Exact32 kdt, Exact32 frzx,
                                thread const Exact32* sice, thread Exact32* ai, thread Exact32* bi,
                                thread Exact32* ci, thread Exact32& runoff1, thread Exact32& runoff2)
{
    const int cvfrz = 3;
    Exact32 dmax[4];
    Exact32 sicemax = 0.0f;
    for (int ks = 0; ks < 4; ++ks)
        if (sice[ks] > sicemax) sicemax = sice[ks];
    Exact32 pddum = pcpdrp;
    runoff1 = 0.0f;
    runoff2 = 0.0f;
    if (pcpdrp != 0.0f) {
        Exact32 dt1 = exact_div(dt, 86400.0f);
        Exact32 smcav = smcmax - smcwlt;
        dmax[0] = -zsoil[0] * smcav;
        Exact32 dice = -zsoil[0] * sice[0];
        dmax[0] = dmax[0] * (1.0f - (sh2oa[0] + sice[0] - smcwlt)
                             / smcav);
        Exact32 dd = dmax[0];
        for (int ks = 1; ks < 4; ++ks) {
            dice = dice + (zsoil[ks - 1] - zsoil[ks]) * sice[ks];
            dmax[ks] = (zsoil[ks - 1] - zsoil[ks]) * smcav;
            dmax[ks] = dmax[ks] * (1.0f - (sh2oa[ks] + sice[ks]
                                           - smcwlt) / smcav);
            dd = dd + dmax[ks];
        }
        Exact32 val = 1.0f - exact_gfk_exp(-kdt * dt1);
        Exact32 ddt = dd * val;
        Exact32 px = pcpdrp * dt;
        if (px < 0.0f) px = 0.0f;
        Exact32 infmax = (px * (ddt / (px + ddt))) / dt;
        Exact32 fcr = 1.0f;
        if (dice > 1.0e-2f) {
            Exact32 acrt = (Exact32)cvfrz * frzx / dice;
            Exact32 ssum = 1.0f;
            int ialp1 = cvfrz - 1;
            for (int j = 1; j <= ialp1; ++j) {
                int k = 1;
                for (int jj = j + 1; jj <= ialp1; ++jj) k = k * jj;
                ssum = ssum + noah_powi(acrt, cvfrz - j) / (Exact32)k;
            }
            fcr = 1.0f - exact_gfk_exp(-acrt) * ssum;
        }
        infmax = infmax * fcr;
        Exact32 wdf0, wcnd0;
        noah_wdfcnd(wdf0, wcnd0, sh2oa[0], smcmax, bexp, dksat, dwsat,
                    sicemax);
        infmax = exact_max(infmax, wcnd0);
        infmax = exact_min(infmax, px / dt);
        if (pcpdrp > infmax) {
            runoff1 = pcpdrp - infmax;
            pddum = infmax;
        }
    }
    Exact32 wdf, wcnd;
    noah_wdfcnd(wdf, wcnd, sh2oa[0], smcmax, bexp, dksat, dwsat,
                sicemax);
    Exact32 ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    bi[0] = wdf * ddz / (-zsoil[0]);
    ci[0] = -bi[0];
    Exact32 dsmdz = (sh2o[0] - sh2o[1]) / (-0.5f * zsoil[1]);
    rhstt[0] = (wdf * dsmdz + wcnd - pddum + edir + et[0]) / zsoil[0];
    Exact32 ddz2 = 0.0f;
    for (int k = 1; k < 4; ++k) {
        Exact32 denom2 = zsoil[k - 1] - zsoil[k];
        Exact32 wdf2, wcnd2, dsmdz2, slopx;
        if (k != 4 - 1) {
            slopx = 1.0f;
            noah_wdfcnd(wdf2, wcnd2, sh2oa[k], smcmax, bexp, dksat,
                        dwsat, sicemax);
            Exact32 denom = zsoil[k - 1] - zsoil[k + 1];
            dsmdz2 = (sh2o[k] - sh2o[k + 1]) / (denom * 0.5f);
            ddz2 = 2.0f / denom;
            ci[k] = -wdf2 * ddz2 / denom2;
        } else {
            slopx = slope;
            noah_wdfcnd(wdf2, wcnd2, sh2oa[4 - 1], smcmax, bexp,
                        dksat, dwsat, sicemax);
            dsmdz2 = 0.0f;
            ci[k] = 0.0f;
        }
        Exact32 numer = wdf2 * dsmdz2 + slopx * wcnd2 - wdf * dsmdz
                     - wcnd + et[k];
        rhstt[k] = numer / (-denom2);
        ai[k] = -wdf * ddz / denom2;
        bi[k] = -(ai[k] + ci[k]);
        if (k == 4 - 1) runoff2 = slopx * wcnd2;
        if (k != 4 - 1) {
            wdf = wdf2;
            wcnd = wcnd2;
            dsmdz = dsmdz2;
            ddz = ddz2;
        }
    }
}

 static void noah_rosr12(thread Exact32* p, thread const Exact32* a,
                                   thread const Exact32* b, thread const Exact32* c_in,
                                   thread const Exact32* d)
{
    Exact32 c_[4], delta[4];
    for (int k = 0; k < 4; ++k) c_[k] = c_in[k];
    c_[4 - 1] = 0.0f;
    p[0] = -c_[0] / b[0];
    delta[0] = d[0] / b[0];
    for (int k = 1; k < 4; ++k) {
        p[k] = -c_[k] * (1.0f / (b[k] + a[k] * p[k - 1]));
        delta[k] = (d[k] - a[k] * delta[k - 1])
                   * (1.0f / (b[k] + a[k] * p[k - 1]));
    }
    p[4 - 1] = delta[4 - 1];
    for (int k = 1; k < 4; ++k) {
        int kk = 4 - k - 1;
        p[kk] = p[kk] * p[kk + 1] + delta[kk];
    }
}

 static void noah_sstep(thread Exact32* sh2oout, thread const Exact32* sh2oin,
                                  thread Exact32& cmc, thread const Exact32* rhstt_in,
                                  Exact32 rhsct, Exact32 dt, Exact32 smcmax,
                                  Exact32 cmcmax, thread const Exact32* zsoil,
                                  thread Exact32* smc, thread const Exact32* sice,
                                  thread const Exact32* ai_in, thread const Exact32* bi_in,
                                  thread const Exact32* ci_in, thread Exact32& runoff3,
                                  bool update_cmc)
{
    Exact32 rhstt[4], ai[4], bi[4], ci[4], p[4];
    for (int k = 0; k < 4; ++k) {
        rhstt[k] = rhstt_in[k] * dt;
        ai[k] = ai_in[k] * dt;
        bi[k] = 1.0f + bi_in[k] * dt;
        ci[k] = ci_in[k] * dt;
    }
    noah_rosr12(p, ai, bi, ci, rhstt);
    Exact32 wplus = 0.0f;
    Exact32 ddz = -zsoil[0];
    for (int k = 0; k < 4; ++k) {
        if (k != 0) ddz = zsoil[k - 1] - zsoil[k];
        sh2oout[k] = sh2oin[k] + p[k] + wplus / ddz;
        Exact32 stot = sh2oout[k] + sice[k];
        if (stot > smcmax) {
            if (k == 0) ddz = -zsoil[0];
            else ddz = -zsoil[k] + zsoil[k - 1];
            wplus = (stot - smcmax) * ddz;
        } else {
            wplus = 0.0f;
        }
        smc[k] = exact_max(exact_min(stot, smcmax), 0.02f);
        sh2oout[k] = exact_max(smc[k] - sice[k], 0.0f);
    }
    runoff3 = wplus;
    if (update_cmc) {
        cmc = cmc + dt * rhsct;
        if (cmc < 1.0e-20f) cmc = 0.0f;
        cmc = exact_min(cmc, cmcmax);
    }
}

 static void noah_smflx(thread Exact32* smc, thread Exact32& cmc, Exact32 dt,
                                  Exact32 prcp1, thread const Exact32* zsoil,
                                  thread Exact32* sh2o, Exact32 slope, Exact32 kdt,
                                  Exact32 frzfact, Exact32 smcmax, Exact32 bexp,
                                  Exact32 smcwlt, Exact32 dksat, Exact32 dwsat,
                                  Exact32 shdfac, Exact32 cmcmax, Exact32 edir,
                                  Exact32 ec, thread const Exact32* et,
                                  thread Exact32& runoff1, thread Exact32& runoff2,
                                  thread Exact32& runoff3, thread Exact32& drip)
{
    Exact32 rhsct = shdfac * prcp1 - ec;
    drip = 0.0f;
    Exact32 trhsct = dt * rhsct;
    Exact32 excess = cmc + trhsct;
    if (excess > cmcmax) drip = excess - cmcmax;
    Exact32 pcpdrp = (1.0f - shdfac) * prcp1 + drip / dt;
    Exact32 sice[4];
    for (int i = 0; i < 4; ++i) sice[i] = smc[i] - sh2o[i];
    Exact32 fac2 = 0.0f;
    for (int i = 0; i < 4; ++i)
        fac2 = exact_max(fac2, sh2o[i] / smcmax);
    Exact32 flimit = noah_fac2mit(smcmax);
    Exact32 rhstt[4], ai[4], bi[4], ci[4];
    if ((pcpdrp * dt) > (0.0001f * 1000.0f * (-zsoil[0]) * smcmax)
        || (fac2 > flimit)) {
        Exact32 sh2ofg[4], sh2oa[4], sh2onew[4];
        Exact32 dummy = 0.0f;
        noah_srt(rhstt, edir, et, sh2o, sh2o, pcpdrp, zsoil, dwsat,
                 dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzfact,
                 sice, ai, bi, ci, runoff1, runoff2);
        noah_sstep(sh2ofg, sh2o, dummy, rhstt, rhsct, dt, smcmax,
                   cmcmax, zsoil, smc, sice, ai, bi, ci, runoff3,
                   false);
        for (int k = 0; k < 4; ++k)
            sh2oa[k] = (sh2o[k] + sh2ofg[k]) * 0.5f;
        noah_srt(rhstt, edir, et, sh2o, sh2oa, pcpdrp, zsoil, dwsat,
                 dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzfact,
                 sice, ai, bi, ci, runoff1, runoff2);
        noah_sstep(sh2onew, sh2o, cmc, rhstt, rhsct, dt, smcmax,
                   cmcmax, zsoil, smc, sice, ai, bi, ci, runoff3,
                   true);
        for (int k = 0; k < 4; ++k) sh2o[k] = sh2onew[k];
    } else {
        Exact32 sh2onew[4];
        noah_srt(rhstt, edir, et, sh2o, sh2o, pcpdrp, zsoil, dwsat,
                 dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzfact,
                 sice, ai, bi, ci, runoff1, runoff2);
        noah_sstep(sh2onew, sh2o, cmc, rhstt, rhsct, dt, smcmax,
                   cmcmax, zsoil, smc, sice, ai, bi, ci, runoff3,
                   true);
        for (int k = 0; k < 4; ++k) sh2o[k] = sh2onew[k];
    }
}

 static Exact32 noah_tbnd(Exact32 tu, Exact32 tb, thread const Exact32* zsoil,
                                 Exact32 zbot, int k)
{
    Exact32 zup, zb;
    if (k == 0) zup = 0.0f;
    else zup = zsoil[k - 1];
    if (k == 4 - 1) zb = 2.0f * zbot - zsoil[k];
    else zb = zsoil[k + 1];
    return tu + (tb - tu) * (zup - zsoil[k]) / (zup - zb);
}

 static Exact32 noah_tmpavg(Exact32 tup, Exact32 tm, Exact32 tdn,
                                   thread const Exact32* zsoil, int k)
{
    const Exact32 t0 = 2.7315e2f;
    Exact32 dz;
    if (k == 0) dz = -zsoil[0];
    else dz = zsoil[k - 1] - zsoil[k];
    Exact32 dzh = dz * 0.5f;
    if (tup < t0) {
        if (tm < t0) {
            if (tdn < t0) {
                return (tup + 2.0f * tm + tdn) / 4.0f;
            }
            Exact32 x0 = (t0 - tm) * dzh / (tdn - tm);
            return 0.5f * (tup * dzh + tm * (dzh + x0)
                           + t0 * (2.0f * dzh - x0)) / dz;
        }
        if (tdn < t0) {
            Exact32 xup = (t0 - tup) * dzh / (tm - tup);
            Exact32 xdn = dzh - (t0 - tm) * dzh / (tdn - tm);
            return 0.5f * (tup * xup + t0 * (2.0f * dz - xup - xdn)
                           + tdn * xdn) / dz;
        }
        Exact32 xup = (t0 - tup) * dzh / (tm - tup);
        return 0.5f * (tup * xup + t0 * (2.0f * dz - xup)) / dz;
    }
    if (tm < t0) {
        if (tdn < t0) {
            Exact32 xup = dzh - (t0 - tup) * dzh / (tm - tup);
            return 0.5f * (t0 * (dz - xup) + tm * (dzh + xup)
                           + tdn * dzh) / dz;
        }
        Exact32 xup = dzh - (t0 - tup) * dzh / (tm - tup);
        Exact32 xdn = (t0 - tm) * dzh / (tdn - tm);
        return 0.5f * (t0 * (2.0f * dz - xup - xdn)
                       + tm * (xup + xdn)) / dz;
    }
    if (tdn < t0) {
        Exact32 xdn = dzh - (t0 - tm) * dzh / (tdn - tm);
        return (t0 * (dz - xdn) + 0.5f * (t0 + tdn) * xdn) / dz;
    }
    return (tup + 2.0f * tm + tdn) / 4.0f;
}

 static Exact32 noah_snksrc(Exact32 qtot, Exact32 tavg, Exact32 smc,
                                   thread Exact32& sh2o, thread const Exact32* zsoil,
                                   Exact32 smcmax, Exact32 psisat, Exact32 bexp,
                                   Exact32 dt, int k)
{
    const Exact32 dh2o = 1.0000e3f, hlice = 3.3350e5f;
    Exact32 dz;
    if (k == 0) dz = -zsoil[0];
    else dz = zsoil[k - 1] - zsoil[k];
    Exact32 freew = noah_frh2o(tavg, smc, sh2o, smcmax, bexp, psisat);
    Exact32 xh2o = sh2o + qtot * dt / (dh2o * hlice * dz);
    if (xh2o < sh2o && xh2o < freew) {
        if (freew > sh2o) xh2o = sh2o;
        else xh2o = freew;
    }
    if (xh2o > sh2o && xh2o > freew) {
        if (freew < sh2o) xh2o = sh2o;
        else xh2o = freew;
    }
    if (xh2o < 0.0f) xh2o = 0.0f;
    if (xh2o > smc) xh2o = smc;
    Exact32 tsnsr = -dh2o * hlice * dz * (xh2o - sh2o) / dt;
    sh2o = xh2o;
    return tsnsr;
}

 static void noah_hrt(thread Exact32* rhsts, thread const Exact32* stc,
                                thread const Exact32* smc, Exact32 smcmax,
                                thread const Exact32* zsoil, Exact32 yy, Exact32 zz1,
                                Exact32 tbot, Exact32 zbot, Exact32 psisat,
                                thread Exact32* sh2o, Exact32 dt, Exact32 bexp,
                                int soiltyp, int opt_thcnd, Exact32 df1,
                                Exact32 quartz, Exact32 csoil, int vegtyp,
                                int isurban, thread Exact32* ai, thread Exact32* bi,
                                thread Exact32* ci)
{
    const Exact32 t0 = 273.15f, cair = 1004.0f, cice = 2.106e6f,
               ch2o = 4.2e6f;
    Exact32 csoil_loc;
    if (vegtyp == isurban) csoil_loc = 3.0e6f;
    else csoil_loc = csoil;
    Exact32 hcpct = sh2o[0] * ch2o + (1.0f - smcmax) * csoil_loc
                 + (smcmax - smc[0]) * cair
                 + (smc[0] - sh2o[0]) * cice;
    Exact32 ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    ci[0] = (df1 * ddz) / (zsoil[0] * hcpct);
    bi[0] = -ci[0] + df1 / (0.5f * zsoil[0] * zsoil[0] * hcpct * zz1);
    Exact32 dtsdz = (stc[0] - stc[1]) / (-0.5f * zsoil[1]);
    Exact32 ssoil = df1 * (stc[0] - yy) / (0.5f * zsoil[0] * zz1);
    Exact32 denom = zsoil[0] * hcpct;
    rhsts[0] = (df1 * dtsdz - ssoil) / denom;
    Exact32 qtot = -1.0f * rhsts[0] * denom;
    Exact32 sice = smc[0] - sh2o[0];
    Exact32 tsurf = (yy + (zz1 - 1.0f) * stc[0]) / zz1;
    Exact32 tbk = noah_tbnd(stc[0], stc[1], zsoil, zbot, 0);
    if (sice > 0.0f || stc[0] < t0 || tsurf < t0 || tbk < t0) {
        Exact32 tavg = noah_tmpavg(tsurf, stc[0], tbk, zsoil, 0);
        Exact32 tsnsr = noah_snksrc(qtot, tavg, smc[0], sh2o[0], zsoil,
                                 smcmax, psisat, bexp, dt, 0);
        rhsts[0] = rhsts[0] - tsnsr / denom;
    }
    Exact32 ddz2 = 0.0f;
    Exact32 df1k = df1;
    Exact32 dtsdz2;
    for (int k = 1; k < 4; ++k) {
        hcpct = sh2o[k] * ch2o + (1.0f - smcmax) * csoil_loc
                + (smcmax - smc[k]) * cair
                + (smc[k] - sh2o[k]) * cice;
        Exact32 df1n, tbk1;
        if (k != 4 - 1) {
            df1n = noah_tdfcnd(smc[k], quartz, smcmax, sh2o[k], bexp,
                               psisat, soiltyp, opt_thcnd);
            if (vegtyp == isurban) df1n = 3.24f;
            denom = 0.5f * (zsoil[k - 1] - zsoil[k + 1]);
            dtsdz2 = (stc[k] - stc[k + 1]) / denom;
            ddz2 = 2.0f / (zsoil[k - 1] - zsoil[k + 1]);
            ci[k] = -df1n * ddz2 / ((zsoil[k - 1] - zsoil[k]) * hcpct);
            tbk1 = noah_tbnd(stc[k], stc[k + 1], zsoil, zbot, k);
        } else {
            df1n = noah_tdfcnd(smc[k], quartz, smcmax, sh2o[k], bexp,
                               psisat, soiltyp, opt_thcnd);
            if (vegtyp == isurban) df1n = 3.24f;
            denom = 0.5f * (zsoil[k - 1] + zsoil[k]) - zbot;
            dtsdz2 = (stc[k] - tbot) / denom;
            ci[k] = 0.0f;
            tbk1 = noah_tbnd(stc[k], tbot, zsoil, zbot, k);
        }
        denom = (zsoil[k] - zsoil[k - 1]) * hcpct;
        rhsts[k] = (df1n * dtsdz2 - df1k * dtsdz) / denom;
        qtot = -1.0f * denom * rhsts[k];
        sice = smc[k] - sh2o[k];
        Exact32 tavg = noah_tmpavg(tbk, stc[k], tbk1, zsoil, k);
        if (sice > 0.0f || stc[k] < t0 || tbk < t0 || tbk1 < t0) {
            Exact32 tsnsr = noah_snksrc(qtot, tavg, smc[k], sh2o[k],
                                     zsoil, smcmax, psisat, bexp, dt,
                                     k);
            rhsts[k] = rhsts[k] - tsnsr / denom;
        }
        ai[k] = -df1k * ddz / ((zsoil[k - 1] - zsoil[k]) * hcpct);
        bi[k] = -(ai[k] + ci[k]);
        tbk = tbk1;
        df1k = df1n;
        dtsdz = dtsdz2;
        ddz = ddz2;
    }
}

 static void noah_hstep(thread Exact32* stcout, thread const Exact32* stcin,
                                  thread const Exact32* rhsts_in, Exact32 dt,
                                  thread const Exact32* ai_in, thread const Exact32* bi_in,
                                  thread const Exact32* ci_in)
{
    Exact32 rhsts[4], ai[4], bi[4], ci[4], p[4];
    for (int k = 0; k < 4; ++k) {
        rhsts[k] = rhsts_in[k] * dt;
        ai[k] = ai_in[k] * dt;
        bi[k] = 1.0f + bi_in[k] * dt;
        ci[k] = ci_in[k] * dt;
    }
    noah_rosr12(p, ai, bi, ci, rhsts);
    for (int k = 0; k < 4; ++k) stcout[k] = stcin[k] + p[k];
}

 static void noah_shflx(thread Exact32* stc, thread const Exact32* smc,
                                  Exact32 smcmax, thread Exact32& t1, Exact32 dt,
                                  Exact32 yy, Exact32 zz1, thread const Exact32* zsoil,
                                  Exact32 tbot, Exact32 zbot, Exact32 psisat,
                                  thread Exact32* sh2o, Exact32 bexp, Exact32 df1,
                                  Exact32 quartz, Exact32 csoil, int vegtyp,
                                  int isurban, int soiltyp,
                                  int opt_thcnd, thread Exact32& ssoil)
{
    Exact32 rhsts[4], ai[4], bi[4], ci[4], stcf[4];
    noah_hrt(rhsts, stc, smc, smcmax, zsoil, yy, zz1, tbot, zbot,
             psisat, sh2o, dt, bexp, soiltyp, opt_thcnd, df1, quartz,
             csoil, vegtyp, isurban, ai, bi, ci);
    noah_hstep(stcf, stc, rhsts, dt, ai, bi, ci);
    for (int i = 0; i < 4; ++i) stc[i] = stcf[i];
    t1 = (yy + (zz1 - 1.0f) * stc[0]) / zz1;
    ssoil = df1 * (stc[0] - t1) / (0.5f * zsoil[0]);
}

 static void noah_snowpack(Exact32 esd, Exact32 dtsec, thread Exact32& snowh,
                                     thread Exact32& sndens, Exact32 tsnow,
                                     Exact32 tsoil)
{
    const Exact32 c1k = 0.01f, c2k = 21.0f;
    Exact32 snowhc = snowh * 100.0f;
    Exact32 esdc = esd * 100.0f;
    Exact32 dthr = exact_div(dtsec, 3600.0f);
    Exact32 tsnowc = tsnow - 273.15f;
    Exact32 tsoilc = tsoil - 273.15f;
    Exact32 tavgc = 0.5f * (tsnowc + tsoilc);
    Exact32 esdcx;
    if (esdc > 1.0e-2f) esdcx = esdc;
    else esdcx = 1.0e-2f;
    Exact32 bfac = dthr * c1k * exact_gfk_exp(0.08f * tavgc - c2k * sndens);
    const int ipol = 4;
    Exact32 pexp = 0.0f;
    for (int j = ipol; j >= 1; --j)
        pexp = exact_div((1.0f + pexp) * bfac * esdcx, (Exact32)(j + 1));
    pexp = pexp + 1.0f;
    Exact32 dsx = sndens * pexp;
    if (dsx > 0.40f) dsx = 0.40f;
    if (dsx < 0.05f) dsx = 0.05f;
    sndens = dsx;
    if (tsnowc >= 0.0f) {
        Exact32 dw = exact_div(0.13f * dthr, 24.0f);
        sndens = sndens * (1.0f - dw) + dw;
        if (sndens >= 0.40f) sndens = 0.40f;
    }
    snowhc = esdc / sndens;
    snowh = snowhc * 0.01f;
}

static inline Soft8 noah_f2d(Exact32 x) { return Soft8::fromBits(soft64_from_f32_bits(x.bits)); }

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

 static void nglac_alcalc(Exact32 alb, Exact32 snoalb, Exact32 embrd,
                                    Exact32 dt, bool snowng, Exact32 lvcoef,
                                    thread Exact32& snotime1, thread Exact32& albedo,
                                    thread Exact32& emissi)
{
    const Exact32 snacca = 0.94f, snaccb = 0.58f;
    emissi = exact_add(embrd, exact_sub(0.95f, embrd));
    Exact32 snoalb1 = exact_add(snoalb, exact_mul(lvcoef,
                                               exact_sub(0.85f, snoalb)));
    Exact32 snoalb2 = snoalb1;
    if (snowng) {
        snotime1 = 0.0f;
    } else {
        snotime1 = exact_add(snotime1, dt);
        snoalb2 = exact_mul(snoalb1, exact_gfk_pow(snacca,
                      exact_gfk_pow(exact_div(snotime1, 86400.0f), snaccb)));
    }
    snoalb2 = exact_max(snoalb2, alb);
    albedo = exact_add(alb, exact_sub(snoalb2, alb));
    if (albedo > snoalb2) albedo = snoalb2;
}

 static void nglac_hrtice(thread Exact32* rhsts, thread const Exact32* stc, Exact32 tbot,
                                    thread const Exact32* zsoil, Exact32 yy, Exact32 zz1,
                                    Exact32 df1, thread Exact32* ai, thread Exact32* bi, thread Exact32* ci)
{
    Exact32 hcpct = exact_mul(1.0e6f, exact_sub(0.8194f,
                     exact_mul(exact_mul(0.1309f, 0.5f), zsoil[0])));
    Exact32 df1k = df1;
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
    df1k = df1;
    Exact32 df1n = df1;
    for (int k = 1; k < 4; ++k) {
        Exact32 zmd = exact_mul(0.5f, exact_add(zsoil[k], zsoil[k - 1]));
        hcpct = exact_mul(1.0e6f, exact_sub(0.8194f, exact_mul(0.1309f, zmd)));
        df1n = exact_sub(0.32333f, exact_mul(0.10073f, zmd));
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
                                         -25.0f));
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

 static void nglac_shflx(thread Exact32* stc, Exact32 dt, Exact32 yy, Exact32 zz1,
                                   thread const Exact32* zsoil, Exact32 tbot, Exact32 df1)
{
    Exact32 rhsts[4], ai[4], bi[4], ci[4];
    nglac_hrtice(rhsts, stc, tbot, zsoil, yy, zz1, df1, ai, bi, ci);
    nice_hstep(stc, dt, rhsts, ai, bi, ci);
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

  static void noah_glacial_column(
                 ulong idx, ulong plane,
                 device const Exact32*  psfc_a,
                 device const Exact32*  sfcprs_a,
                 device const Exact32*  sfctmp_a,
                 device const Exact32*  qv1_a,
                 device const Exact32*  qgh_a,
                 device const Exact32*  glw_a,
                 device const Exact32*  swdown_a,
                 device const Exact32*  rainbl_a,
                 device const Exact32*  sr_a,
                 device const Exact32*  chs_a,
                 device const Exact32*  cqs2_a,
                 device Exact32*  chs2_a,
                 device const Exact32*  rib_a,
                 device const Exact32*  tmn_a,
                 device const Exact32*  snoalb_a,
                 device const Exact32*  embck_a,
                 device Exact32*  tsk_a,
                 device Exact32*  hfx_a,
                 device Exact32*  qfx_a,
                 device Exact32*  lh_a,
                 device Exact32*  grdflx_a,
                 device Exact32*  qsfc_a,
                 device Exact32*  canwat_a,
                 device Exact32*  snow_a,
                 device Exact32*  snowc_a,
                 device Exact32*  snowh_a,
                 device Exact32*  albedo_a,
                 device Exact32*  albbck_a,
                 device Exact32*  emiss_a,
                 device Exact32*  znt_a,
                 device Exact32*  z0_a,
                 device Exact32*  snotime_a,
                 device Exact32*  lai_a,
                 device Exact32*  smstav_a,
                 device Exact32*  smstot_a,
                 device Exact32*  sfcrunoff_a,
                 device Exact32*  udrunoff_a,
                 device Exact32*  acsnow_a,
                 device Exact32*  acsnom_a,
                 device Exact32*  snopcx_a,
                 device Exact32*  potevp_a,
                 device Exact32*  noahres_a,
                 device Exact32*  reslin_a,
                 device Exact32*  smois_a,
                 device Exact32*  tslb_a,
                 device Exact32*  sh2o_a,
                 device Exact32*  smcrel_a,
                 device int*  ebal_a,
                 device const Exact32*  genp,
                 device const Exact32*  dzs,
                 Exact32 dt, int frpcpn,
                 int bep_opt, device Exact32*  rural_bep)
{
    const Exact32 a2 = 17.67f, a3 = 273.15f, a4 = 29.65f;
    const Exact32 a23m4 = exact_mul(a2, exact_sub(a3, a4));
    Exact32 psfc = psfc_a[idx];
    Exact32 sfcprs = sfcprs_a[idx];
    Exact32 q2k = exact_div(qv1_a[idx], exact_add(1.0f, qv1_a[idx]));
    Exact32 q2sat = exact_div(qgh_a[idx], exact_add(1.0f, qgh_a[idx]));
    Exact32 sfctmp = sfctmp_a[idx];
    Exact32 apes = exact_gfk_pow(exact_div(1.0e5f, psfc), Exact32::from_bits(0x3e924925u));
    Exact32 apelm = exact_gfk_pow(exact_div(1.0e5f, sfcprs), Exact32::from_bits(0x3e924925u));
    Exact32 sfcth2 = exact_mul(sfctmp, apelm);
    Exact32 th2 = exact_div(sfcth2, apes);
    Exact32 emissi = emiss_a[idx];
    Exact32 lwdn = exact_mul(glw_a[idx], emissi);
    Exact32 soldn = swdown_a[idx];
    Exact32 solnet = exact_mul(soldn, exact_sub(1.0f, albedo_a[idx]));
    Exact32 prcp = exact_div(rainbl_a[idx], dt);
    Exact32 t1 = tsk_a[idx];
    Exact32 chk = chs_a[idx];
    Exact32 sneqv = exact_mul(snow_a[idx], 0.001f);
    Exact32 snowhk = snowh_a[idx];
    Exact32 sncovr = snowc_a[idx];
    Exact32 ffrozp;
    if (frpcpn) ffrozp = sr_a[idx];
    else ffrozp = (sfctmp <= 273.15f) ? 1.0f : 0.0f;
    Exact32 sfcma4 = exact_sub(sfctmp, a4);
    Exact32 dqsdt2 = exact_div(exact_mul(q2sat, a23m4), exact_mul(sfcma4, sfcma4));
    if (noah_f2d(snow_a[idx]) > Soft8(0)) {
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
        if (t1 > 273.0f && noah_f2d(snowc) > Soft8(0) && swdown_a[idx] > 10.0f)
            dqsdt2 = exact_mul(dqsdt2, exact_sub(1.0f, snowc));
    }
    Exact32 tbot = tmn_a[idx];
    Exact32 snoalb1 = snoalb_a[idx];
    Exact32 cmc = exact_div(canwat_a[idx], 1000.0f);
    Exact32 alb = albbck_a[idx];
    Exact32 z0brd = z0_a[idx];
    Exact32 embrd = embck_a[idx];
    Exact32 snotime1 = snotime_a[idx];
    Exact32 ribb = rib_a[idx];
    Exact32 stc[4];
    for (int k = 0; k < 4; ++k) stc[k] = tslb_a[k * plane + idx];
    if ((sneqv != 0.0f && snowhk == 0.0f) || (snowhk <= sneqv))
        snowhk = exact_mul(5.0f, sneqv);
    Exact32 zsoil[4];
    zsoil[0] = -dzs[0];
    for (int k = 1; k < 4; ++k) zsoil[k] = exact_add(-dzs[k], zsoil[k - 1]);
    if (sneqv < 0.10f) {
        sneqv = 0.10f;
        snowhk = 0.50f;
    }
    Exact32 sndens = exact_div(sneqv, snowhk);
    Exact32 sncond = nglac_csnow(sndens);
    bool snowng = false, frzgra = false;
    if (prcp > 0.0f) {
        if (ffrozp > 0.5f) snowng = true;
        else if (t1 <= 273.15f) frzgra = true;
    }
    Exact32 prcpf;
    if (snowng || frzgra) {
        Exact32 sn_new = exact_mul(exact_mul(prcp, dt), 0.001f);
        sneqv = exact_add(sneqv, sn_new);
        prcpf = 0.0f;
        nglac_snow_new(sfctmp, sn_new, snowhk, sndens);
        if (sncovr > 0.99f) {
            if (stc[0] < exact_sub(273.15f, 5.0f)) sndens = 0.2f;
            if (snowng && t1 < 273.0f && sfctmp < 273.0f) sndens = 0.2f;
        }
        sncond = nglac_csnow(sndens);
    } else {
        prcpf = prcp;
    }
    (void)prcpf;
    sncovr = 1.0f;
    Exact32 albedo;
    nglac_alcalc(alb, snoalb1, embrd, dt, snowng, genp[12],
                 snotime1, albedo, emissi);
    Exact32 df1 = sncond;
    Exact32 dsoil = -exact_mul(0.5f, zsoil[0]);
    Exact32 dtot = exact_add(snowhk, dsoil);
    Exact32 frcsno = exact_div(snowhk, dtot);
    Exact32 frcsoi = exact_div(dsoil, dtot);
    df1 = exact_add(exact_mul(frcsno, sncond), exact_mul(frcsoi, df1));
    if (dtot > exact_mul(2.0f, dsoil)) dtot = exact_mul(2.0f, dsoil);
    Exact32 ssoil = exact_div(exact_mul(df1, exact_sub(t1, stc[0])), dtot);
    Exact32 burial = exact_sub(exact_mul(7.0f, z0brd), snowhk);
    Exact32 z0k = (burial <= 0.0007f) ? 0.001f : exact_div(burial, 7.0f);
    Exact32 fdown = exact_add(solnet, lwdn);
    Exact32 t2v = exact_mul(sfctmp, exact_add(1.0f, exact_mul(0.61f, q2k)));
    Exact32 rho = exact_div(sfcprs, exact_mul(287.04f, t2v));
    Exact32 rch = exact_mul(exact_mul(rho, 1004.6f), chk);
    Exact32 t24 = exact_mul(exact_mul(exact_mul(sfctmp, sfctmp), sfctmp), sfctmp);
    Exact32 elcp1, lvs;
    if (t1 > 273.15f) {
        elcp1 = 2.4888e+3f;
        lvs = 2.501e+6f;
    } else {
        elcp1 = exact_div(exact_mul(2.4888e+3f, 2.83e+6f), 2.501e+6f);
        lvs = 2.83e+6f;
    }
    Exact32 delta = exact_mul(elcp1, dqsdt2);
    Exact32 a = exact_mul(elcp1, exact_sub(q2sat, q2k));
    Exact32 rr = exact_add(exact_div(exact_mul(exact_mul(emissi, t24), 6.48e-8f),
                                  exact_mul(sfcprs, chk)), 1.0f);
    if (!snowng) {
        if (prcp > 0.0f) rr = exact_add(rr, exact_div(exact_mul(4.218e+3f, prcp), rch));
    } else {
        rr = exact_add(rr, exact_div(exact_mul(2.106e+3f, prcp), rch));
    }
    Exact32 flx2 = frzgra ? exact_mul(-3.335e+5f, prcp) : 0.0f;
    Exact32 fnet = exact_sub(exact_sub(exact_sub(fdown,
                    exact_mul(exact_mul(emissi, 5.67e-8f), t24)), ssoil), flx2);
    Exact32 rad = exact_sub(exact_add(exact_div(fnet, rch), th2), sfctmp);
    Exact32 epsca = exact_div(exact_add(exact_mul(a, rr), exact_mul(rad, delta)),
                           exact_add(delta, rr));
    Exact32 etp = exact_div(exact_mul(epsca, rch), lvs);
    const Exact32 esdmin = 1.0e-6f;
    Exact32 snomlt = 0.0f, esnow = 0.0f, esnow2 = 0.0f, etanrg;
    if (etp <= 0.0f) {
        if (ribb >= 0.1f && fdown > 150.0f) {
            etp = exact_div(exact_add(
                      exact_div(exact_min(exact_mul(etp, exact_sub(1.0f, ribb)), 0.0f),
                                0.980f),
                      exact_mul(etp, exact_sub(0.980f, 1.0f))), 0.980f);
        }
        Exact32 etp1 = exact_mul(etp, 0.001f);
        esnow2 = exact_mul(etp1, dt);
        etanrg = exact_mul(etp, 2.83e+6f);
    } else {
        esnow = etp;
        Exact32 esnow1 = exact_mul(esnow, 0.001f);
        esnow2 = exact_mul(esnow1, dt);
        etanrg = exact_mul(esnow, 2.83e+6f);
    }
    Exact32 flx1 = 0.0f;
    if (snowng) flx1 = exact_mul(exact_mul(2.106e+3f, prcp), exact_sub(t1, sfctmp));
    else if (prcp > 0.0f)
        flx1 = exact_mul(exact_mul(4.218e+3f, prcp), exact_sub(t1, sfctmp));
    dsoil = -exact_mul(0.5f, zsoil[0]);
    dtot = exact_add(snowhk, dsoil);
    Exact32 denom = exact_add(1.0f, exact_div(df1, exact_mul(exact_mul(dtot, rr), rch)));
    Exact32 t12a = exact_div(exact_sub(exact_sub(exact_add(exact_div(
                    exact_sub(exact_sub(exact_sub(fdown, flx1), flx2),
                              exact_mul(exact_mul(emissi, 5.67e-8f), t24)), rch),
                    th2), sfctmp), exact_div(etanrg, rch)), rr);
    Exact32 t12b = exact_div(exact_mul(df1, stc[0]),
                          exact_mul(exact_mul(dtot, rr), rch));
    Exact32 t12 = exact_div(exact_add(exact_add(sfctmp, t12a), t12b), denom);
    Exact32 stc1_old = stc[0];
    Exact32 flx3 = 0.0f;
    int ebal_case;
    Soft8 reslin;
    if (t12 <= 273.15f) {
        ebal_case = 1;
        t1 = t12;
        ssoil = exact_div(exact_mul(df1, exact_sub(t1, stc[0])), dtot);
        sneqv = exact_max(0.0f, exact_sub(sneqv, esnow2));
        reslin = (widen(fdown) - widen(flx1) - widen(flx2)
                  - widen(emissi) * widen(5.67e-8f) * widen(t24)
                  + widen(rch) * (widen(th2) - widen(sfctmp))
                  - widen(etanrg)
                  - widen(rch) * widen(rr) * (widen(t12) - widen(sfctmp))
                  - widen(df1) * (widen(t12) - widen(stc1_old))
                    / widen(dtot));
    } else {
        bool clipped = false;
        t1 = 273.15f;
        if (dtot > exact_mul(2.0f, dsoil)) dtot = exact_mul(2.0f, dsoil);
        ssoil = exact_div(exact_mul(df1, exact_sub(t1, stc[0])), dtot);
        Exact32 seh = exact_mul(rch, exact_sub(t1, th2));
        Exact32 t14 = exact_mul(exact_mul(t1, t1), exact_mul(t1, t1));
        if (exact_sub(sneqv, esnow2) <= esdmin) {
            sneqv = 0.0f;
            snomlt = 0.0f;
            flx3 = 0.0f;
            clipped = true;
        } else {
            sneqv = exact_sub(sneqv, esnow2);
            flx3 = exact_sub(exact_sub(exact_sub(exact_sub(exact_sub(
                       exact_sub(fdown, flx1), flx2),
                       exact_mul(exact_mul(emissi, 5.67e-8f), t14)), ssoil),
                       seh), etanrg);
            if (flx3 <= 0.0f) {
                flx3 = 0.0f;
                clipped = true;
            }
            Exact32 ex = exact_div(exact_mul(flx3, 0.001f), 3.335e+5f);
            snomlt = exact_mul(ex, dt);
            if (exact_sub(sneqv, snomlt) >= esdmin) {
                sneqv = exact_sub(sneqv, snomlt);
            } else {
                ex = exact_div(sneqv, dt);
                flx3 = exact_mul(exact_mul(ex, 1000.0f), 3.335e+5f);
                snomlt = sneqv;
                sneqv = 0.0f;
                clipped = true;
            }
        }
        ebal_case = clipped ? 3 : 2;
        reslin = (widen(fdown) - widen(flx1) - widen(flx2)
                  - widen(emissi) * widen(5.67e-8f) * widen(t14)
                  - widen(ssoil) - widen(seh)
                  - widen(etanrg) - widen(flx3));
    }
    Exact32 yy = exact_sub(stc[0], exact_div(exact_mul(exact_mul(
                  exact_mul(0.5f, ssoil), zsoil[0]), 1.0f), df1));
    nglac_shflx(stc, dt, yy, 1.0f, zsoil, tbot, df1);
    if (sneqv >= 0.10f) {
        nglac_snowpack(sneqv, dt, snowhk, sndens, t1, yy);
    } else {
        sneqv = 0.10f;
        snowhk = 0.50f;
    }
    Exact32 eta_kin = etp;
    Exact32 q1 = exact_add(q2k, exact_div(exact_mul(eta_kin, 1004.5f), rch));
    Exact32 sheat = exact_mul(exact_div(-exact_mul(exact_mul(chk, 1004.5f), sfcprs),
                                     exact_mul(287.04f, t2v)),
                           exact_sub(th2, t1));
    Exact32 esnow_e = exact_mul(esnow, 2.83e+6f);
    Exact32 etp_e = exact_mul(etp, 2.83e+6f);
    Exact32 eta_e = (etp_e > 0.0f) ? esnow_e : etp_e;
    Exact32 ssoil_out = exact_mul(-1.0f, ssoil);
    Exact32 runoff1 = exact_div(snomlt, dt);
    lai_a[idx] = 0.01f;
    canwat_a[idx] = exact_mul(cmc, 1000.0f);
    snow_a[idx] = exact_mul(sneqv, 1000.0f);
    snowh_a[idx] = snowhk;
    albedo_a[idx] = albedo;
    albbck_a[idx] = alb;
    z0_a[idx] = z0brd;
    emiss_a[idx] = emissi;
    znt_a[idx] = z0k;
    tsk_a[idx] = t1;
    if (bep_opt >= 2 && rural_bep != nullptr) rural_bep[idx] = t1;
    hfx_a[idx] = sheat;
    potevp_a[idx] = exact_add(potevp_a[idx], exact_mul(etp_e,
                        exact_div(dt, exact_mul(2500000.0f, 1000.0f))));
    qfx_a[idx] = eta_kin;
    lh_a[idx] = eta_e;
    grdflx_a[idx] = ssoil_out;
    snowc_a[idx] = sncovr;
    chs2_a[idx] = cqs2_a[idx];
    snotime_a[idx] = snotime1;
    qsfc_a[idx] = exact_div(q1, exact_sub(1.0f, q1));
    for (int k = 0; k < 4; ++k) {
        smois_a[k * plane + idx] = 1.0f;
        tslb_a[k * plane + idx] = stc[k];
        sh2o_a[k * plane + idx] = 1.0f;
        smcrel_a[k * plane + idx] = 1.0f;
    }
    Exact32 t14 = exact_mul(exact_mul(t1, t1), exact_mul(t1, t1));
    noahres_a[idx] = exact_sub(exact_sub(exact_sub(exact_sub(exact_sub(
                         exact_add(exact_sub(exact_add(solnet, lwdn), sheat),
                                   ssoil_out), eta_e),
                         exact_mul(exact_mul(emissi, 5.67051e-8f), t14)),
                         flx1), flx2), flx3);
    reslin_a[idx] = narrow(reslin);
    ebal_a[idx] = ebal_case;
    smstav_a[idx] = 0.0f;
    smstot_a[idx] = exact_mul(0.0f, 1000.0f);
    sfcrunoff_a[idx] = exact_add(sfcrunoff_a[idx],
                                 exact_mul(exact_mul(runoff1, dt), 1000.0f));
    udrunoff_a[idx] = exact_add(udrunoff_a[idx],
                                exact_mul(exact_mul(0.0f, dt), 1000.0f));
    if (ffrozp > 0.5f)
        acsnow_a[idx] = exact_add(acsnow_a[idx], exact_mul(prcp, dt));
    if (noah_f2d(snow_a[idx]) > Soft8(0)) {
        acsnom_a[idx] = exact_add(acsnom_a[idx], exact_mul(snomlt, 1000.0f));
        snopcx_a[idx] = exact_sub(snopcx_a[idx], exact_div(snomlt,
                            exact_div(dt, exact_mul(1000.0f, 3.50e+5f))));
    }
}

template <bool URBAN>

void noah_column_body(device const int*  ivgtyp,
                 device const int*  isltyp,
                 device const Exact32*  psfc_a,
                 device const Exact32*  sfcprs_a,
                 device const Exact32*  sfctmp_a,
                 device const Exact32*  qv1_a,
                 device const Exact32*  qgh_a,
                 device const Exact32*  dz8w1_a,
                 device const Exact32*  glw_a,
                 device const Exact32*  swdown_a,
                 device const Exact32*  rainbl_a,
                 device const Exact32*  sr_a,
                 device const Exact32*  chs_a,
                 device const Exact32*  cqs2_a,
                 device Exact32*  chs2_a,
                 device const Exact32*  rib_a,
                 device const Exact32*  vegfra_a,
                 device const Exact32*  shdmin_a,
                 device const Exact32*  shdmax_a,
                 device const Exact32*  tmn_a,
                 device const Exact32*  xland_a,
                 device const Exact32*  xice_a,
                 device const Exact32*  snoalb_a,
                 device const Exact32*  embck_a,
                 device Exact32*  tsk_a,
                 device Exact32*  hfx_a,
                 device Exact32*  qfx_a,
                 device Exact32*  lh_a,
                 device Exact32*  grdflx_a,
                 device Exact32*  qsfc_a,
                 device Exact32*  canwat_a,
                 device Exact32*  snow_a,
                 device Exact32*  snowc_a,
                 device Exact32*  snowh_a,
                 device Exact32*  albedo_a,
                 device Exact32*  albbck_a,
                 device Exact32*  emiss_a,
                 device Exact32*  znt_a,
                 device Exact32*  z0_a,
                 device Exact32*  snotime_a,
                 device Exact32*  lai_a,
                 device Exact32*  smstav_a,
                 device Exact32*  smstot_a,
                 device Exact32*  sfcrunoff_a,
                 device Exact32*  udrunoff_a,
                 device Exact32*  acsnow_a,
                 device Exact32*  acsnom_a,
                 device Exact32*  snopcx_a,
                 device Exact32*  potevp_a,
                 device Exact32*  noahres_a,
                 device Exact32*  reslin_a,
                 device Exact32*  chklowq_a,
                 device Exact32*  smois_a,
                 device Exact32*  tslb_a,
                 device Exact32*  sh2o_a,
                 device Exact32*  smcrel_a,
                 device int*  ebal_a,
                 device const Exact32*  vegtbl,
                 device const Exact32*  soiltbl,
                 device const Exact32*  genp,
                 device const Exact32*  dzs,
                 Exact32 dt, int lucats, int slcats, int isurban,
                 int isice, Exact32 xice_threshold, int itimestep, int frpcpn,
                 int usemonalb, int rdlai2d, int opt_thcnd,
                 int ny, int nx,
                 int urban_opt,
                 device const int*  urban_cat,
                 int urban_cat_size, int natural,
                 device const Exact32*  frc_urb2d,
                 device const Exact32*  ts_urb2d,
                 device Exact32*  tsk_rural_bep,
                 device Exact32*  rural_q1,
                 device Exact32*  rural_q2k,
                 device Exact32*  rural_zlvl, int col)
{

    if (col >= ny * nx) return;
    ulong idx = (ulong)col;
    ulong plane = (ulong)ny * nx;
    if (itimestep == 1) {
        if ((xland_a[idx] - 1.5f) >= 0.0f) {
            smstav_a[idx] = 1.0f;
            smstot_a[idx] = 1.0f;
            for (int k = 0; k < 4; ++k) {
                smois_a[k * plane + idx] = 1.0f;
                tslb_a[k * plane + idx] = 273.16f;
                smcrel_a[k * plane + idx] = 1.0f;
            }
        } else if (xice_a[idx] >= xice_threshold) {
            smstav_a[idx] = 1.0f;
            smstot_a[idx] = 1.0f;
            for (int k = 0; k < 4; ++k) {
                smois_a[k * plane + idx] = 1.0f;
                smcrel_a[k * plane + idx] = 1.0f;
            }
        }
    }
    chklowq_a[idx] = 1.0f;
    if ((xland_a[idx] - 1.5f) >= 0.0f) return;
    if (xice_a[idx] >= xice_threshold) {
        for (int k = 0; k < 4; ++k) sh2o_a[k * plane + idx] = 1.0f;
        lai_a[idx] = 0.01f;
        return;
    }
    int vegtyp = ivgtyp[idx];
    int soiltyp = isltyp[idx];
    if (vegtyp == isice) {
        int glacial_bep_opt = 0;
        device Exact32* glacial_bep = nullptr;
        if constexpr (URBAN) {
            glacial_bep_opt = urban_opt;
            glacial_bep = tsk_rural_bep;
        }
        noah_glacial_column(idx, plane, psfc_a, sfcprs_a, sfctmp_a, qv1_a,
                            qgh_a, glw_a, swdown_a, rainbl_a, sr_a, chs_a,
                            cqs2_a, chs2_a, rib_a, tmn_a, snoalb_a, embck_a,
                            tsk_a, hfx_a, qfx_a, lh_a, grdflx_a, qsfc_a,
                            canwat_a, snow_a, snowc_a, snowh_a, albedo_a,
                            albbck_a, emiss_a, znt_a, z0_a, snotime_a, lai_a,
                            smstav_a, smstot_a, sfcrunoff_a, udrunoff_a,
                            acsnow_a, acsnom_a, snopcx_a, potevp_a,
                            noahres_a, reslin_a, smois_a, tslb_a, sh2o_a,
                            smcrel_a, ebal_a, genp, dzs, dt, frpcpn,
                            glacial_bep_opt, glacial_bep);
        return;
    }
    const Exact32 tresh = 0.95f, a2 = 17.67f, a3 = 273.15f, a4 = 29.65f;
    const Exact32 a23m4 = a2 * (a3 - a4);
    Exact32 psfc = psfc_a[idx];
    Exact32 sfcprs = sfcprs_a[idx];
    Exact32 q2k = qv1_a[idx] / (1.0f + qv1_a[idx]);
    Exact32 q2sat = qgh_a[idx] / (1.0f + qgh_a[idx]);
    Exact32 sfctmp = sfctmp_a[idx];
    Exact32 apes = exact_gfk_pow(1.0e5f / psfc, 0.2857142857142857f);
    Exact32 apelm = exact_gfk_pow(1.0e5f / sfcprs, 0.2857142857142857f);
    Exact32 sfcth2 = sfctmp * apelm;
    Exact32 th2 = sfcth2 / apes;
    Exact32 emissi = emiss_a[idx];
    Exact32 lwdn = glw_a[idx] * emissi;
    Exact32 soldn = swdown_a[idx];
    Exact32 solnet = soldn * (1.0f - albedo_a[idx]);
    Exact32 prcp = rainbl_a[idx] / dt;
    Exact32 shdfac = exact_div(vegfra_a[idx], 100.0f);
    Exact32 t1 = tsk_a[idx];
    Exact32 chk = chs_a[idx];
    Exact32 shmin = exact_div(shdmin_a[idx], 100.0f);
    Exact32 shmax = exact_div(shdmax_a[idx], 100.0f);
    Exact32 sneqv = snow_a[idx] * 0.001f;
    Exact32 snowhk = snowh_a[idx];
    Exact32 sncovr = snowc_a[idx];
    Exact32 ffrozp;
    if (frpcpn) ffrozp = sr_a[idx];
    else ffrozp = (sfctmp <= 273.15f) ? 1.0f : 0.0f;
    Exact32 dqsdt2 = q2sat * a23m4 / ((sfctmp - a4) * (sfctmp - a4));
    if (noah_f2d(snow_a[idx]) > Soft8(0)) {
        Exact32 sfctsno = sfctmp;
        Exact32 e2sat = 611.2f * exact_gfk_exp(6174.0f * (1.0f / 273.15f
                                              - 1.0f / sfctsno));
        Exact32 q2sati = 0.622f * e2sat / (sfcprs - e2sat);
        q2sati = q2sati / (1.0f + q2sati);
        if (t1 > 273.14f) {
            q2sat = q2sat * (1.0f - snowc_a[idx])
                    + q2sati * snowc_a[idx];
            dqsdt2 = dqsdt2 * (1.0f - snowc_a[idx])
                     + q2sati * 6174.0f / (sfctsno * sfctsno)
                       * snowc_a[idx];
        } else {
            q2sat = q2sati;
            dqsdt2 = q2sati * 6174.0f / (sfctsno * sfctsno);
        }
        if (t1 > 273.0f && snowc_a[idx] > 0.0f && soldn > 10.0f)
            dqsdt2 = dqsdt2 * (1.0f - snowc_a[idx]);
    }
    Exact32 tbot = tmn_a[idx];
    if (soiltyp == 14 && xice_a[idx] == 0.0f) soiltyp = 7;
    Exact32 snoalb1 = snoalb_a[idx];
    Exact32 cmc = exact_div(canwat_a[idx], 1000.0f);
    Exact32 alb = albbck_a[idx];
    Exact32 z0brd = z0_a[idx];
    Exact32 embrd = embck_a[idx];
    Exact32 snotime1 = snotime_a[idx];
    Exact32 ribb = rib_a[idx];
    Exact32 smc[4], stc[4], swc[4];
    for (int k = 0; k < 4; ++k) {
        smc[k] = smois_a[k * plane + idx];
        stc[k] = tslb_a[k * plane + idx];
        swc[k] = sh2o_a[k * plane + idx];
    }
    if ((sneqv != 0.0f && snowhk == 0.0f) || (snowhk <= sneqv))
        snowhk = 5.0f * sneqv;
    bool urban_col = false;
    if constexpr (URBAN) {
        if (urban_opt > 0) {
            int cat = ivgtyp[idx];
            if (cat >= 0 && cat < urban_cat_size && urban_cat[cat] != 0) {
                urban_col = true;
                vegtyp = natural;
                shdfac = vegtbl[(ulong)(natural - 1) * 15 + 13];
                alb = 0.2f;
                emissi = 0.98f;
                lwdn = glw_a[idx] * emissi;
                solnet = soldn * (1.0f - 0.2f);
                Exact32 frc = frc_urb2d[idx];
                if (frc < 0.99f) {
                    if (urban_opt == 1) {
                        t1 = exact_div(exact_sub(tsk_a[idx],
                                                 exact_mul(frc, ts_urb2d[idx])),
                                       exact_sub(1.0f, frc));
                    } else {
                        t1 = tsk_rural_bep[idx];
                    }
                } else {
                    t1 = tsk_a[idx];
                }
            }
        }
    }
    Exact32 xlai = lai_a[idx];
    if (rdlai2d) {
        if (shdfac > 0.0f && xlai <= 0.0f) xlai = 0.01f;
    }
    Exact32 sldpth[4], zsoil[4];
    for (int k = 0; k < 4; ++k) sldpth[k] = dzs[k];
    zsoil[0] = -sldpth[0];
    for (int kz = 1; kz < 4; ++kz)
        zsoil[kz] = -sldpth[kz] + zsoil[kz - 1];
    device const Exact32* sv = soiltbl + (ulong)(soiltyp - 1) * 10;
    device const Exact32* vv = vegtbl + (ulong)(vegtyp - 1) * 15;
    Exact32 csoil = genp[6];
    Exact32 bexp = sv[0];
    Exact32 dksat = sv[6];
    Exact32 dwsat = sv[7];
    Exact32 f1 = sv[2];
    Exact32 psisat = sv[5];
    Exact32 quartz = sv[9];
    Exact32 smcdry = sv[1];
    Exact32 smcmax = sv[3];
    Exact32 smcref = sv[4];
    Exact32 smcwlt = sv[8];
    Exact32 zbot = genp[11];
    Exact32 salp = genp[7];
    Exact32 sbeta = genp[4];
    Exact32 refdk = genp[8];
    Exact32 frzk = genp[10];
    Exact32 fxexp = genp[5];
    Exact32 refkdt = genp[9];
    Exact32 kdt = refkdt * dksat / refdk;
    Exact32 slope = genp[13];
    Exact32 lvcoef = genp[12];
    Exact32 frzfact = (smcmax / smcref) * (0.412f / 0.468f);
    Exact32 frzx = frzk * frzfact;
    Exact32 topt = genp[0];
    Exact32 cmcmax = genp[1];
    Exact32 cfactr = genp[2];
    Exact32 rsmax = genp[3];
    int nroot = as_int(vv[0]);
    Exact32 snup = vv[4];
    Exact32 rsmin = vv[1];
    Exact32 rgl = vv[2];
    Exact32 hs = vv[3];
    Exact32 emissmin = vv[7];
    Exact32 emissmax = vv[8];
    Exact32 laimin = vv[5];
    Exact32 laimax = vv[6];
    Exact32 z0min = vv[11];
    Exact32 z0max = vv[12];
    Exact32 albedomin = vv[9];
    Exact32 albedomax = vv[10];
    int bare = as_int(genp[14]);
    if (vegtyp == bare) shdfac = 0.0f;
    Exact32 rtdis[4];
    for (int i = 0; i < 4; ++i) rtdis[i] = 0.0f;
    for (int i = 0; i < nroot; ++i)
        rtdis[i] = -sldpth[i] / zsoil[nroot - 1];
    if (vegtyp == isurban) {
        shdfac = 0.05f;
        rsmin = 400.0f;
        smcmax = 0.45f;
        smcref = 0.42f;
        smcwlt = 0.40f;
        smcdry = 0.40f;
    }
    Exact32 embrd_o;
    if (shdfac >= shmax) {
        embrd_o = emissmax;
        if (!rdlai2d) xlai = laimax;
        if (!usemonalb) alb = albedomin;
        z0brd = z0max;
    } else if (shdfac <= shmin) {
        embrd_o = emissmin;
        if (!rdlai2d) xlai = laimin;
        if (!usemonalb) alb = albedomax;
        z0brd = z0min;
    } else {
        if (shmax > shmin) {
            Exact32 interp_fraction = (shdfac - shmin) / (shmax - shmin);
            interp_fraction = exact_min(interp_fraction, 1.0f);
            interp_fraction = exact_max(interp_fraction, 0.0f);
            embrd_o = (1.0f - interp_fraction) * emissmin
                      + interp_fraction * emissmax;
            if (!rdlai2d) xlai = (1.0f - interp_fraction) * laimin
                                 + interp_fraction * laimax;
            if (!usemonalb) alb = (1.0f - interp_fraction) * albedomax
                                  + interp_fraction * albedomin;
            z0brd = (1.0f - interp_fraction) * z0min
                    + interp_fraction * z0max;
        } else {
            embrd_o = 0.5f * emissmin + 0.5f * emissmax;
            if (!rdlai2d) xlai = 0.5f * laimin + 0.5f * laimax;
            if (!usemonalb) alb = 0.5f * albedomin + 0.5f * albedomax;
            z0brd = 0.5f * z0min + 0.5f * z0max;
        }
    }
    embrd = embrd_o;
    bool snowng = false, frzgra = false;
    Exact32 sndens, sncond;
    if (sneqv <= 1.0e-7f) {
        sneqv = 0.0f;
        sndens = 0.0f;
        snowhk = 0.0f;
        sncond = 1.0f;
    } else {
        sndens = sneqv / snowhk;
        sncond = noah_csnow(sndens);
    }
    if (prcp > 0.0f) {
        if (ffrozp > 0.5f) snowng = true;
        else if (t1 <= 273.15f) frzgra = true;
    }
    Exact32 prcpf;
    if (snowng || frzgra) {
        Exact32 sn_new = prcp * dt * 0.001f;
        sneqv = sneqv + sn_new;
        prcpf = 0.0f;
        noah_snow_new(sfctmp, sn_new, snowhk, sndens);
        sncond = noah_csnow(sndens);
    } else {
        prcpf = prcp;
    }
    Exact32 albedo;
    if (sneqv == 0.0f) {
        sncovr = 0.0f;
        albedo = alb;
        emissi = embrd;
    } else {
        sncovr = noah_snfrac(sneqv, snup, salp);
        sncovr = exact_min(sncovr, 0.98f);
        noah_alcalc(alb, snoalb1, embrd, sncovr, dt, snowng, snotime1,
                    lvcoef, albedo, emissi);
    }
    Exact32 df1 = noah_tdfcnd(smc[0], quartz, smcmax, swc[0], bexp,
                           psisat, soiltyp, opt_thcnd);
    if (vegtyp == isurban) df1 = 3.24f;
    df1 = df1 * exact_gfk_exp(sbeta * shdfac);
    if (sncovr > 0.97f) df1 = sncond;
    Exact32 dsoil = -(0.5f * zsoil[0]);
    Exact32 dtot = 0.0f;
    Exact32 ssoil;
    if (sneqv == 0.0f) {
        ssoil = df1 * (t1 - stc[0]) / dsoil;
    } else {
        dtot = snowhk + dsoil;
        Exact32 frcsno = snowhk / dtot;
        Exact32 frcsoi = dsoil / dtot;
        Exact32 df1a = frcsno * sncond + frcsoi * df1;
        df1 = df1a * sncovr + df1 * (1.0f - sncovr);
        ssoil = df1 * (t1 - stc[0]) / dtot;
    }
    Exact32 z0k;
    if (sncovr > 0.0f) z0k = noah_snowz0(sncovr, z0brd, snowhk);
    else z0k = z0brd;
    Exact32 fdown = solnet + lwdn;
    Exact32 t2v = sfctmp * (1.0f + 0.61f * q2k);
    Exact32 etp, rch, rr, epsca, t24, flx2;
    noah_penman(sfctmp, sfcprs, chk, t2v, th2, prcp, fdown, ssoil, q2k,
                q2sat, dqsdt2, snowng, frzgra, emissi, sncovr, etp,
                rch, rr, epsca, t24, flx2);
    Exact32 rc = 0.0f, pc = 0.0f;
    if (shdfac > 0.0f && xlai > 0.0f)
        noah_canres(soldn, chk, sfctmp, q2k, swc, zsoil, smcwlt,
                    smcref, rsmin, nroot, q2sat, dqsdt2, topt, rsmax,
                    rgl, hs, xlai, sfcprs, emissi, rc, pc);
    Exact32 eta_kin, beta, flx1, flx3, ssoil_pac;
    Exact32 runoff1, runoff2, runoff3, dew, drip;
    Exact32 edir1k = 0.0f, ec1k = 0.0f, ett1k = 0.0f;
    Exact32 et1k[4];
    for (int k = 0; k < 4; ++k) et1k[k] = 0.0f;
    Exact32 etns = 0.0f, esnow = 0.0f, snomlt = 0.0f;
    int ebal_case;
    Soft8 reslin;
    if (sneqv == 0.0f) {
        ebal_case = 0;
        Exact32 prcp1 = prcp * 0.001f;
        Exact32 etp1 = etp * 0.001f;
        dew = 0.0f;
        Exact32 eta = 0.0f;
        if (etp > 0.0f) {
            Exact32 eta1;
            noah_evapo(eta1, smc, cmc, etp1, dt, swc, smcmax, pc,
                       smcwlt, smcref, shdfac, cmcmax, smcdry, cfactr,
                       nroot, rtdis, fxexp, edir1k, ec1k, et1k, ett1k);
            noah_smflx(smc, cmc, dt, prcp1, zsoil, swc, slope, kdt,
                       frzx, smcmax, bexp, smcwlt, dksat, dwsat,
                       shdfac, cmcmax, edir1k, ec1k, et1k, runoff1,
                       runoff2, runoff3, drip);
            eta = eta1 * 1000.0f;
        } else {
            dew = -etp1;
            prcp1 = prcp1 + dew;
            noah_smflx(smc, cmc, dt, prcp1, zsoil, swc, slope, kdt,
                       frzx, smcmax, bexp, smcwlt, dksat, dwsat,
                       shdfac, cmcmax, edir1k, ec1k, et1k, runoff1,
                       runoff2, runoff3, drip);
        }
        if (etp <= 0.0f) {
            beta = 0.0f;
            eta = etp;
            if (etp < 0.0f) beta = 1.0f;
        } else {
            beta = eta / etp;
        }
        Exact32 df1n = noah_tdfcnd(smc[0], quartz, smcmax, swc[0], bexp,
                                psisat, soiltyp, opt_thcnd);
        if (vegtyp == isurban) df1n = 3.24f;
        df1n = df1n * exact_gfk_exp(sbeta * shdfac);
        Exact32 yynum = fdown - emissi * 5.67e-8f * t24;
        Exact32 yy = sfctmp + (yynum / rch + th2 - sfctmp - beta * epsca)
                           / rr;
        Exact32 zz1 = df1n / (-0.5f * zsoil[0] * rch * rr) + 1.0f;
        noah_shflx(stc, smc, smcmax, t1, dt, yy, zz1, zsoil, tbot,
                   zbot, psisat, swc, bexp, df1n, quartz, csoil,
                   vegtyp, isurban, soiltyp, opt_thcnd, ssoil_pac);
        flx1 = 4.218e+3f * prcp * (t1 - sfctmp);
        flx3 = 0.0f;
        eta_kin = eta;
        reslin = Soft8(0);
    } else {
        const Exact32 esdmin = 1.0e-6f, snoexp = 2.0f;
        Exact32 esd = sneqv;
        dew = 0.0f;
        Exact32 etns1 = 0.0f;
        Exact32 esnow1, esnow2;
        esnow = 0.0f;
        esnow1 = 0.0f;
        esnow2 = 0.0f;
        Exact32 prcp1 = prcpf * 0.001f;
        beta = 1.0f;
        Exact32 etanrg, etp1;
        if (etp <= 0.0f) {
            if (ribb >= 0.1f && fdown > 150.0f) {
                etp = exact_div((exact_div(exact_min(etp * (1.0f - ribb), 0.0f) * sncovr,
                       0.980f) + etp * (0.980f - sncovr)), 0.980f);
            }
            if (etp == 0.0f) beta = 0.0f;
            etp1 = etp * 0.001f;
            dew = -etp1;
            esnow2 = etp1 * dt;
            etanrg = etp * ((1.0f - sncovr) * 2.501e+6f
                            + sncovr * 2.83e+6f);
        } else {
            etp1 = etp * 0.001f;
            if (sncovr < 1.0f) {
                noah_evapo(etns1, smc, cmc, etp1, dt, swc, smcmax, pc,
                           smcwlt, smcref, shdfac, cmcmax, smcdry,
                           cfactr, nroot, rtdis, fxexp, edir1k, ec1k,
                           et1k, ett1k);
                edir1k = edir1k * (1.0f - sncovr);
                ec1k = ec1k * (1.0f - sncovr);
                for (int k = 0; k < 4; ++k)
                    et1k[k] = et1k[k] * (1.0f - sncovr);
                ett1k = ett1k * (1.0f - sncovr);
                etns1 = etns1 * (1.0f - sncovr);
                etns = etns1 * 1000.0f;
            }
            esnow = etp * sncovr;
            esnow1 = esnow * 0.001f;
            esnow2 = esnow1 * dt;
            etanrg = esnow * 2.83e+6f + etns * 2.501e+6f;
        }
        flx1 = 0.0f;
        if (snowng) flx1 = 2.106e+3f * prcp * (t1 - sfctmp);
        else if (prcp > 0.0f) flx1 = 4.218e+3f * prcp * (t1 - sfctmp);
        Exact32 dsoil2 = -0.5f * zsoil[0];
        dtot = snowhk + dsoil2;
        Exact32 denom = 1.0f + df1 / (dtot * rr * rch);
        Exact32 t12a = ((fdown - flx1 - flx2 - emissi * 5.67e-8f * t24)
                     / rch + th2 - sfctmp - etanrg / rch) / rr;
        Exact32 t12b = df1 * stc[0] / (dtot * rr * rch);
        Exact32 t12 = (sfctmp + t12a + t12b) / denom;
        Exact32 stc1_old = stc[0];
        Exact32 ex = 0.0f;
        snomlt = 0.0f;
        if (t12 <= 273.15f) {
            ebal_case = 1;
            t1 = t12;
            ssoil_pac = df1 * (t1 - stc[0]) / dtot;
            esd = exact_max(0.0f, esd - esnow2);
            flx3 = 0.0f;
            reslin = (widen(fdown) - widen(flx1) - widen(flx2)
                      - widen(emissi) * widen(5.67e-8f) * widen(t24)
                      + widen(rch) * (widen(th2) - widen(sfctmp))
                      - widen(etanrg)
                      - widen(rch) * widen(rr)
                        * (widen(t12) - widen(sfctmp))
                      - widen(df1) * (widen(t12) - widen(stc1_old))
                        / widen(dtot));
        } else {
            t1 = 273.15f * exact_max(0.01f, exact_gfk_pow(sncovr, snoexp))
                 + t12 * (1.0f - exact_max(0.01f, exact_gfk_pow(sncovr, snoexp)));
            beta = 1.0f;
            ssoil_pac = df1 * (t1 - stc[0]) / dtot;
            bool clipped = false;
            if (esd - esnow2 <= esdmin) {
                esd = 0.0f;
                ex = 0.0f;
                snomlt = 0.0f;
                flx3 = 0.0f;
                clipped = true;
            } else {
                esd = esd - esnow2;
                Exact32 seh = rch * (t1 - th2);
                Exact32 t14 = t1 * t1;
                t14 = t14 * t14;
                flx3 = fdown - flx1 - flx2 - emissi * 5.67e-8f * t14
                       - ssoil_pac - seh - etanrg;
                Exact32 flx3_def = flx3;
                if (flx3 <= 0.0f) flx3 = 0.0f;
                if (flx3 != flx3_def) clipped = true;
                ex = exact_div(flx3 * 0.001f, 3.335e+5f);
                snomlt = ex * dt;
                if (esd - snomlt >= esdmin) {
                    esd = esd - snomlt;
                } else {
                    ex = esd / dt;
                    flx3 = ex * 1000.0f * 3.335e+5f;
                    snomlt = esd;
                    esd = 0.0f;
                    clipped = true;
                }
            }
            ebal_case = clipped ? 3 : 2;
            prcp1 = prcp1 + ex;
            Exact32 seh = rch * (t1 - th2);
            Exact32 t14 = t1 * t1;
            t14 = t14 * t14;
            reslin = (widen(fdown) - widen(flx1) - widen(flx2)
                      - widen(emissi) * widen(5.67e-8f) * widen(t14)
                      - widen(ssoil_pac) - widen(seh)
                      - widen(etanrg) - widen(flx3));
        }
        noah_smflx(smc, cmc, dt, prcp1, zsoil, swc, slope, kdt,
                   frzx, smcmax, bexp, smcwlt, dksat, dwsat, shdfac,
                   cmcmax, edir1k, ec1k, et1k, runoff1, runoff2,
                   runoff3, drip);
        Exact32 zz1 = 1.0f;
        Exact32 yy = stc[0] - 0.5f * ssoil_pac * zsoil[0] * zz1 / df1;
        Exact32 t11 = t1;
        Exact32 ssoil1;
        noah_shflx(stc, smc, smcmax, t11, dt, yy, zz1, zsoil, tbot,
                   zbot, psisat, swc, bexp, df1, quartz, csoil,
                   vegtyp, isurban, soiltyp, opt_thcnd, ssoil1);
        if (esd > 0.0f) {
            noah_snowpack(esd, dt, snowhk, sndens, t1, yy);
        } else {
            esd = 0.0f;
            snowhk = 0.0f;
            sndens = 0.0f;
            sncovr = 0.0f;
        }
        sneqv = esd;
        eta_kin = esnow + etns - 1000.0f * dew;
    }
    Exact32 q1 = q2k + eta_kin * 1004.5f / rch;
    Exact32 sheat = -(chk * 1004.5f * sfcprs) / (287.04f * t2v) * (th2 - t1);
    Exact32 edir_e = (edir1k * 1000.0f) * 2.501e+6f;
    Exact32 ec_e = (ec1k * 1000.0f) * 2.501e+6f;
    Exact32 ett_e = (ett1k * 1000.0f) * 2.501e+6f;
    Exact32 esnow_e = esnow * 2.83e+6f;
    Exact32 etp_e = etp * ((1.0f - sncovr) * 2.501e+6f + sncovr * 2.83e+6f);
    Exact32 eta_e;
    if (etp_e > 0.0f) eta_e = edir_e + ec_e + ett_e + esnow_e;
    else eta_e = etp_e;
    if (etp_e == 0.0f) beta = 0.0f;
    else beta = eta_e / etp_e;
    if (ebal_case == 0) {
        reslin = (widen(fdown)
                  - widen(emissi) * widen(5.67e-8f) * widen(t24)
                  + widen(rch) * (widen(th2) - widen(sfctmp))
                  - widen(eta_e)
                  - widen(rch) * widen(rr)
                    * (widen(t1) - widen(sfctmp))
                  - widen(ssoil_pac));
    }
    Exact32 ssoil_out = -1.0f * ssoil_pac;
    runoff3 = runoff3 / dt;
    runoff2 = runoff2 + runoff3;
    Exact32 soilm = -1.0f * smc[0] * zsoil[0];
    for (int k = 1; k < 4; ++k)
        soilm = soilm + smc[k] * (zsoil[k - 1] - zsoil[k]);
    Exact32 soilwm = -1.0f * (smcmax - smcwlt) * zsoil[0];
    Exact32 soilww = -1.0f * (smc[0] - smcwlt) * zsoil[0];
    Exact32 smav[4];
    for (int k = 0; k < 4; ++k)
        smav[k] = (smc[k] - smcwlt) / (smcmax - smcwlt);
    if (nroot >= 2) {
        for (int k = 1; k < nroot; ++k) {
            soilwm = soilwm + (smcmax - smcwlt) * (zsoil[k - 1]
                                                   - zsoil[k]);
            soilww = soilww + (smc[k] - smcwlt) * (zsoil[k - 1]
                                                   - zsoil[k]);
        }
    }
    Exact32 soilw;
    if (soilwm < 1.0e-6f) {
        soilwm = 0.0f;
        soilw = 0.0f;
        soilm = 0.0f;
    } else {
        soilw = soilww / soilwm;
    }
    lai_a[idx] = xlai;
    canwat_a[idx] = cmc * 1000.0f;
    snow_a[idx] = sneqv * 1000.0f;
    snowh_a[idx] = snowhk;
    albedo_a[idx] = albedo;
    albbck_a[idx] = alb;
    z0_a[idx] = z0brd;
    emiss_a[idx] = emissi;
    znt_a[idx] = z0k;
    tsk_a[idx] = t1;
    if constexpr (URBAN) {
        if (urban_opt >= 2) tsk_rural_bep[idx] = t1;
    }
    hfx_a[idx] = sheat;
    potevp_a[idx] = potevp_a[idx] + etp_e * (exact_div(dt, (2500000.0f * 1000.0f)));
    qfx_a[idx] = eta_kin;
    lh_a[idx] = eta_e;
    grdflx_a[idx] = ssoil_out;
    snowc_a[idx] = sncovr;
    chs2_a[idx] = cqs2_a[idx];
    snotime_a[idx] = snotime1;
    qsfc_a[idx] = q1 / (1.0f - q1);
    if constexpr (URBAN) {
        if (urban_col) {
            rural_q1[idx] = q1;
            rural_q2k[idx] = q2k;
            rural_zlvl[idx] = 0.5f * dz8w1_a[idx];
        }
    }
    for (int k = 0; k < 4; ++k) {
        smois_a[k * plane + idx] = smc[k];
        tslb_a[k * plane + idx] = stc[k];
        sh2o_a[k * plane + idx] = swc[k];
        smcrel_a[k * plane + idx] = smav[k];
    }
    noahres_a[idx] = exact_sub(exact_sub(exact_sub(exact_sub(exact_sub(
                         exact_add(exact_sub(exact_add(solnet, lwdn), sheat),
                                   ssoil_out), eta_e),
                         exact_mul(exact_mul(emissi, 5.67051e-8f),
                                   noah_powi(t1, 4))),
                         flx1), flx2), flx3);
    reslin_a[idx] = narrow(reslin);
    ebal_a[idx] = ebal_case;
    chklowq_a[idx] = 1.0f;
    smstav_a[idx] = soilw;
    smstot_a[idx] = soilm * 1000.0f;
    sfcrunoff_a[idx] = sfcrunoff_a[idx] + runoff1 * dt * 1000.0f;
    udrunoff_a[idx] = udrunoff_a[idx] + runoff2 * dt * 1000.0f;
    if (ffrozp > 0.5f)
        acsnow_a[idx] = acsnow_a[idx] + prcp * dt;
    if (snow_a[idx] > 0.0f) {
        acsnom_a[idx] = acsnom_a[idx] + snomlt * 1000.0f;
        snopcx_a[idx] = exact_sub(snopcx_a[idx], exact_div(snomlt,
                            exact_div(dt, exact_mul(1000.0f, 3.50e+5f))));
    }
}
kernel void wrf_exact_noah_column(
    device WrfExactNoahColumnResources& r [[buffer(0)]],
    constant WrfExactNoahColumnParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const uint n = uint(p.ny) * uint(p.nx);
    if (uint(p.ny) == 0u || uint(p.nx) == 0u || uint(p.lucats) == 0u || uint(p.slcats) == 0u || gid >= n) return;
    noah_column_body<false>(
        r.ivgtyp,
        r.isltyp,
        reinterpret_cast<device const Exact32*>(r.psfc),
        reinterpret_cast<device const Exact32*>(r.sfcprs),
        reinterpret_cast<device const Exact32*>(r.sfctmp),
        reinterpret_cast<device const Exact32*>(r.qv1),
        reinterpret_cast<device const Exact32*>(r.qgh),
        reinterpret_cast<device const Exact32*>(r.dz8w1),
        reinterpret_cast<device const Exact32*>(r.glw),
        reinterpret_cast<device const Exact32*>(r.swdown),
        reinterpret_cast<device const Exact32*>(r.rainbl),
        reinterpret_cast<device const Exact32*>(r.sr),
        reinterpret_cast<device const Exact32*>(r.chs),
        reinterpret_cast<device const Exact32*>(r.cqs2),
        reinterpret_cast<device Exact32*>(r.chs2),
        reinterpret_cast<device const Exact32*>(r.rib),
        reinterpret_cast<device const Exact32*>(r.vegfra),
        reinterpret_cast<device const Exact32*>(r.shdmin),
        reinterpret_cast<device const Exact32*>(r.shdmax),
        reinterpret_cast<device const Exact32*>(r.tmn),
        reinterpret_cast<device const Exact32*>(r.xland),
        reinterpret_cast<device const Exact32*>(r.xice),
        reinterpret_cast<device const Exact32*>(r.snoalb),
        reinterpret_cast<device const Exact32*>(r.embck),
        reinterpret_cast<device Exact32*>(r.tsk),
        reinterpret_cast<device Exact32*>(r.hfx),
        reinterpret_cast<device Exact32*>(r.qfx),
        reinterpret_cast<device Exact32*>(r.lh),
        reinterpret_cast<device Exact32*>(r.grdflx),
        reinterpret_cast<device Exact32*>(r.qsfc),
        reinterpret_cast<device Exact32*>(r.canwat),
        reinterpret_cast<device Exact32*>(r.snow),
        reinterpret_cast<device Exact32*>(r.snowc),
        reinterpret_cast<device Exact32*>(r.snowh),
        reinterpret_cast<device Exact32*>(r.albedo),
        reinterpret_cast<device Exact32*>(r.albbck),
        reinterpret_cast<device Exact32*>(r.emiss),
        reinterpret_cast<device Exact32*>(r.znt),
        reinterpret_cast<device Exact32*>(r.z0),
        reinterpret_cast<device Exact32*>(r.snotime),
        reinterpret_cast<device Exact32*>(r.lai),
        reinterpret_cast<device Exact32*>(r.smstav),
        reinterpret_cast<device Exact32*>(r.smstot),
        reinterpret_cast<device Exact32*>(r.sfcrunoff),
        reinterpret_cast<device Exact32*>(r.udrunoff),
        reinterpret_cast<device Exact32*>(r.acsnow),
        reinterpret_cast<device Exact32*>(r.acsnom),
        reinterpret_cast<device Exact32*>(r.snopcx),
        reinterpret_cast<device Exact32*>(r.potevp),
        reinterpret_cast<device Exact32*>(r.noahres),
        reinterpret_cast<device Exact32*>(r.reslin),
        reinterpret_cast<device Exact32*>(r.chklowq),
        reinterpret_cast<device Exact32*>(r.smois),
        reinterpret_cast<device Exact32*>(r.tslb),
        reinterpret_cast<device Exact32*>(r.sh2o),
        reinterpret_cast<device Exact32*>(r.smcrel),
        r.ebal,
        reinterpret_cast<device const Exact32*>(r.vegtbl),
        reinterpret_cast<device const Exact32*>(r.soiltbl),
        reinterpret_cast<device const Exact32*>(r.genp),
        reinterpret_cast<device const Exact32*>(r.dzs),
        p.dt,
        p.lucats,
        p.slcats,
        p.isurban,
        p.isice,
        p.xice_threshold,
        p.itimestep,
        p.frpcpn,
        p.usemonalb,
        p.rdlai2d,
        p.opt_thcnd,
        p.ny,
        p.nx,
        0, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, int(gid));
}


kernel void wrf_exact_noah_column_urban(
    device WrfExactNoahColumnUrbanResources& r [[buffer(0)]],
    constant WrfExactNoahColumnUrbanParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const uint n = uint(p.ny) * uint(p.nx);
    if (uint(p.ny) == 0u || uint(p.nx) == 0u || uint(p.lucats) == 0u || uint(p.slcats) == 0u || gid >= n) return;
    noah_column_body<true>(
        r.ivgtyp,
        r.isltyp,
        reinterpret_cast<device const Exact32*>(r.psfc),
        reinterpret_cast<device const Exact32*>(r.sfcprs),
        reinterpret_cast<device const Exact32*>(r.sfctmp),
        reinterpret_cast<device const Exact32*>(r.qv1),
        reinterpret_cast<device const Exact32*>(r.qgh),
        reinterpret_cast<device const Exact32*>(r.dz8w1),
        reinterpret_cast<device const Exact32*>(r.glw),
        reinterpret_cast<device const Exact32*>(r.swdown),
        reinterpret_cast<device const Exact32*>(r.rainbl),
        reinterpret_cast<device const Exact32*>(r.sr),
        reinterpret_cast<device const Exact32*>(r.chs),
        reinterpret_cast<device const Exact32*>(r.cqs2),
        reinterpret_cast<device Exact32*>(r.chs2),
        reinterpret_cast<device const Exact32*>(r.rib),
        reinterpret_cast<device const Exact32*>(r.vegfra),
        reinterpret_cast<device const Exact32*>(r.shdmin),
        reinterpret_cast<device const Exact32*>(r.shdmax),
        reinterpret_cast<device const Exact32*>(r.tmn),
        reinterpret_cast<device const Exact32*>(r.xland),
        reinterpret_cast<device const Exact32*>(r.xice),
        reinterpret_cast<device const Exact32*>(r.snoalb),
        reinterpret_cast<device const Exact32*>(r.embck),
        reinterpret_cast<device Exact32*>(r.tsk),
        reinterpret_cast<device Exact32*>(r.hfx),
        reinterpret_cast<device Exact32*>(r.qfx),
        reinterpret_cast<device Exact32*>(r.lh),
        reinterpret_cast<device Exact32*>(r.grdflx),
        reinterpret_cast<device Exact32*>(r.qsfc),
        reinterpret_cast<device Exact32*>(r.canwat),
        reinterpret_cast<device Exact32*>(r.snow),
        reinterpret_cast<device Exact32*>(r.snowc),
        reinterpret_cast<device Exact32*>(r.snowh),
        reinterpret_cast<device Exact32*>(r.albedo),
        reinterpret_cast<device Exact32*>(r.albbck),
        reinterpret_cast<device Exact32*>(r.emiss),
        reinterpret_cast<device Exact32*>(r.znt),
        reinterpret_cast<device Exact32*>(r.z0),
        reinterpret_cast<device Exact32*>(r.snotime),
        reinterpret_cast<device Exact32*>(r.lai),
        reinterpret_cast<device Exact32*>(r.smstav),
        reinterpret_cast<device Exact32*>(r.smstot),
        reinterpret_cast<device Exact32*>(r.sfcrunoff),
        reinterpret_cast<device Exact32*>(r.udrunoff),
        reinterpret_cast<device Exact32*>(r.acsnow),
        reinterpret_cast<device Exact32*>(r.acsnom),
        reinterpret_cast<device Exact32*>(r.snopcx),
        reinterpret_cast<device Exact32*>(r.potevp),
        reinterpret_cast<device Exact32*>(r.noahres),
        reinterpret_cast<device Exact32*>(r.reslin),
        reinterpret_cast<device Exact32*>(r.chklowq),
        reinterpret_cast<device Exact32*>(r.smois),
        reinterpret_cast<device Exact32*>(r.tslb),
        reinterpret_cast<device Exact32*>(r.sh2o),
        reinterpret_cast<device Exact32*>(r.smcrel),
        r.ebal,
        reinterpret_cast<device const Exact32*>(r.tables),
        (reinterpret_cast<device const Exact32*>(r.tables) + p.lucats * 15),
        (reinterpret_cast<device const Exact32*>(r.tables) + p.lucats * 15 + p.slcats * 10),
        (reinterpret_cast<device const Exact32*>(r.tables) + p.lucats * 15 + p.slcats * 10 + 16),
        p.dt,
        p.lucats,
        p.slcats,
        p.isurban,
        p.isice,
        p.xice_threshold,
        p.itimestep,
        p.frpcpn,
        p.usemonalb,
        p.rdlai2d,
        p.opt_thcnd,
        p.ny,
        p.nx,
        p.urban_opt, r.urban_cat, p.urban_cat_size, p.natural,
        (reinterpret_cast<device const Exact32*>(r.tables) + p.lucats * 15 + p.slcats * 10 + 20),
        (reinterpret_cast<device const Exact32*>(r.tables) + p.lucats * 15 + p.slcats * 10 + 20 + n), reinterpret_cast<device Exact32*>(r.tsk_rural_bep), reinterpret_cast<device Exact32*>(r.rural_q1),
        reinterpret_cast<device Exact32*>(r.rural_q2k), reinterpret_cast<device Exact32*>(r.rural_zlvl), int(gid));
}
