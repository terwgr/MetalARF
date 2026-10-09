#include <metal_stdlib>
using namespace metal;
#include "../generated/noah_column_bindings.metalh"
#include "../generated/noah_check_soil_categories_bindings.metalh"
#include "../generated/noah_initialize_sh2o_bindings.metalh"
#include "soft_binary64.metalh"
#pragma clang fp contract(off)
typedef float real;
#define CP 1004.5f
#define RCP 0.2857142984867096f
#define XLV 2.5e6f
#define RHOWATER 1000.0f
static inline real __fdiv_rn(real a, real b) { return a / b; }
static inline real __fmul_rn(real a, real b) { return a * b; }
static inline real __fadd_rn(real a, real b) { return a + b; }
static inline real __fsub_rn(real a, real b) { return a - b; }
#define NSOIL 4

#define NRD      287.04f      // noahlsm module RD
#define NSIGMA   5.67e-8f     // noahlsm SIGMA
#define NCPH2O   4.218e+3f    // noahlsm CPH2O
#define NCPICE   2.106e+3f    // noahlsm CPICE
#define NLSUBF   3.335e+5f    // noahlsm LSUBF
#define NEMISSI_S 0.95f       // noahlsm EMISSI_S
#define NTFREEZ  273.15f      // SFLX TFREEZ
#define NLVH2O   2.501e+6f    // SFLX LVH2O
#define NLSUBS   2.83e+6f     // SFLX/PENMAN/SNOPAC LSUBS
#define NR_SHEAT 287.04f      // SFLX local R
#define NCP_PEN  1004.6f      // PENMAN's local CP
#define NELCP    2.4888e+3f   // PENMAN ELCP
#define NLSUBC   2.501e+6f    // PENMAN/SNOPAC LSUBC
#define NSTBOLT  5.67051e-8f  // module_model_constants STBOLT (NOAHRES)
#define NXLF     3.50e+5f     // module_model_constants XLF (SNOPCX)

// packed-table column indices (gpuwm/core/noah.py VEG_COLS/SOIL_COLS/GEN)
#define NVEGC 15
#define VG_NROOT 0
#define VG_RSMIN 1
#define VG_RGL 2
#define VG_HS 3
#define VG_SNUP 4
#define VG_LAIMIN 5
#define VG_LAIMAX 6
#define VG_EMISSMIN 7
#define VG_EMISSMAX 8
#define VG_ALBEDOMIN 9
#define VG_ALBEDOMAX 10
#define VG_Z0MIN 11
#define VG_Z0MAX 12
#define VG_SHDTBL 13
#define VG_MAXALB 14
#define NSOILC 10
#define SO_BEXP 0
#define SO_SMCDRY 1
#define SO_F1 2
#define SO_SMCMAX 3
#define SO_SMCREF 4
#define SO_PSISAT 5
#define SO_DKSAT 6
#define SO_DWSAT 7
#define SO_SMCWLT 8
#define SO_QUARTZ 9
#define GEN_TOPT 0
#define GEN_CMCMAX 1
#define GEN_CFACTR 2
#define GEN_RSMAX 3
#define GEN_SBETA 4
#define GEN_FXEXP 5
#define GEN_CSOIL 6
#define GEN_SALP 7
#define GEN_REFDK 8
#define GEN_REFKDT 9
#define GEN_FRZK 10
#define GEN_ZBOT 11
#define GEN_LVCOEF 12
#define GEN_SLOPE 13
#define GEN_BARE 14
#define GEN_NATURAL 15

static inline ulong noah_widen64(real x) { return soft64_from_f32_bits(as_type<uint>(x)); }
static inline real noah_narrow64(ulong x) { return as_type<float>(soft64_to_f32_bits(x)); }
static inline ulong noah_reslin_snow_cold(real fdown, real flx1, real flx2, real emissi, real t24, real rch, real th2, real sfctmp, real etanrg, real rr, real t12, real df1, real stc1_old, real dtot) {
    ulong x = soft64_sub(noah_widen64(fdown), noah_widen64(flx1));
    x = soft64_sub(x, noah_widen64(flx2));
    ulong rad = soft64_mul(soft64_mul(noah_widen64(emissi), noah_widen64(NSIGMA)), noah_widen64(t24));
    x = soft64_sub(x, rad);
    ulong sensible = soft64_mul(noah_widen64(rch), soft64_sub(noah_widen64(th2), noah_widen64(sfctmp)));
    x = soft64_add(x, sensible);
    x = soft64_sub(x, noah_widen64(etanrg));
    ulong turbulent = soft64_mul(soft64_mul(noah_widen64(rch), noah_widen64(rr)), soft64_sub(noah_widen64(t12), noah_widen64(sfctmp)));
    x = soft64_sub(x, turbulent);
    ulong soil = soft64_div(soft64_mul(noah_widen64(df1), soft64_sub(noah_widen64(t12), noah_widen64(stc1_old))), noah_widen64(dtot));
    return soft64_sub(x, soil);
}
static inline ulong noah_reslin_snow_warm(real fdown, real flx1, real flx2, real emissi, real t14, real ssoil_pac, real seh, real etanrg, real flx3) {
    ulong x = soft64_sub(noah_widen64(fdown), noah_widen64(flx1));
    x = soft64_sub(x, noah_widen64(flx2));
    ulong rad = soft64_mul(soft64_mul(noah_widen64(emissi), noah_widen64(NSIGMA)), noah_widen64(t14));
    x = soft64_sub(x, rad);
    x = soft64_sub(x, noah_widen64(ssoil_pac));
    x = soft64_sub(x, noah_widen64(seh));
    x = soft64_sub(x, noah_widen64(etanrg));
    return soft64_sub(x, noah_widen64(flx3));
}
static inline ulong noah_reslin_nopac(real fdown, real emissi, real t24, real rch, real th2, real sfctmp, real eta_e, real rr, real t1, real ssoil_pac) {
    ulong x = noah_widen64(fdown);
    ulong rad = soft64_mul(soft64_mul(noah_widen64(emissi), noah_widen64(NSIGMA)), noah_widen64(t24));
    x = soft64_sub(x, rad);
    ulong sensible = soft64_mul(noah_widen64(rch), soft64_sub(noah_widen64(th2), noah_widen64(sfctmp)));
    x = soft64_add(x, sensible);
    x = soft64_sub(x, noah_widen64(eta_e));
    ulong turbulent = soft64_mul(soft64_mul(noah_widen64(rch), noah_widen64(rr)), soft64_sub(noah_widen64(t1), noah_widen64(sfctmp)));
    x = soft64_sub(x, turbulent);
    return soft64_sub(x, noah_widen64(ssoil_pac));
}
static inline ulong noah_energy_residual(real solnet, real lwdn, real sheat, real ssoil_out, real eta_e, real emissi, real t1, real flx1, real flx2, real flx3) {
    ulong x = soft64_add(noah_widen64(solnet), noah_widen64(lwdn));
    x = soft64_sub(x, noah_widen64(sheat));
    x = soft64_add(x, noah_widen64(ssoil_out));
    x = soft64_sub(x, noah_widen64(eta_e));
    ulong t = noah_widen64(t1);
    ulong emitted = soft64_mul(noah_widen64(emissi), noah_widen64(NSTBOLT));
    emitted = soft64_mul(emitted, t);
    emitted = soft64_mul(emitted, t);
    emitted = soft64_mul(emitted, t);
    emitted = soft64_mul(emitted, t);
    x = soft64_sub(x, emitted);
    x = soft64_sub(x, noah_widen64(flx1));
    x = soft64_sub(x, noah_widen64(flx2));
    return soft64_sub(x, noah_widen64(flx3));
}
// ---------------------------------------------------------------- FRH2O
 static real noah_frh2o(real tkelv, real smc, real sh2o,
                                  real smcmax, real bexp, real psis)
{
    const real ck = 8.0f, blim = 5.5f, error = 0.005f;
    const real hlice = 3.335e5f, gs = 9.81f, t0 = 273.15f;
    real bx = (bexp <= blim) ? bexp : blim;
    int nlog = 0, kcount = 0;
    if (tkelv > (t0 - 1.0e-3f)) return smc;
    real swl = smc - sh2o;
    if (swl > (smc - 0.02f)) swl = smc - 0.02f;
    if (swl < 0.0f) swl = 0.0f;
    while (nlog < 10 && kcount == 0) {
        nlog += 1;
        real df = log(__fdiv_rn(psis * gs, hlice)
                       * pow(1.0f + ck * swl, 2.0f)
                       * pow(smcmax / (smc - swl), bx))
                  - log(-(tkelv - t0) / tkelv);
        real denom = 2.0f * ck / (1.0f + ck * swl) + bx / (smc - swl);
        real swlk = swl - df / denom;
        if (swlk > (smc - 0.02f)) swlk = smc - 0.02f;
        if (swlk < 0.0f) swlk = 0.0f;
        real dswl = fabs(swlk - swl);
        swl = swlk;
        if (dswl <= error) kcount += 1;
    }
    real freew = smc - swl;
    if (kcount == 0) {                 // Flerchinger explicit fallback
        real fk = pow((hlice / (gs * (-psis)))
                       * ((tkelv - t0) / tkelv), -1.0f / bx) * smcmax;
        if (fk < 0.02f) fk = 0.02f;
        freew = fmin(fk, smc);
    }
    return freew;
}

// ---------------------------------------------------------------- CSNOW
 static real noah_csnow(real dsnow)
{
    return 2.0f * 0.11631f * (0.328f * pow(10.0f, 2.25f * dsnow));
}

// ------------------------------------------------------------- SNOW_NEW
 static void noah_snow_new(real temp, real newsn, thread real& snowh,
                                     thread real& sndens)
{
    real snowhc = snowh * 100.0f;
    real newsnc = newsn * 100.0f;
    real tempc = temp - 273.15f;
    real dsnew;
    if (tempc <= -15.0f) dsnew = 0.05f;
    else dsnew = 0.05f + 0.0017f * pow(tempc + 15.0f, 1.5f);
    real hnewc = newsnc / dsnew;
    if (snowhc + hnewc < 1.0e-3f) sndens = fmax(dsnew, sndens);
    else sndens = (snowhc * sndens + hnewc * dsnew) / (snowhc + hnewc);
    snowhc = snowhc + hnewc;
    snowh = snowhc * 0.01f;
}

// --------------------------------------------------------------- SNFRAC
 static real noah_snfrac(real sneqv, real snup, real salp)
{
    if (sneqv < snup) {
        real rsnow = sneqv / snup;
        return 1.0f - (exp(-salp * rsnow) - rsnow * exp(-salp));
    }
    return 1.0f;
}

// --------------------------------------------------------------- ALCALC
 static void noah_alcalc(real alb, real snoalb, real embrd,
                                   real sncovr, real dt, bool snowng,
                                   thread real& snotime1, real lvcoef,
                                   thread real& albedo, thread real& emissi)
{
    const real snacca = 0.94f, snaccb = 0.58f;
    albedo = alb + sncovr * (snoalb - alb);
    emissi = embrd + sncovr * (NEMISSI_S - embrd);
    real snoalb1 = snoalb + lvcoef * (0.85f - snoalb);
    real snoalb2 = snoalb1;
    if (snowng) {
        snotime1 = 0.0f;
    } else {
        snotime1 = snotime1 + dt;
        snoalb2 = snoalb1 * pow(snacca,
                                 pow(__fdiv_rn(snotime1, 86400.0f), snaccb));
    }
    snoalb2 = fmax(snoalb2, alb);
    albedo = alb + sncovr * (snoalb2 - alb);
    if (albedo > snoalb2) albedo = snoalb2;
}

// --------------------------------------------------------------- TDFCND
 static real noah_tdfcnd(real smc, real qz, real smcmax,
                                   real sh2o, real bexp, real psisat,
                                   int soiltyp, int opt_thcnd)
{
    if (opt_thcnd == 1 || (opt_thcnd == 2 && soiltyp != 4
                           && soiltyp != 3)) {
        real satratio = smc / smcmax;
        const real thkice = 2.2f, thkw = 0.57f, thko = 2.0f,
                   thkqtz = 7.7f;
        real thks = pow(thkqtz, qz) * pow(thko, 1.0f - qz);
        real xunfroz = sh2o / smc;
        real xu = xunfroz * smcmax;
        real thksat = pow(thks, 1.0f - smcmax)
                      * pow(thkice, smcmax - xu) * pow(thkw, xu);
        real gammd = (1.0f - smcmax) * 2700.0f;
        real thkdry = (0.135f * gammd + 64.7f)
                      / (2700.0f - 0.947f * gammd);
        real akei = satratio;
        real akel;
        if (satratio > 0.1f) akel = log10(satratio) + 1.0f;
        else akel = 0.0f;
        real ake = ((smc - sh2o) * akei + sh2o * akel) / smc;
        return ake * (thksat - thkdry) + thkdry;
    }
    real psif = psisat * 100.0f * pow(smcmax / smc, bexp);
    real pf = log10(fabs(psif));
    if (pf <= 5.1f) return 420.0f * exp(-(pf + 2.7f));
    return 0.1744f;
}

// --------------------------------------------------------------- SNOWZ0
 static real noah_snowz0(real sncovr, real z0brd, real snowh)
{
    const real z0s = 0.001f;
    real burial = 7.0f * z0brd - snowh;
    real z0eff;
    if (burial <= 0.0007f) z0eff = z0s;
    else z0eff = __fdiv_rn(burial, 7.0f);
    return (1.0f - sncovr) * z0brd + sncovr * z0eff;
}

// --------------------------------------------------------------- PENMAN
 static void noah_penman(real sfctmp, real sfcprs, real ch,
                                   real t2v, real th2, real prcp,
                                   real fdown, real ssoil, real q2,
                                   real q2sat, real dqsdt2, bool snowng,
                                   bool frzgra, real emissi_in,
                                   real sncovr, thread real& etp, thread real& rch,
                                   thread real& rr, thread real& epsca, thread real& t24,
                                   thread real& flx2)
{
    real emissi = emissi_in;
    real elcp1 = (1.0f - sncovr) * NELCP
                 + __fdiv_rn(sncovr * NELCP * NLSUBS, NLSUBC);
    real lvs = (1.0f - sncovr) * NLSUBC + sncovr * NLSUBS;
    flx2 = 0.0f;
    real delta = elcp1 * dqsdt2;
    t24 = sfctmp * sfctmp * sfctmp * sfctmp;
    rr = emissi * t24 * 6.48e-8f / (sfcprs * ch) + 1.0f;
    real rho = sfcprs / (NRD * t2v);
    rch = rho * NCP_PEN * ch;
    if (!snowng) {
        if (prcp > 0.0f) rr = rr + NCPH2O * prcp / rch;
    } else {
        rr = rr + NCPICE * prcp / rch;
    }
    real fnet = fdown - emissi * NSIGMA * t24 - ssoil;
    if (frzgra) {
        flx2 = -NLSUBF * prcp;
        fnet = fnet - flx2;
    }
    real rad = fnet / rch + th2 - sfctmp;
    real a = elcp1 * (q2sat - q2);
    epsca = (a * rr + rad * delta) / (delta + rr);
    etp = epsca * rch / lvs;           // AOASIS = 1 without the UCM
}

// --------------------------------------------------------------- CANRES
 static void noah_canres(real solar, real ch, real sfctmp,
                                   real q2, thread const real* sh2o,
                                   thread const real* zsoil, real smcwlt,
                                   real smcref, real rsmin, int nroot,
                                   real q2sat, real dqsdt2, real topt,
                                   real rsmax, real rgl, real hs,
                                   real xlai, real sfcprs, real emissi,
                                   thread real& rc, thread real& pc)
{
    const real slv = 2.501000e6f;
    real ff = 0.55f * 2.0f * solar / (rgl * xlai);
    real rcs = (ff + rsmin / rsmax) / (1.0f + ff);
    rcs = fmax(rcs, 0.0001f);
    real rct = 1.0f - 0.0016f * pow(topt - sfctmp, 2.0f);
    rct = fmax(rct, 0.0001f);
    real rcq = 1.0f / (1.0f + hs * (q2sat - q2));
    rcq = fmax(rcq, 0.01f);
    real rcsoil = 0.0f;
    real gx = (sh2o[0] - smcwlt) / (smcref - smcwlt);
    gx = fmin(fmax(gx, 0.0f), 1.0f);
    real part[NSOIL];
    part[0] = (zsoil[0] / zsoil[nroot - 1]) * gx;
    for (int k = 1; k < nroot; ++k) {
        gx = (sh2o[k] - smcwlt) / (smcref - smcwlt);
        gx = fmin(fmax(gx, 0.0f), 1.0f);
        part[k] = ((zsoil[k] - zsoil[k - 1]) / zsoil[nroot - 1]) * gx;
    }
    for (int k = 0; k < nroot; ++k) rcsoil = rcsoil + part[k];
    rcsoil = fmax(rcsoil, 0.0001f);
    rc = rsmin / (xlai * rcs * rct * rcq * rcsoil);
    real rr2 = (__fdiv_rn(4.0f * emissi * NSIGMA * NRD, CP))
               * pow(sfctmp, 4.0f) / (sfcprs * ch) + 1.0f;
    real delta = (slv / CP) * dqsdt2;
    pc = (rr2 + delta) / (rr2 * (1.0f + rc * ch) + delta);
}

// ---------------------------------------------------------------- DEVAP
 static real noah_devap(real etp1, real smc, real shdfac,
                                  real smcmax, real smcdry, real fxexp)
{
    real sratio = (smc - smcdry) / (smcmax - smcdry);
    real fx;
    if (sratio > 0.0f) {
        fx = pow(sratio, fxexp);
        fx = fmax(fmin(fx, 1.0f), 0.0f);
    } else {
        fx = 0.0f;
    }
    return fx * (1.0f - shdfac) * etp1;
}

// --------------------------------------------------------------- TRANSP
 static void noah_transp(thread real* et, real etp1,
                                   thread const real* sh2o, real cmc,
                                   real shdfac, real smcwlt,
                                   real cmcmax, real pc, real cfactr,
                                   real smcref, int nroot,
                                   thread const real* rtdis)
{
    for (int k = 0; k < NSOIL; ++k) et[k] = 0.0f;
    real etp1a;
    if (cmc != 0.0f)
        etp1a = shdfac * pc * etp1
                * (1.0f - pow(cmc / cmcmax, cfactr));
    else
        etp1a = shdfac * pc * etp1;
    real gx[NSOIL];
    real sgx = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        gx[i] = (sh2o[i] - smcwlt) / (smcref - smcwlt);
        gx[i] = fmax(fmin(gx[i], 1.0f), 0.0f);
        sgx = sgx + gx[i];
    }
    sgx = sgx / (real)nroot;
    real denom = 0.0f;
    for (int i = 0; i < nroot; ++i) {
        real rtx = rtdis[i] + gx[i] - sgx;
        gx[i] = gx[i] * fmax(rtx, 0.0f);
        denom = denom + gx[i];
    }
    if (denom <= 0.0f) denom = 1.0f;
    for (int i = 0; i < nroot; ++i) et[i] = etp1a * gx[i] / denom;
}

// ---------------------------------------------------------------- EVAPO
 static void noah_evapo(thread real& eta1, thread const real* smc, real cmc,
                                  real etp1, real dt, thread const real* sh2o,
                                  real smcmax, real pc, real smcwlt,
                                  real smcref, real shdfac, real cmcmax,
                                  real smcdry, real cfactr, int nroot,
                                  thread const real* rtdis, real fxexp,
                                  thread real& edir, thread real& ec, thread real* et,
                                  thread real& ett)
{
    edir = 0.0f;
    ec = 0.0f;
    ett = 0.0f;
    for (int k = 0; k < NSOIL; ++k) et[k] = 0.0f;
    if (etp1 > 0.0f) {
        if (shdfac < 1.0f)
            edir = noah_devap(etp1, smc[0], shdfac, smcmax, smcdry,
                              fxexp);
        if (shdfac > 0.0f) {
            noah_transp(et, etp1, sh2o, cmc, shdfac, smcwlt, cmcmax,
                        pc, cfactr, smcref, nroot, rtdis);
            for (int k = 0; k < NSOIL; ++k) ett = ett + et[k];
            if (cmc > 0.0f)
                ec = shdfac * pow(cmc / cmcmax, cfactr) * etp1;
            else
                ec = 0.0f;
            real cmc2ms = cmc / dt;
            ec = fmin(cmc2ms, ec);
        }
    }
    eta1 = edir + ett + ec;
}

// -------------------------------------------------------------- FAC2MIT
 static real noah_fac2mit(real smcmax)
{
    real flimit = 0.90f;
    if (smcmax == 0.395f) flimit = 0.59f;
    else if (smcmax == 0.434f || smcmax == 0.404f) flimit = 0.85f;
    else if (smcmax == 0.465f || smcmax == 0.406f) flimit = 0.86f;
    else if (smcmax == 0.476f || smcmax == 0.439f) flimit = 0.74f;
    else if (smcmax == 0.200f || smcmax == 0.464f) flimit = 0.80f;
    return flimit;
}

// --------------------------------------------------------------- WDFCND
 static void noah_wdfcnd(thread real& wdf, thread real& wcnd, real smc,
                                   real smcmax, real bexp, real dksat,
                                   real dwsat, real sicemax)
{
    real factr1 = 0.05f / smcmax;
    real factr2 = smc / smcmax;
    factr1 = fmin(factr1, factr2);
    real expon = bexp + 2.0f;
    wdf = dwsat * pow(factr2, expon);
    if (sicemax > 0.0f) {
        real vkwgt = 1.0f / (1.0f + pow(500.0f * sicemax, 3.0f));
        wdf = vkwgt * wdf + (1.0f - vkwgt) * dwsat
              * pow(factr1, expon);
    }
    expon = 2.0f * bexp + 3.0f;
    wcnd = dksat * pow(factr2, expon);
}

// ------------------------------------------------------------------ SRT
 static void noah_srt(thread real* rhstt, real edir, thread const real* et,
                                thread const real* sh2o, thread const real* sh2oa,
                                real pcpdrp, thread const real* zsoil,
                                real dwsat, real dksat, real smcmax,
                                real bexp, real dt, real smcwlt,
                                real slope, real kdt, real frzx,
                                thread const real* sice, thread real* ai, thread real* bi,
                                thread real* ci, thread real& runoff1, thread real& runoff2)
{
    const int cvfrz = 3;
    real dmax[NSOIL];
    real sicemax = 0.0f;
    for (int ks = 0; ks < NSOIL; ++ks)
        if (sice[ks] > sicemax) sicemax = sice[ks];
    real pddum = pcpdrp;
    runoff1 = 0.0f;
    runoff2 = 0.0f;
    if (pcpdrp != 0.0f) {
        real dt1 = __fdiv_rn(dt, 86400.0f);
        real smcav = smcmax - smcwlt;
        dmax[0] = -zsoil[0] * smcav;
        real dice = -zsoil[0] * sice[0];
        dmax[0] = dmax[0] * (1.0f - (sh2oa[0] + sice[0] - smcwlt)
                             / smcav);
        real dd = dmax[0];
        for (int ks = 1; ks < NSOIL; ++ks) {
            dice = dice + (zsoil[ks - 1] - zsoil[ks]) * sice[ks];
            dmax[ks] = (zsoil[ks - 1] - zsoil[ks]) * smcav;
            dmax[ks] = dmax[ks] * (1.0f - (sh2oa[ks] + sice[ks]
                                           - smcwlt) / smcav);
            dd = dd + dmax[ks];
        }
        real val = 1.0f - exp(-kdt * dt1);
        real ddt = dd * val;
        real px = pcpdrp * dt;
        if (px < 0.0f) px = 0.0f;
        real infmax = (px * (ddt / (px + ddt))) / dt;
        real fcr = 1.0f;
        if (dice > 1.0e-2f) {
            real acrt = (real)cvfrz * frzx / dice;
            real ssum = 1.0f;
            int ialp1 = cvfrz - 1;
            for (int j = 1; j <= ialp1; ++j) {
                int k = 1;
                for (int jj = j + 1; jj <= ialp1; ++jj) k = k * jj;
                ssum = ssum + pow(acrt, (real)(cvfrz - j))
                       / (real)k;
            }
            fcr = 1.0f - exp(-acrt) * ssum;
        }
        infmax = infmax * fcr;
        real wdf0, wcnd0;
        noah_wdfcnd(wdf0, wcnd0, sh2oa[0], smcmax, bexp, dksat, dwsat,
                    sicemax);
        infmax = fmax(infmax, wcnd0);
        infmax = fmin(infmax, px / dt);
        if (pcpdrp > infmax) {
            runoff1 = pcpdrp - infmax;
            pddum = infmax;
        }
    }
    real wdf, wcnd;
    noah_wdfcnd(wdf, wcnd, sh2oa[0], smcmax, bexp, dksat, dwsat,
                sicemax);
    real ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    bi[0] = wdf * ddz / (-zsoil[0]);
    ci[0] = -bi[0];
    real dsmdz = (sh2o[0] - sh2o[1]) / (-0.5f * zsoil[1]);
    rhstt[0] = (wdf * dsmdz + wcnd - pddum + edir + et[0]) / zsoil[0];
    real ddz2 = 0.0f;
    for (int k = 1; k < NSOIL; ++k) {
        real denom2 = zsoil[k - 1] - zsoil[k];
        real wdf2, wcnd2, dsmdz2, slopx;
        if (k != NSOIL - 1) {
            slopx = 1.0f;
            noah_wdfcnd(wdf2, wcnd2, sh2oa[k], smcmax, bexp, dksat,
                        dwsat, sicemax);
            real denom = zsoil[k - 1] - zsoil[k + 1];
            dsmdz2 = (sh2o[k] - sh2o[k + 1]) / (denom * 0.5f);
            ddz2 = 2.0f / denom;
            ci[k] = -wdf2 * ddz2 / denom2;
        } else {
            slopx = slope;
            noah_wdfcnd(wdf2, wcnd2, sh2oa[NSOIL - 1], smcmax, bexp,
                        dksat, dwsat, sicemax);
            dsmdz2 = 0.0f;
            ci[k] = 0.0f;
        }
        real numer = wdf2 * dsmdz2 + slopx * wcnd2 - wdf * dsmdz
                     - wcnd + et[k];
        rhstt[k] = numer / (-denom2);
        ai[k] = -wdf * ddz / denom2;
        bi[k] = -(ai[k] + ci[k]);
        if (k == NSOIL - 1) runoff2 = slopx * wcnd2;
        if (k != NSOIL - 1) {
            wdf = wdf2;
            wcnd = wcnd2;
            dsmdz = dsmdz2;
            ddz = ddz2;
        }
    }
}

// --------------------------------------------------------------- ROSR12
 static void noah_rosr12(thread real* p, thread const real* a,
                                   thread const real* b, thread const real* c_in,
                                   thread const real* d)
{
    real c_[NSOIL], delta[NSOIL];
    for (int k = 0; k < NSOIL; ++k) c_[k] = c_in[k];
    c_[NSOIL - 1] = 0.0f;
    p[0] = -c_[0] / b[0];
    delta[0] = d[0] / b[0];
    for (int k = 1; k < NSOIL; ++k) {
        p[k] = -c_[k] * (1.0f / (b[k] + a[k] * p[k - 1]));
        delta[k] = (d[k] - a[k] * delta[k - 1])
                   * (1.0f / (b[k] + a[k] * p[k - 1]));
    }
    p[NSOIL - 1] = delta[NSOIL - 1];
    for (int k = 1; k < NSOIL; ++k) {
        int kk = NSOIL - k - 1;
        p[kk] = p[kk] * p[kk + 1] + delta[kk];
    }
}

// ---------------------------------------------------------------- SSTEP
 static void noah_sstep(thread real* sh2oout, thread const real* sh2oin,
                                  thread real& cmc, thread const real* rhstt_in,
                                  real rhsct, real dt, real smcmax,
                                  real cmcmax, thread const real* zsoil,
                                  thread real* smc, thread const real* sice,
                                  thread const real* ai_in, thread const real* bi_in,
                                  thread const real* ci_in, thread real& runoff3,
                                  bool update_cmc)
{
    real rhstt[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL], p[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        rhstt[k] = rhstt_in[k] * dt;
        ai[k] = ai_in[k] * dt;
        bi[k] = 1.0f + bi_in[k] * dt;
        ci[k] = ci_in[k] * dt;
    }
    noah_rosr12(p, ai, bi, ci, rhstt);
    real wplus = 0.0f;
    real ddz = -zsoil[0];
    for (int k = 0; k < NSOIL; ++k) {
        if (k != 0) ddz = zsoil[k - 1] - zsoil[k];
        sh2oout[k] = sh2oin[k] + p[k] + wplus / ddz;
        real stot = sh2oout[k] + sice[k];
        if (stot > smcmax) {
            if (k == 0) ddz = -zsoil[0];
            else ddz = -zsoil[k] + zsoil[k - 1];
            wplus = (stot - smcmax) * ddz;
        } else {
            wplus = 0.0f;
        }
        smc[k] = fmax(fmin(stot, smcmax), 0.02f);
        sh2oout[k] = fmax(smc[k] - sice[k], 0.0f);
    }
    runoff3 = wplus;
    if (update_cmc) {
        cmc = cmc + dt * rhsct;
        if (cmc < 1.0e-20f) cmc = 0.0f;
        cmc = fmin(cmc, cmcmax);
    }
}

// ---------------------------------------------------------------- SMFLX
 static void noah_smflx(thread real* smc, thread real& cmc, real dt,
                                  real prcp1, thread const real* zsoil,
                                  thread real* sh2o, real slope, real kdt,
                                  real frzfact, real smcmax, real bexp,
                                  real smcwlt, real dksat, real dwsat,
                                  real shdfac, real cmcmax, real edir,
                                  real ec, thread const real* et,
                                  thread real& runoff1, thread real& runoff2,
                                  thread real& runoff3, thread real& drip)
{
    real rhsct = shdfac * prcp1 - ec;
    drip = 0.0f;
    real trhsct = dt * rhsct;
    real excess = cmc + trhsct;
    if (excess > cmcmax) drip = excess - cmcmax;
    real pcpdrp = (1.0f - shdfac) * prcp1 + drip / dt;
    real sice[NSOIL];
    for (int i = 0; i < NSOIL; ++i) sice[i] = smc[i] - sh2o[i];
    real fac2 = 0.0f;
    for (int i = 0; i < NSOIL; ++i)
        fac2 = fmax(fac2, sh2o[i] / smcmax);
    real flimit = noah_fac2mit(smcmax);
    real rhstt[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL];
    if ((pcpdrp * dt) > (0.0001f * 1000.0f * (-zsoil[0]) * smcmax)
        || (fac2 > flimit)) {
        real sh2ofg[NSOIL], sh2oa[NSOIL], sh2onew[NSOIL];
        real dummy = 0.0f;
        noah_srt(rhstt, edir, et, sh2o, sh2o, pcpdrp, zsoil, dwsat,
                 dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzfact,
                 sice, ai, bi, ci, runoff1, runoff2);
        noah_sstep(sh2ofg, sh2o, dummy, rhstt, rhsct, dt, smcmax,
                   cmcmax, zsoil, smc, sice, ai, bi, ci, runoff3,
                   false);
        for (int k = 0; k < NSOIL; ++k)
            sh2oa[k] = (sh2o[k] + sh2ofg[k]) * 0.5f;
        noah_srt(rhstt, edir, et, sh2o, sh2oa, pcpdrp, zsoil, dwsat,
                 dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzfact,
                 sice, ai, bi, ci, runoff1, runoff2);
        noah_sstep(sh2onew, sh2o, cmc, rhstt, rhsct, dt, smcmax,
                   cmcmax, zsoil, smc, sice, ai, bi, ci, runoff3,
                   true);
        for (int k = 0; k < NSOIL; ++k) sh2o[k] = sh2onew[k];
    } else {
        real sh2onew[NSOIL];
        noah_srt(rhstt, edir, et, sh2o, sh2o, pcpdrp, zsoil, dwsat,
                 dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzfact,
                 sice, ai, bi, ci, runoff1, runoff2);
        noah_sstep(sh2onew, sh2o, cmc, rhstt, rhsct, dt, smcmax,
                   cmcmax, zsoil, smc, sice, ai, bi, ci, runoff3,
                   true);
        for (int k = 0; k < NSOIL; ++k) sh2o[k] = sh2onew[k];
    }
}

// ------------------------------------------------------------------ TBND
 static real noah_tbnd(real tu, real tb, thread const real* zsoil,
                                 real zbot, int k)
{
    real zup, zb;
    if (k == 0) zup = 0.0f;
    else zup = zsoil[k - 1];
    if (k == NSOIL - 1) zb = 2.0f * zbot - zsoil[k];
    else zb = zsoil[k + 1];
    return tu + (tb - tu) * (zup - zsoil[k]) / (zup - zb);
}

// ---------------------------------------------------------------- TMPAVG
 static real noah_tmpavg(real tup, real tm, real tdn,
                                   thread const real* zsoil, int k)
{
    const real t0 = 2.7315e2f;
    real dz;
    if (k == 0) dz = -zsoil[0];
    else dz = zsoil[k - 1] - zsoil[k];
    real dzh = dz * 0.5f;
    if (tup < t0) {
        if (tm < t0) {
            if (tdn < t0) {
                return (tup + 2.0f * tm + tdn) / 4.0f;
            }
            real x0 = (t0 - tm) * dzh / (tdn - tm);
            return 0.5f * (tup * dzh + tm * (dzh + x0)
                           + t0 * (2.0f * dzh - x0)) / dz;
        }
        if (tdn < t0) {
            real xup = (t0 - tup) * dzh / (tm - tup);
            real xdn = dzh - (t0 - tm) * dzh / (tdn - tm);
            return 0.5f * (tup * xup + t0 * (2.0f * dz - xup - xdn)
                           + tdn * xdn) / dz;
        }
        real xup = (t0 - tup) * dzh / (tm - tup);
        return 0.5f * (tup * xup + t0 * (2.0f * dz - xup)) / dz;
    }
    if (tm < t0) {
        if (tdn < t0) {
            real xup = dzh - (t0 - tup) * dzh / (tm - tup);
            return 0.5f * (t0 * (dz - xup) + tm * (dzh + xup)
                           + tdn * dzh) / dz;
        }
        real xup = dzh - (t0 - tup) * dzh / (tm - tup);
        real xdn = (t0 - tm) * dzh / (tdn - tm);
        return 0.5f * (t0 * (2.0f * dz - xup - xdn)
                       + tm * (xup + xdn)) / dz;
    }
    if (tdn < t0) {
        real xdn = dzh - (t0 - tm) * dzh / (tdn - tm);
        return (t0 * (dz - xdn) + 0.5f * (t0 + tdn) * xdn) / dz;
    }
    return (tup + 2.0f * tm + tdn) / 4.0f;
}

// ---------------------------------------------------------------- SNKSRC
 static real noah_snksrc(real qtot, real tavg, real smc,
                                   thread real& sh2o, thread const real* zsoil,
                                   real smcmax, real psisat, real bexp,
                                   real dt, int k)
{
    const real dh2o = 1.0000e3f, hlice = 3.3350e5f;
    real dz;
    if (k == 0) dz = -zsoil[0];
    else dz = zsoil[k - 1] - zsoil[k];
    real freew = noah_frh2o(tavg, smc, sh2o, smcmax, bexp, psisat);
    real xh2o = sh2o + qtot * dt / (dh2o * hlice * dz);
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
    real tsnsr = -dh2o * hlice * dz * (xh2o - sh2o) / dt;
    sh2o = xh2o;
    return tsnsr;
}

// ------------------------------------------------------------------ HRT
 static void noah_hrt(thread real* rhsts, thread const real* stc,
                                thread const real* smc, real smcmax,
                                thread const real* zsoil, real yy, real zz1,
                                real tbot, real zbot, real psisat,
                                thread real* sh2o, real dt, real bexp,
                                int soiltyp, int opt_thcnd, real df1,
                                real quartz, real csoil, int vegtyp,
                                int isurban, thread real* ai, thread real* bi,
                                thread real* ci)
{
    const real t0 = 273.15f, cair = 1004.0f, cice = 2.106e6f,
               ch2o = 4.2e6f;
    real csoil_loc;
    if (vegtyp == isurban) csoil_loc = 3.0e6f;
    else csoil_loc = csoil;
    real hcpct = sh2o[0] * ch2o + (1.0f - smcmax) * csoil_loc
                 + (smcmax - smc[0]) * cair
                 + (smc[0] - sh2o[0]) * cice;
    real ddz = 1.0f / (-0.5f * zsoil[1]);
    ai[0] = 0.0f;
    ci[0] = (df1 * ddz) / (zsoil[0] * hcpct);
    bi[0] = -ci[0] + df1 / (0.5f * zsoil[0] * zsoil[0] * hcpct * zz1);
    real dtsdz = (stc[0] - stc[1]) / (-0.5f * zsoil[1]);
    real ssoil = df1 * (stc[0] - yy) / (0.5f * zsoil[0] * zz1);
    real denom = zsoil[0] * hcpct;
    rhsts[0] = (df1 * dtsdz - ssoil) / denom;
    real qtot = -1.0f * rhsts[0] * denom;
    real sice = smc[0] - sh2o[0];
    real tsurf = (yy + (zz1 - 1.0f) * stc[0]) / zz1;
    real tbk = noah_tbnd(stc[0], stc[1], zsoil, zbot, 0);
    if (sice > 0.0f || stc[0] < t0 || tsurf < t0 || tbk < t0) {
        real tavg = noah_tmpavg(tsurf, stc[0], tbk, zsoil, 0);
        real tsnsr = noah_snksrc(qtot, tavg, smc[0], sh2o[0], zsoil,
                                 smcmax, psisat, bexp, dt, 0);
        rhsts[0] = rhsts[0] - tsnsr / denom;
    }
    real ddz2 = 0.0f;
    real df1k = df1;
    real dtsdz2;
    for (int k = 1; k < NSOIL; ++k) {
        hcpct = sh2o[k] * ch2o + (1.0f - smcmax) * csoil_loc
                + (smcmax - smc[k]) * cair
                + (smc[k] - sh2o[k]) * cice;
        real df1n, tbk1;
        if (k != NSOIL - 1) {
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
        real tavg = noah_tmpavg(tbk, stc[k], tbk1, zsoil, k);
        if (sice > 0.0f || stc[k] < t0 || tbk < t0 || tbk1 < t0) {
            real tsnsr = noah_snksrc(qtot, tavg, smc[k], sh2o[k],
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

// ---------------------------------------------------------------- HSTEP
 static void noah_hstep(thread real* stcout, thread const real* stcin,
                                  thread const real* rhsts_in, real dt,
                                  thread const real* ai_in, thread const real* bi_in,
                                  thread const real* ci_in)
{
    real rhsts[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL], p[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        rhsts[k] = rhsts_in[k] * dt;
        ai[k] = ai_in[k] * dt;
        bi[k] = 1.0f + bi_in[k] * dt;
        ci[k] = ci_in[k] * dt;
    }
    noah_rosr12(p, ai, bi, ci, rhsts);
    for (int k = 0; k < NSOIL; ++k) stcout[k] = stcin[k] + p[k];
}

// ---------------------------------------------------------------- SHFLX
 static void noah_shflx(thread real* stc, thread const real* smc,
                                  real smcmax, thread real& t1, real dt,
                                  real yy, real zz1, thread const real* zsoil,
                                  real tbot, real zbot, real psisat,
                                  thread real* sh2o, real bexp, real df1,
                                  real quartz, real csoil, int vegtyp,
                                  int isurban, int soiltyp,
                                  int opt_thcnd, thread real& ssoil)
{
    real rhsts[NSOIL], ai[NSOIL], bi[NSOIL], ci[NSOIL], stcf[NSOIL];
    noah_hrt(rhsts, stc, smc, smcmax, zsoil, yy, zz1, tbot, zbot,
             psisat, sh2o, dt, bexp, soiltyp, opt_thcnd, df1, quartz,
             csoil, vegtyp, isurban, ai, bi, ci);
    noah_hstep(stcf, stc, rhsts, dt, ai, bi, ci);
    for (int i = 0; i < NSOIL; ++i) stc[i] = stcf[i];
    t1 = (yy + (zz1 - 1.0f) * stc[0]) / zz1;
    ssoil = df1 * (stc[0] - t1) / (0.5f * zsoil[0]);
}

// -------------------------------------------------------------- SNOWPACK
 static void noah_snowpack(real esd, real dtsec, thread real& snowh,
                                     thread real& sndens, real tsnow,
                                     real tsoil)
{
    const real c1k = 0.01f, c2k = 21.0f;
    real snowhc = snowh * 100.0f;
    real esdc = esd * 100.0f;
    real dthr = __fdiv_rn(dtsec, 3600.0f);
    real tsnowc = tsnow - 273.15f;
    real tsoilc = tsoil - 273.15f;
    real tavgc = 0.5f * (tsnowc + tsoilc);
    real esdcx;
    if (esdc > 1.0e-2f) esdcx = esdc;
    else esdcx = 1.0e-2f;
    real bfac = dthr * c1k * exp(0.08f * tavgc - c2k * sndens);
    const int ipol = 4;
    real pexp = 0.0f;
    for (int j = ipol; j >= 1; --j)
        pexp = __fdiv_rn((1.0f + pexp) * bfac * esdcx, (real)(j + 1));
    pexp = pexp + 1.0f;
    real dsx = sndens * pexp;
    if (dsx > 0.40f) dsx = 0.40f;
    if (dsx < 0.05f) dsx = 0.05f;
    sndens = dsx;
    if (tsnowc >= 0.0f) {
        real dw = __fdiv_rn(0.13f * dthr, 24.0f);
        sndens = sndens * (1.0f - dw) + dw;
        if (sndens >= 0.40f) sndens = 0.40f;
    }
    snowhc = esdc / sndens;
    snowh = snowhc * 0.01f;
}

// ---------------------------------------------------------------- kernel
// Ordinary non-urban Noah column. Urban routing is explicitly refused by the
// Python launcher and is intentionally absent from this entry point.
#include "../generated/noah_mosaic_pack_bindings.metalh"
#include "../generated/noah_mosaic_unpack_bindings.metalh"
#define NOAH_MOSAIC_UCM 0
#define FADD(a,b) ((a)+(b))
#define FSUB(a,b) ((a)-(b))
#define FMUL(a,b) ((a)*(b))
#define FDIV(a,b) ((a)/(b))
#define __uint_as_float(x) as_type<float>(x)
static inline real gfk_log(real x) { return log(x); }
static inline real gfk_exp(real x) { return exp(x); }
static inline real gfk_pow(real x, real y) { return pow(x,y); }
struct MosaicState {
#if NOAH_MOSAIC_UCM
 bool urban_tile; int natural; real rural_t1, rural_q1, rural_initial;
#endif
 int ivgtyp,isltyp;
 real psfc;
 real sfcprs;
 real sfctmp;
 real qv1;
 real qgh;
 real dz8w1;
 real glw;
 real swdown;
 real rainbl;
 real sr;
 real chs;
 real cqs2;
 real chs2;
 real rib;
 real vegfra;
 real shdmin;
 real shdmax;
 real tmn;
 real xland;
 real xice;
 real snoalb;
 real embck;
 real tsk;
 real hfx;
 real qfx;
 real lh;
 real grdflx;
 real qsfc;
 real canwat;
 real snow;
 real snowc;
 real snowh;
 real albedo;
 real albbck;
 real emiss;
 real znt;
 real z0;
 real snotime;
 real lai;
 real smstav;
 real smstot;
 real sfcrunoff;
 real udrunoff;
 real acsnow;
 real acsnom;
 real snopcx;
 real potevp;
 real noahres;
 real chklowq;
 real smois[4];
 real tslb[4];
 real sh2o[4];
 real smcrel[4];
};
// module_sf_noahlsm_glacial_only.F:565-707
static inline void mosaic_hrtice(thread real* rhs,thread const real* stc,real tbot,
 thread const real* zsoil,real yy,real zz1,real df1,thread real* ai,thread real* bi,thread real* ci) {
 real hcpct=FMUL(1.e6f, (FSUB(0.8194f, FMUL(__uint_as_float(0x3d860aa6u) /* gfortran15.2 fold_008 */, zsoil[0]))));
 real df1k=df1,ddz=FDIV(1.0f, (FMUL(-0.5f, zsoil[1])));
 ai[0]=0.0f;ci[0]=FDIV((FMUL(df1, ddz)), (FMUL(zsoil[0], hcpct)));
 bi[0]=FADD(-ci[0], FDIV(df1, (FMUL(FMUL(FMUL(FMUL(0.5f, zsoil[0]), zsoil[0]), hcpct), zz1))));
 real dtsdz=FDIV((FSUB(stc[0], stc[1])), (FMUL(-0.5f, zsoil[1])));
 real ssoil=FDIV(FMUL(df1, (FSUB(stc[0], yy))), (FMUL(FMUL(0.5f, zsoil[0]), zz1)));
 rhs[0]=FDIV((FSUB(FMUL(df1, dtsdz), ssoil)), (FMUL(zsoil[0], hcpct)));
 real ddz2=0.0f,df1n=df1;
 for(int k=1;k<4;++k) {
  real zmd=FMUL(0.5f, (FADD(zsoil[k], zsoil[k-1])));
  hcpct=FMUL(1.e6f, (FSUB(0.8194f, FMUL(0.1309f, zmd))));
  df1n=FSUB(0.32333f, (FMUL(0.10073f, zmd)));
  real dtsdz2,denom;
  if(k!=3) {
   denom=FMUL(0.5f, (FSUB(zsoil[k-1], zsoil[k+1])));
   dtsdz2=FDIV((FSUB(stc[k], stc[k+1])), denom);
   ddz2=FDIV(2.0f, (FSUB(zsoil[k-1], zsoil[k+1])));
   ci[k]=FDIV(FMUL(-df1n, ddz2), (FMUL((FSUB(zsoil[k-1], zsoil[k])), hcpct)));
  } else {
   dtsdz2=FDIV((FSUB(stc[k], tbot)), (FSUB(FMUL(0.5f, (FADD(zsoil[k-1], zsoil[k]))), (-25.0f))));
   ci[k]=0.0f;
  }
  denom=FMUL((FSUB(zsoil[k], zsoil[k-1])), hcpct);
  rhs[k]=FDIV((FSUB(FMUL(df1n, dtsdz2), FMUL(df1k, dtsdz))), denom);
  ai[k]=FDIV(FMUL(-df1k, ddz), (FMUL((FSUB(zsoil[k-1], zsoil[k])), hcpct)));
  bi[k]=-(FADD(ai[k], ci[k]));df1k=df1n;dtsdz=dtsdz2;ddz=ddz2;
 }
}
// module_sf_noahlsm_glacial_only.F:824-853, HSTEP:710-754 calls Noah ROSR12
static inline void mosaic_shflx_ice(thread real* stc,real dt,real yy,real zz1,
 thread const real* zsoil,real tbot,real df1) {
 real rhs[4],ai[4],bi[4],ci[4],stcf[4];
 mosaic_hrtice(rhs,stc,tbot,zsoil,yy,zz1,df1,ai,bi,ci);
 noah_hstep(stcf,stc,rhs,dt,ai,bi,ci);
 for(int k=0;k<4;++k)stc[k]=stcf[k];
}
// module_sf_noahdrv.F:3251-3729; module_sf_noahlsm_glacial_only.F:32-409
static inline void noah_mosaic_tile_glacial(thread MosaicState& s,
 device const real* vegtbl,device const real* soiltbl,device const real* genp,device const real* dzs,
 real dt,int lucats,int slcats,int isurban,int isice,real xice_threshold,
 int itimestep,int frpcpn,int usemonalb,int rdlai2d,int opt_thcnd)
{
 const size_t idx=0,plane=1;
 thread const int* ivgtyp=&s.ivgtyp; thread const int* isltyp=&s.isltyp;
 thread real* psfc_a=&s.psfc;
 thread real* sfcprs_a=&s.sfcprs;
 thread real* sfctmp_a=&s.sfctmp;
 thread real* qv1_a=&s.qv1;
 thread real* qgh_a=&s.qgh;
 thread real* dz8w1_a=&s.dz8w1;
 thread real* glw_a=&s.glw;
 thread real* swdown_a=&s.swdown;
 thread real* rainbl_a=&s.rainbl;
 thread real* sr_a=&s.sr;
 thread real* chs_a=&s.chs;
 thread real* cqs2_a=&s.cqs2;
 thread real* chs2_a=&s.chs2;
 thread real* rib_a=&s.rib;
 thread real* vegfra_a=&s.vegfra;
 thread real* shdmin_a=&s.shdmin;
 thread real* shdmax_a=&s.shdmax;
 thread real* tmn_a=&s.tmn;
 thread real* xland_a=&s.xland;
 thread real* xice_a=&s.xice;
 thread real* snoalb_a=&s.snoalb;
 thread real* embck_a=&s.embck;
 thread real* tsk_a=&s.tsk;
 thread real* hfx_a=&s.hfx;
 thread real* qfx_a=&s.qfx;
 thread real* lh_a=&s.lh;
 thread real* grdflx_a=&s.grdflx;
 thread real* qsfc_a=&s.qsfc;
 thread real* canwat_a=&s.canwat;
 thread real* snow_a=&s.snow;
 thread real* snowc_a=&s.snowc;
 thread real* snowh_a=&s.snowh;
 thread real* albedo_a=&s.albedo;
 thread real* albbck_a=&s.albbck;
 thread real* emiss_a=&s.emiss;
 thread real* znt_a=&s.znt;
 thread real* z0_a=&s.z0;
 thread real* snotime_a=&s.snotime;
 thread real* lai_a=&s.lai;
 thread real* smstav_a=&s.smstav;
 thread real* smstot_a=&s.smstot;
 thread real* sfcrunoff_a=&s.sfcrunoff;
 thread real* udrunoff_a=&s.udrunoff;
 thread real* acsnow_a=&s.acsnow;
 thread real* acsnom_a=&s.acsnom;
 thread real* snopcx_a=&s.snopcx;
 thread real* potevp_a=&s.potevp;
 thread real* noahres_a=&s.noahres;
 thread real* chklowq_a=&s.chklowq;
 thread real* smois_a=s.smois;
 thread real* tslb_a=s.tslb;
 thread real* sh2o_a=s.sh2o;
 thread real* smcrel_a=s.smcrel;
    // ================= driver (module_sf_noahdrv.F lsm) prep =========
    // module_sf_noahdrv.F:809-814 writes CHKLOWQ for EVERY column, before the
    // XLAND land/sea branch in mosaic prep -- so open water, sea ice and land ice all
    // leave the driver with CHKLOWQ = 1.  This kernel used to return from the
    // three skip paths below without writing it, leaving whatever the caller
    // had in the array; the WRF oracle's water and sea-ice rows found it (0.0
    // against 1.0).  The `myj` arm that can set 0 instead is not ported --
    // launch_noah has no myj argument -- so the value is unconditionally 1.
    chklowq_a[idx] = 1.0f;
    if ((FSUB(xland_a[idx], 1.5f)) >= 0.0f) return;        // open water
    if (xice_a[idx] >= xice_threshold) {              // sea ice
        for (int k = 0; k < NSOIL; ++k) sh2o_a[k * plane + idx] = 1.0f;
        lai_a[idx] = 0.01f;
        return;
    }
    int vegtyp = ivgtyp[idx];
    int soiltyp = isltyp[idx];


    const real tresh = 0.95f, a2 = 17.67f, a3 = 273.15f, a4 = 29.65f;
    const real a23m4 = __uint_as_float(0x45867529u) /* gfortran15.2 fold_009 */;
    real psfc = psfc_a[idx];
    real sfcprs = sfcprs_a[idx];
    real q2k = FDIV(qv1_a[idx], (FADD(1.0f, qv1_a[idx])));
    real q2sat = FDIV(qgh_a[idx], (FADD(1.0f, qgh_a[idx])));
    real sfctmp = sfctmp_a[idx];
    real apes = gfk_pow(FDIV(1.0e5f, psfc), RCP);             // CAPA = R_d/CP
    real apelm = gfk_pow(FDIV(1.0e5f, sfcprs), RCP);
    real sfcth2 = FMUL(sfctmp, apelm);
    real th2 = FDIV(sfcth2, apes);
    real emissi = emiss_a[idx];
    real lwdn = FMUL(glw_a[idx], emissi);
    real soldn = swdown_a[idx];
    real solnet = FMUL(soldn, (FSUB(1.0f, albedo_a[idx])));
    real prcp = FDIV(rainbl_a[idx], dt);
    real shdfac = FDIV(vegfra_a[idx], 100.0f);
    if (vegtyp == 25 || vegtyp == 26 || vegtyp == 27) shdfac = 0.0f;
    real t1 = tsk_a[idx];
    real chk = chs_a[idx];
    real shmin = FDIV(shdmin_a[idx], 100.0f);
    real shmax = FDIV(shdmax_a[idx], 100.0f);
    real sneqv = FMUL(snow_a[idx], 0.001f);
    real snowhk = snowh_a[idx];
    real sncovr = snowc_a[idx];
    real ffrozp;
    if (frpcpn) ffrozp = sr_a[idx];
    else ffrozp = (sfctmp <= 273.15f) ? 1.0f : 0.0f;

    real dqsdt2 = FDIV(FMUL(q2sat, a23m4), (FMUL((FSUB(sfctmp, a4)), (FSUB(sfctmp, a4)))));
    if (snow_a[idx] > 0.0f) {
        real sfctsno = sfctmp;
        real e2sat = FMUL(611.2f, gfk_exp(FMUL(6174.0f, (FSUB(__uint_as_float(0x3b6fed42u) /* gfortran15.2 fold_011 */, FDIV(1.0f, sfctsno))))));
        real q2sati = FDIV(FMUL(0.622f, e2sat), (FSUB(sfcprs, e2sat)));
        q2sati = FDIV(q2sati, (FADD(1.0f, q2sati)));
        if (t1 > 273.14f) {
            q2sat = FADD(FMUL(q2sat, (FSUB(1.0f, snowc_a[idx]))), FMUL(q2sati, snowc_a[idx]));
            dqsdt2 = FADD(FMUL(dqsdt2, (FSUB(1.0f, snowc_a[idx]))), FMUL(FDIV(FMUL(q2sati, 6174.0f), (FMUL(sfctsno, sfctsno))), snowc_a[idx]));
        } else {
            q2sat = q2sati;
            dqsdt2 = FDIV(FMUL(q2sati, 6174.0f), (FMUL(sfctsno, sfctsno)));
        }
        if (t1 > 273.0f && snowc_a[idx] > 0.0f)
            dqsdt2 = FMUL(dqsdt2, (FSUB(1.0f, snowc_a[idx])));
    }
    real tbot = tmn_a[idx];
    if (soiltyp == 14 && xice_a[idx] == 0.0f) soiltyp = 7;
    real snoalb1 = snoalb_a[idx];
    real cmc = FDIV(canwat_a[idx], 1000.0f);
    real alb = albbck_a[idx];
    real z0brd = z0_a[idx];
    real embrd = embck_a[idx];
    real snotime1 = snotime_a[idx];
    real ribb = rib_a[idx];
    real smc[NSOIL], stc[NSOIL], swc[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        smc[k] = smois_a[k * plane + idx];
        stc[k] = tslb_a[k * plane + idx];
        swc[k] = sh2o_a[k * plane + idx];
    }
    if ((sneqv != 0.0f && snowhk == 0.0f) || (snowhk <= sneqv))
        snowhk = FMUL(5.0f, sneqv);
    real xlai = lai_a[idx];
    if (rdlai2d) {
        if (shdfac > 0.0f && xlai <= 0.0f) xlai = 0.01f;
    }

 // module_sf_noahlsm_glacial_only.F:208-406, SFLX_GLACIAL
 real zsoil[4];zsoil[0]=-dzs[0];
 for(int k=1;k<4;++k)zsoil[k]=FADD(-dzs[k], zsoil[k-1]);
 if(sneqv<0.10f){sneqv=0.10f;snowhk=0.50f;}
 real sndens=FDIV(sneqv, snowhk);
 real sncond=noah_csnow(sndens);
 bool snowng=false,frzgra=false;
 if(prcp>0.0f){if(ffrozp>0.5f)snowng=true;else if(t1<=NTFREEZ)frzgra=true;}
 real prcpf;
 if(snowng||frzgra){
  real sn_new=FMUL(FMUL(prcp, dt), 0.001f);sneqv=FADD(sneqv, sn_new);prcpf=0.0f;
  noah_snow_new(sfctmp,sn_new,snowhk,sndens);
  if(sncovr>0.99f){
   if(stc[0]<(__uint_as_float(0x43861333u) /* gfortran15.2 fold_012 */))sndens=0.2f;
   if(snowng && t1<273.0f && sfctmp<273.0f)sndens=0.2f;
  }
  sncond=noah_csnow(sndens);
 } else prcpf=prcp;
 sncovr=1.0f;
 real albedo;
 // glacial ALCALC:412-523 has the same aging expression, SNCOVR fixed at 1.
 noah_alcalc(alb,snoalb1,embrd,1.0f,dt,snowng,snotime1,genp[GEN_LVCOEF],albedo,emissi);
 real df1=sncond;
 real dsoil=-(FMUL(0.5f, zsoil[0]));
 real dtot=FADD(snowhk, dsoil);
 real frcsno=FDIV(snowhk, dtot),frcsoi=FDIV(dsoil, dtot);
 df1=FADD(FMUL(frcsno, sncond), FMUL(frcsoi, df1));
 if(dtot>FMUL(2.0f, dsoil))dtot=FMUL(2.0f, dsoil);
 real ssoil=FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dtot);
 // glacial SNOWZ0:1202-1226: complete cover, no ordinary-land blend.
 real burial=FSUB(FMUL(7.0f, z0brd), snowhk);
 real z0k=burial<=0.0007f?0.001f:FDIV(burial, 7.0f);
 real fdown=FADD(solnet, lwdn);
 real t2v=FMUL(sfctmp, (FADD(1.0f, FMUL(0.61f, q2k))));
 real rho=FDIV(sfcprs, (FMUL(NRD, t2v)));
 real rch=FMUL(FMUL(rho, 1004.6f), chk);
 real t24=FMUL(FMUL(FMUL(sfctmp, sfctmp), sfctmp), sfctmp);
 // glacial PENMAN:756-821 differs from ordinary PENMAN's ELCP1 choice.
 real elcp1,lvs;
 if(t1>273.15f){elcp1=NELCP;lvs=NLSUBC;}
 else {elcp1=__uint_as_float(0x4530031fu) /* gfortran15.2 fold_013 */;lvs=NLSUBS;}
 real delta=FMUL(elcp1, dqsdt2);
 real a=FMUL(elcp1, (FSUB(q2sat, q2k)));
 real rr=FADD(FDIV(FMUL(FMUL(emissi, t24), 6.48e-8f), (FMUL(sfcprs, chk))), 1.0f);
 if(!snowng){if(prcp>0.0f)rr=FADD(rr, FDIV(FMUL(NCPH2O, prcp), rch));}
 else rr=FADD(rr, FDIV(FMUL(NCPICE, prcp), rch));
 real flx2=frzgra?FMUL(-NLSUBF, prcp):0.0f;
 real fnet=FSUB(FSUB(FSUB(fdown, (FMUL(FMUL(emissi, NSIGMA), t24))), ssoil), flx2);
 real rad=FSUB(FADD(FDIV(fnet, rch), th2), sfctmp);
 real epsca=FDIV((FADD(FMUL(a, rr), FMUL(rad, delta))), (FADD(delta, rr)));
 real etp=FDIV(FMUL(epsca, rch), lvs);
 // glacial SNOPAC:856-1073
 real snomlt=0.0f,dew=0.0f,esnow=0.0f,esnow1=0.0f,esnow2=0.0f;
 real etanrg;
 if(etp<=0.0f){
  if(ribb>=0.1f && fdown>150.0f)
   etp=FDIV((FADD(FDIV(fmin(FMUL(etp, (FSUB(1.0f, ribb))),0.0f), 0.980f), FMUL(etp, (__uint_as_float(0xbca3d700u) /* gfortran15.2 fold_015 */)))), 0.980f);
  real etp1=FMUL(etp, 0.001f);dew=-etp1;esnow2=FMUL(etp1, dt);etanrg=FMUL(etp, NLSUBS);
 } else {esnow=etp;esnow1=FMUL(esnow, 0.001f);esnow2=FMUL(esnow1, dt);etanrg=FMUL(esnow, NLSUBS);}
 real flx1=0.0f;
 if(snowng)flx1=FMUL(FMUL(NCPICE, prcp), (FSUB(t1, sfctmp)));
 else if(prcp>0.0f)flx1=FMUL(FMUL(NCPH2O, prcp), (FSUB(t1, sfctmp)));
 dsoil=-(FMUL(0.5f, zsoil[0]));dtot=FADD(snowhk, dsoil);
 real denom=FADD(1.0f, FDIV(df1, (FMUL(FMUL(dtot, rr), rch))));
 real t12a=FDIV((FSUB(FSUB(FADD(FDIV((FSUB(FSUB(FSUB(fdown, flx1), flx2), FMUL(FMUL(emissi, NSIGMA), t24))), rch), th2), sfctmp), FDIV(etanrg, rch))), rr);
 real t12b=FDIV(FMUL(df1, stc[0]), (FMUL(FMUL(dtot, rr), rch)));
 real t12=FDIV((FADD(FADD(sfctmp, t12a), t12b)), denom);
 real flx3=0.0f,ex=0.0f;
 if(t12<=NTFREEZ){
  t1=t12;ssoil=FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dtot);
  sneqv=fmax(0.0f,FSUB(sneqv, esnow2));flx3=0.0f;ex=0.0f;snomlt=0.0f;
 } else {
  t1=NTFREEZ;if(dtot>FMUL(2.0f, dsoil))dtot=FMUL(2.0f, dsoil);
  ssoil=FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dtot);
  if(FSUB(sneqv, esnow2)<=1.e-6f){sneqv=0.0f;ex=0.0f;snomlt=0.0f;flx3=0.0f;}
  else {
   sneqv=FSUB(sneqv, esnow2);
   real seh=FMUL(rch, (FSUB(t1, th2)));
   real t14=FMUL((FMUL(t1, t1)), (FMUL(t1, t1)));
   flx3=FSUB(FSUB(FSUB(FSUB(FSUB(FSUB(fdown, flx1), flx2), FMUL(FMUL(emissi, NSIGMA), t14)), ssoil), seh), etanrg);
   if(flx3<=0.0f)flx3=0.0f;
   ex=FDIV(FMUL(flx3, 0.001f), NLSUBF);snomlt=FMUL(ex, dt);
   if(FSUB(sneqv, snomlt)>=1.e-6f)sneqv=FSUB(sneqv, snomlt);
   else {ex=FDIV(sneqv, dt);flx3=FMUL(FMUL(ex, 1000.0f), NLSUBF);snomlt=sneqv;sneqv=0.0f;}
  }
 }
 real zz1=1.0f;
 real yy=FSUB(stc[0], FDIV(FMUL(FMUL(FMUL(0.5f, ssoil), zsoil[0]), zz1), df1));
 mosaic_shflx_ice(stc,dt,yy,zz1,zsoil,tbot,df1);
 // glacial SNOWPACK:1076-1199 shares the ordinary routine's expression order.
 if(sneqv>=0.10f)noah_snowpack(sneqv,dt,snowhk,sndens,t1,yy);
 else {sneqv=0.10f;snowhk=0.50f;}
 real eta_kin=etp;
 real q1=FADD(q2k, FDIV(FMUL(eta_kin, CP), rch));
 real sheat=FMUL(FDIV(-(FMUL(FMUL(chk, CP), sfcprs)), (FMUL(NR_SHEAT, t2v))), (FSUB(th2, t1)));
 real esnow_e=FMUL(esnow, NLSUBS),etp_e=FMUL(etp, NLSUBS);
 real eta_e=etp_e>0.0f?esnow_e:etp_e;
 real ssoil_out=FMUL(-1.0f, ssoil);
 real runoff1=FDIV(snomlt, dt);
 // noahdrv:3605-3729 and 4062-4080. SOILW retains its previous tile value.
 s.lai=0.01f;s.canwat=FMUL(cmc, 1000.0f);s.snow=FMUL(sneqv, 1000.0f);s.snowh=snowhk;
 s.albedo=albedo;s.albbck=alb;s.z0=z0brd;s.emiss=emissi;s.znt=z0k;
 s.tsk=t1;s.hfx=sheat;s.potevp=FADD(s.potevp, FMUL(etp_e, (FDIV(dt, (__uint_as_float(0x4f1502f9u) /* gfortran15.2 fold_016 */)))));
 s.qfx=eta_kin;s.lh=eta_e;s.grdflx=ssoil_out;s.snowc=sncovr;
 s.chs2=s.cqs2;s.snotime=snotime1;
 if(q1>s.qsfc)s.cqs2=s.chs;
 s.qsfc=FDIV(q1, (FSUB(1.0f, q1)));
 for(int k=0;k<4;++k){s.smois[k]=1.0f;s.sh2o[k]=1.0f;s.smcrel[k]=1.0f;s.tslb[k]=stc[k];}
 real t14=FMUL((FMUL(t1, t1)), (FMUL(t1, t1)));
 s.noahres=FSUB(FSUB(FSUB(FSUB(FSUB(FADD(FSUB((FADD(solnet, lwdn)), sheat), ssoil_out), eta_e), (FMUL(FMUL(emissi, NSTBOLT), t14))), flx1), flx2), flx3);
 s.smstot=0.0f;s.sfcrunoff=FADD(s.sfcrunoff, FMUL(FMUL(runoff1, dt), 1000.0f));
 if(ffrozp>0.5f)s.acsnow=FADD(s.acsnow, FMUL(prcp, dt));
 if(s.snow>0.0f){s.acsnom=FADD(s.acsnom, FMUL(snomlt, 1000.0f));s.snopcx=FSUB(s.snopcx, FDIV(snomlt, (FDIV(dt, __uint_as_float(0x4da6e49cu) /* gfortran ROWLIW=ROW*ELIW */))));}
}
// module_sf_noahdrv.F:3251-3729; module_sf_noahlsm.F:69-888
static inline void noah_mosaic_tile(thread MosaicState& s,
 device const real* vegtbl,device const real* soiltbl,device const real* genp,device const real* dzs,
 real dt,int lucats,int slcats,int isurban,int isice,real xice_threshold,
 int itimestep,int frpcpn,int usemonalb,int rdlai2d,int opt_thcnd)
{
 const size_t idx=0,plane=1;
 thread const int* ivgtyp=&s.ivgtyp; thread const int* isltyp=&s.isltyp;
 thread real* psfc_a=&s.psfc;
 thread real* sfcprs_a=&s.sfcprs;
 thread real* sfctmp_a=&s.sfctmp;
 thread real* qv1_a=&s.qv1;
 thread real* qgh_a=&s.qgh;
 thread real* dz8w1_a=&s.dz8w1;
 thread real* glw_a=&s.glw;
 thread real* swdown_a=&s.swdown;
 thread real* rainbl_a=&s.rainbl;
 thread real* sr_a=&s.sr;
 thread real* chs_a=&s.chs;
 thread real* cqs2_a=&s.cqs2;
 thread real* chs2_a=&s.chs2;
 thread real* rib_a=&s.rib;
 thread real* vegfra_a=&s.vegfra;
 thread real* shdmin_a=&s.shdmin;
 thread real* shdmax_a=&s.shdmax;
 thread real* tmn_a=&s.tmn;
 thread real* xland_a=&s.xland;
 thread real* xice_a=&s.xice;
 thread real* snoalb_a=&s.snoalb;
 thread real* embck_a=&s.embck;
 thread real* tsk_a=&s.tsk;
 thread real* hfx_a=&s.hfx;
 thread real* qfx_a=&s.qfx;
 thread real* lh_a=&s.lh;
 thread real* grdflx_a=&s.grdflx;
 thread real* qsfc_a=&s.qsfc;
 thread real* canwat_a=&s.canwat;
 thread real* snow_a=&s.snow;
 thread real* snowc_a=&s.snowc;
 thread real* snowh_a=&s.snowh;
 thread real* albedo_a=&s.albedo;
 thread real* albbck_a=&s.albbck;
 thread real* emiss_a=&s.emiss;
 thread real* znt_a=&s.znt;
 thread real* z0_a=&s.z0;
 thread real* snotime_a=&s.snotime;
 thread real* lai_a=&s.lai;
 thread real* smstav_a=&s.smstav;
 thread real* smstot_a=&s.smstot;
 thread real* sfcrunoff_a=&s.sfcrunoff;
 thread real* udrunoff_a=&s.udrunoff;
 thread real* acsnow_a=&s.acsnow;
 thread real* acsnom_a=&s.acsnom;
 thread real* snopcx_a=&s.snopcx;
 thread real* potevp_a=&s.potevp;
 thread real* noahres_a=&s.noahres;
 thread real* chklowq_a=&s.chklowq;
 thread real* smois_a=s.smois;
 thread real* tslb_a=s.tslb;
 thread real* sh2o_a=s.sh2o;
 thread real* smcrel_a=s.smcrel;
    // ================= driver (module_sf_noahdrv.F lsm) prep =========
    // module_sf_noahdrv.F:809-814 writes CHKLOWQ for EVERY column, before the
    // XLAND land/sea branch in mosaic prep -- so open water, sea ice and land ice all
    // leave the driver with CHKLOWQ = 1.  This kernel used to return from the
    // three skip paths below without writing it, leaving whatever the caller
    // had in the array; the WRF oracle's water and sea-ice rows found it (0.0
    // against 1.0).  The `myj` arm that can set 0 instead is not ported --
    // launch_noah has no myj argument -- so the value is unconditionally 1.
    chklowq_a[idx] = 1.0f;
    if ((FSUB(xland_a[idx], 1.5f)) >= 0.0f) return;        // open water
    if (xice_a[idx] >= xice_threshold) {              // sea ice
        for (int k = 0; k < NSOIL; ++k) sh2o_a[k * plane + idx] = 1.0f;
        lai_a[idx] = 0.01f;
        return;
    }
    int vegtyp = ivgtyp[idx];
    int soiltyp = isltyp[idx];


    const real tresh = 0.95f, a2 = 17.67f, a3 = 273.15f, a4 = 29.65f;
    const real a23m4 = __uint_as_float(0x45867529u) /* gfortran15.2 fold_009 */;
    real psfc = psfc_a[idx];
    real sfcprs = sfcprs_a[idx];
    real q2k = FDIV(qv1_a[idx], (FADD(1.0f, qv1_a[idx])));
    real q2sat = FDIV(qgh_a[idx], (FADD(1.0f, qgh_a[idx])));
    real sfctmp = sfctmp_a[idx];
    real apes = gfk_pow(FDIV(1.0e5f, psfc), RCP);             // CAPA = R_d/CP
    real apelm = gfk_pow(FDIV(1.0e5f, sfcprs), RCP);
    real sfcth2 = FMUL(sfctmp, apelm);
    real th2 = FDIV(sfcth2, apes);
    real emissi = emiss_a[idx];
    real lwdn = FMUL(glw_a[idx], emissi);
    real soldn = swdown_a[idx];
    real solnet = FMUL(soldn, (FSUB(1.0f, albedo_a[idx])));
    real prcp = FDIV(rainbl_a[idx], dt);
    real shdfac = FDIV(vegfra_a[idx], 100.0f);
    if (vegtyp == 25 || vegtyp == 26 || vegtyp == 27) shdfac = 0.0f;
    real t1 = tsk_a[idx];
    real chk = chs_a[idx];
    real shmin = FDIV(shdmin_a[idx], 100.0f);
    real shmax = FDIV(shdmax_a[idx], 100.0f);
    real sneqv = FMUL(snow_a[idx], 0.001f);
    real snowhk = snowh_a[idx];
    real sncovr = snowc_a[idx];
    real ffrozp;
    if (frpcpn) ffrozp = sr_a[idx];
    else ffrozp = (sfctmp <= 273.15f) ? 1.0f : 0.0f;

    real dqsdt2 = FDIV(FMUL(q2sat, a23m4), (FMUL((FSUB(sfctmp, a4)), (FSUB(sfctmp, a4)))));
    if (snow_a[idx] > 0.0f) {
        real sfctsno = sfctmp;
        real e2sat = FMUL(611.2f, gfk_exp(FMUL(6174.0f, (FSUB(__uint_as_float(0x3b6fed42u) /* gfortran15.2 fold_011 */, FDIV(1.0f, sfctsno))))));
        real q2sati = FDIV(FMUL(0.622f, e2sat), (FSUB(sfcprs, e2sat)));
        q2sati = FDIV(q2sati, (FADD(1.0f, q2sati)));
        if (t1 > 273.14f) {
            q2sat = FADD(FMUL(q2sat, (FSUB(1.0f, snowc_a[idx]))), FMUL(q2sati, snowc_a[idx]));
            dqsdt2 = FADD(FMUL(dqsdt2, (FSUB(1.0f, snowc_a[idx]))), FMUL(FDIV(FMUL(q2sati, 6174.0f), (FMUL(sfctsno, sfctsno))), snowc_a[idx]));
        } else {
            q2sat = q2sati;
            dqsdt2 = FDIV(FMUL(q2sati, 6174.0f), (FMUL(sfctsno, sfctsno)));
        }
        if (t1 > 273.0f && snowc_a[idx] > 0.0f)
            dqsdt2 = FMUL(dqsdt2, (FSUB(1.0f, snowc_a[idx])));
    }
    real tbot = tmn_a[idx];
    if (soiltyp == 14 && xice_a[idx] == 0.0f) soiltyp = 7;
    real snoalb1 = snoalb_a[idx];
    real cmc = FDIV(canwat_a[idx], 1000.0f);
    real alb = albbck_a[idx];
    real z0brd = z0_a[idx];
    real embrd = embck_a[idx];
    real snotime1 = snotime_a[idx];
    real ribb = rib_a[idx];
    real smc[NSOIL], stc[NSOIL], swc[NSOIL];
    for (int k = 0; k < NSOIL; ++k) {
        smc[k] = smois_a[k * plane + idx];
        stc[k] = tslb_a[k * plane + idx];
        swc[k] = sh2o_a[k * plane + idx];
    }
    if ((sneqv != 0.0f && snowhk == 0.0f) || (snowhk <= sneqv))
        snowhk = FMUL(5.0f, sneqv);
    real xlai = lai_a[idx];
    if (rdlai2d) {
        if (shdfac > 0.0f && xlai <= 0.0f) xlai = 0.01f;
    }

#if NOAH_MOSAIC_UCM
    // noahdrv.F:3455-3474. No FRC_URB2D test on the rural temperature.
    if(s.urban_tile) {
        vegtyp=s.natural;
        shdfac=vegtbl[(s.natural-1)*NVEGC+VG_SHDTBL];
        albedo_a[idx]=0.2f; alb=0.2f; emissi=0.98f;
        lwdn=FMUL(glw_a[idx],emissi);
        solnet=FMUL(soldn,FSUB(1.0f,albedo_a[idx]));
        t1=s.rural_initial;
    }
#endif
    // ======================== SFLX ===================================
    real sldpth[NSOIL], zsoil[NSOIL];
    for (int k = 0; k < NSOIL; ++k) sldpth[k] = dzs[k];
    zsoil[0] = -sldpth[0];
    for (int kz = 1; kz < NSOIL; ++kz)
        zsoil[kz] = FADD(-sldpth[kz], zsoil[kz - 1]);

    // REDPRM from the packed tables (SLOPETYP = 1 as the driver sets)
    device const real* sv = soiltbl + (size_t)(soiltyp - 1) * NSOILC;
    device const real* vv = vegtbl + (size_t)(vegtyp - 1) * NVEGC;
    real csoil = genp[GEN_CSOIL];
    real bexp = sv[SO_BEXP];
    real dksat = sv[SO_DKSAT];
    real dwsat = sv[SO_DWSAT];
    real f1 = sv[SO_F1];
    real psisat = sv[SO_PSISAT];
    real quartz = sv[SO_QUARTZ];
    real smcdry = sv[SO_SMCDRY];
    real smcmax = sv[SO_SMCMAX];
    real smcref = sv[SO_SMCREF];
    real smcwlt = sv[SO_SMCWLT];
    real zbot = genp[GEN_ZBOT];
    real salp = genp[GEN_SALP];
    real sbeta = genp[GEN_SBETA];
    real refdk = genp[GEN_REFDK];
    real frzk = genp[GEN_FRZK];
    real fxexp = genp[GEN_FXEXP];
    real refkdt = genp[GEN_REFKDT];
    real kdt = FDIV(FMUL(refkdt, dksat), refdk);
    real slope = genp[GEN_SLOPE];
    real lvcoef = genp[GEN_LVCOEF];
    real frzfact = FMUL((FDIV(smcmax, smcref)), (__uint_as_float(0x3f615e16u) /* gfortran15.2 fold_017 */));
    real frzx = FMUL(frzk, frzfact);
    real topt = genp[GEN_TOPT];
    real cmcmax = genp[GEN_CMCMAX];
    real cfactr = genp[GEN_CFACTR];
    real rsmax = genp[GEN_RSMAX];
    int nroot = (int)vv[VG_NROOT];
    real snup = vv[VG_SNUP];
    real rsmin = vv[VG_RSMIN];
    real rgl = vv[VG_RGL];
    real hs = vv[VG_HS];
    real emissmin = vv[VG_EMISSMIN];
    real emissmax = vv[VG_EMISSMAX];
    real laimin = vv[VG_LAIMIN];
    real laimax = vv[VG_LAIMAX];
    real z0min = vv[VG_Z0MIN];
    real z0max = vv[VG_Z0MAX];
    real albedomin = vv[VG_ALBEDOMIN];
    real albedomax = vv[VG_ALBEDOMAX];
    int bare = (int)genp[GEN_BARE];
    if (vegtyp == bare) shdfac = 0.0f;
    real rtdis[NSOIL];
    for (int i = 0; i < NSOIL; ++i) rtdis[i] = 0.0f;
    for (int i = 0; i < nroot; ++i)
        rtdis[i] = FDIV(-sldpth[i], zsoil[nroot - 1]);

    // urban parameter overrides (plain Noah, no UCM)
    if (vegtyp == isurban) {
        shdfac = 0.05f;
        rsmin = 400.0f;
        smcmax = 0.45f;
        smcref = 0.42f;
        smcwlt = 0.40f;
        smcdry = 0.40f;
    }

    // background emissivity / LAI / albedo / roughness interpolation
    real embrd_o;
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
            real interp_fraction = FDIV((FSUB(shdfac, shmin)), (FSUB(shmax, shmin)));
            interp_fraction = fmin(interp_fraction, 1.0f);
            interp_fraction = fmax(interp_fraction, 0.0f);
            embrd_o = FADD(FMUL((FSUB(1.0f, interp_fraction)), emissmin), FMUL(interp_fraction, emissmax));
            if (!rdlai2d) xlai = FADD(FMUL((FSUB(1.0f, interp_fraction)), laimin), FMUL(interp_fraction, laimax));
            if (!usemonalb) alb = FADD(FMUL((FSUB(1.0f, interp_fraction)), albedomax), FMUL(interp_fraction, albedomin));
            z0brd = FADD(FMUL((FSUB(1.0f, interp_fraction)), z0min), FMUL(interp_fraction, z0max));
        } else {
            embrd_o = FADD(FMUL(0.5f, emissmin), FMUL(0.5f, emissmax));
            if (!rdlai2d) xlai = FADD(FMUL(0.5f, laimin), FMUL(0.5f, laimax));
            if (!usemonalb) alb = FADD(FMUL(0.5f, albedomin), FMUL(0.5f, albedomax));
            z0brd = FADD(FMUL(0.5f, z0min), FMUL(0.5f, z0max));
        }
    }
    embrd = embrd_o;

    // snowpack density / precipitation type
    bool snowng = false, frzgra = false;
    real sndens, sncond;
    if (sneqv <= 1.0e-7f) {
        sneqv = 0.0f;
        sndens = 0.0f;
        snowhk = 0.0f;
        sncond = 1.0f;
    } else {
        sndens = FDIV(sneqv, snowhk);
        sncond = noah_csnow(sndens);
    }
    if (prcp > 0.0f) {
        if (ffrozp > 0.5f) snowng = true;
        else if (t1 <= NTFREEZ) frzgra = true;
    }
    real prcpf;
    if (snowng || frzgra) {
        real sn_new = FMUL(FMUL(prcp, dt), 0.001f);
        sneqv = FADD(sneqv, sn_new);
        prcpf = 0.0f;
        noah_snow_new(sfctmp, sn_new, snowhk, sndens);
        sncond = noah_csnow(sndens);
    } else {
        prcpf = prcp;
    }

    // snow cover fraction, snow albedo, emissivity
    real albedo;
    if (sneqv == 0.0f) {
        sncovr = 0.0f;
        albedo = alb;
        emissi = embrd;
    } else {
        sncovr = noah_snfrac(sneqv, snup, salp);
        sncovr = fmin(sncovr, 0.98f);
        noah_alcalc(alb, snoalb1, embrd, sncovr, dt, snowng, snotime1,
                    lvcoef, albedo, emissi);
    }

    // surface thermal conductivity + first-guess soil heat flux
    real df1 = noah_tdfcnd(smc[0], quartz, smcmax, swc[0], bexp,
                           psisat, soiltyp, opt_thcnd);
    if (vegtyp == isurban) df1 = 3.24f;
    df1 = FMUL(df1, gfk_exp(FMUL(sbeta, shdfac)));
    if (sncovr > 0.97f) df1 = sncond;
    real dsoil = -(FMUL(0.5f, zsoil[0]));
    real dtot = 0.0f;
    real ssoil;
    if (sneqv == 0.0f) {
        ssoil = FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dsoil);
    } else {
        dtot = FADD(snowhk, dsoil);
        real frcsno = FDIV(snowhk, dtot);
        real frcsoi = FDIV(dsoil, dtot);
        real df1a = FADD(FMUL(frcsno, sncond), FMUL(frcsoi, df1));
        df1 = FADD(FMUL(df1a, sncovr), FMUL(df1, (FSUB(1.0f, sncovr))));
        ssoil = FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dtot);
    }

    // roughness over snow
    real z0k;
    if (sncovr > 0.0f) z0k = noah_snowz0(sncovr, z0brd, snowhk);
    else z0k = z0brd;

    // Penman potential evaporation
    real fdown = FADD(solnet, lwdn);
    real t2v = FMUL(sfctmp, (FADD(1.0f, FMUL(0.61f, q2k))));
    real etp, rch, rr, epsca, t24, flx2;
    noah_penman(sfctmp, sfcprs, chk, t2v, th2, prcp, fdown, ssoil, q2k,
                q2sat, dqsdt2, snowng, frzgra, emissi, sncovr, etp,
                rch, rr, epsca, t24, flx2);

    // canopy resistance
    real rc = 0.0f, pc = 0.0f;
    if (shdfac > 0.0f && xlai > 0.0f)
        noah_canres(soldn, chk, sfctmp, q2k, swc, zsoil, smcwlt,
                    smcref, rsmin, nroot, q2sat, dqsdt2, topt, rsmax,
                    rgl, hs, xlai, sfcprs, emissi, rc, pc);

    // ---- NOPAC / SNOPAC -------------------------------------------
    real eta_kin, beta, flx1, flx3, ssoil_pac;
    real runoff1, runoff2, runoff3, dew, drip;
    real edir1k = 0.0f, ec1k = 0.0f, ett1k = 0.0f;
    real et1k[NSOIL];
    for (int k = 0; k < NSOIL; ++k) et1k[k] = 0.0f;
    real etns = 0.0f, esnow = 0.0f, snomlt = 0.0f;
    int ebal_case;


    if (sneqv == 0.0f) {
        // =================== NOPAC ===================================
        ebal_case = 0;
        real prcp1 = FMUL(prcp, 0.001f);
        real etp1 = FMUL(etp, 0.001f);
        dew = 0.0f;
        real eta = 0.0f;
        if (etp > 0.0f) {
            real eta1;
            noah_evapo(eta1, smc, cmc, etp1, dt, swc, smcmax, pc,
                       smcwlt, smcref, shdfac, cmcmax, smcdry, cfactr,
                       nroot, rtdis, fxexp, edir1k, ec1k, et1k, ett1k);
            noah_smflx(smc, cmc, dt, prcp1, zsoil, swc, slope, kdt,
                       frzx, smcmax, bexp, smcwlt, dksat, dwsat,
                       shdfac, cmcmax, edir1k, ec1k, et1k, runoff1,
                       runoff2, runoff3, drip);
            eta = FMUL(eta1, 1000.0f);
        } else {
            dew = -etp1;
            prcp1 = FADD(prcp1, dew);
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
            beta = FDIV(eta, etp);
        }
        real df1n = noah_tdfcnd(smc[0], quartz, smcmax, swc[0], bexp,
                                psisat, soiltyp, opt_thcnd);
        if (vegtyp == isurban) df1n = 3.24f;
        df1n = FMUL(df1n, gfk_exp(FMUL(sbeta, shdfac)));
        real yynum = FSUB(fdown, FMUL(FMUL(emissi, NSIGMA), t24));
        real yy = FADD(sfctmp, FDIV((FSUB(FSUB(FADD(FDIV(yynum, rch), th2), sfctmp), FMUL(beta, epsca))), rr));
        real zz1 = FADD(FDIV(df1n, (FMUL(FMUL(FMUL(-0.5f, zsoil[0]), rch), rr))), 1.0f);
        noah_shflx(stc, smc, smcmax, t1, dt, yy, zz1, zsoil, tbot,
                   zbot, psisat, swc, bexp, df1n, quartz, csoil,
                   vegtyp, isurban, soiltyp, opt_thcnd, ssoil_pac);
        flx1 = FMUL(FMUL(NCPH2O, prcp), (FSUB(t1, sfctmp)));
        flx3 = 0.0f;
        eta_kin = eta;
                               // filled below (needs eta_e)
    } else {
        // =================== SNOPAC ==================================
        const real esdmin = 1.0e-6f, snoexp = 2.0f;
        real esd = sneqv;
        dew = 0.0f;
        real etns1 = 0.0f;
        real esnow1, esnow2;
        esnow = 0.0f;
        esnow1 = 0.0f;
        esnow2 = 0.0f;
        real prcp1 = FMUL(prcpf, 0.001f);
        beta = 1.0f;
        real etanrg, etp1;
        if (etp <= 0.0f) {
            if (ribb >= 0.1f && fdown > 150.0f) {
                etp = FDIV((FADD(FDIV(FMUL(fmin(FMUL(etp, (FSUB(1.0f, ribb))), 0.0f), sncovr), 0.980f), FMUL(etp, (FSUB(0.980f, sncovr))))), 0.980f);
            }
            if (etp == 0.0f) beta = 0.0f;
            etp1 = FMUL(etp, 0.001f);
            dew = -etp1;
            esnow2 = FMUL(etp1, dt);
            etanrg = FMUL(etp, (FADD(FMUL((FSUB(1.0f, sncovr)), NLSUBC), FMUL(sncovr, NLSUBS))));
        } else {
            etp1 = FMUL(etp, 0.001f);
            if (sncovr < 1.0f) {
                noah_evapo(etns1, smc, cmc, etp1, dt, swc, smcmax, pc,
                           smcwlt, smcref, shdfac, cmcmax, smcdry,
                           cfactr, nroot, rtdis, fxexp, edir1k, ec1k,
                           et1k, ett1k);
                edir1k = FMUL(edir1k, (FSUB(1.0f, sncovr)));
                ec1k = FMUL(ec1k, (FSUB(1.0f, sncovr)));
                for (int k = 0; k < NSOIL; ++k)
                    et1k[k] = FMUL(et1k[k], (FSUB(1.0f, sncovr)));
                ett1k = FMUL(ett1k, (FSUB(1.0f, sncovr)));
                etns1 = FMUL(etns1, (FSUB(1.0f, sncovr)));
                etns = FMUL(etns1, 1000.0f);
            }
            esnow = FMUL(etp, sncovr);
            esnow1 = FMUL(esnow, 0.001f);
            esnow2 = FMUL(esnow1, dt);
            etanrg = FADD(FMUL(esnow, NLSUBS), FMUL(etns, NLSUBC));
        }
        flx1 = 0.0f;
        if (snowng) flx1 = FMUL(FMUL(NCPICE, prcp), (FSUB(t1, sfctmp)));
        else if (prcp > 0.0f) flx1 = FMUL(FMUL(NCPH2O, prcp), (FSUB(t1, sfctmp)));
        real dsoil2 = FMUL(-0.5f, zsoil[0]);
        dtot = FADD(snowhk, dsoil2);
        real denom = FADD(1.0f, FDIV(df1, (FMUL(FMUL(dtot, rr), rch))));
        real t12a = FDIV((FSUB(FSUB(FADD(FDIV((FSUB(FSUB(FSUB(fdown, flx1), flx2), FMUL(FMUL(emissi, NSIGMA), t24))), rch), th2), sfctmp), FDIV(etanrg, rch))), rr);
        real t12b = FDIV(FMUL(df1, stc[0]), (FMUL(FMUL(dtot, rr), rch)));
        real t12 = FDIV((FADD(FADD(sfctmp, t12a), t12b)), denom);
        real stc1_old = stc[0];
        real ex = 0.0f;
        snomlt = 0.0f;
        if (t12 <= NTFREEZ) {
            ebal_case = 1;
            t1 = t12;
            ssoil_pac = FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dtot);
            esd = fmax(0.0f, FSUB(esd, esnow2));
            flx3 = 0.0f;

        } else {
            t1 = FADD(FMUL(NTFREEZ, fmax(0.01f, gfk_pow(sncovr, snoexp))), FMUL(t12, (FSUB(1.0f, fmax(0.01f, gfk_pow(sncovr, snoexp))))));
            beta = 1.0f;
            ssoil_pac = FDIV(FMUL(df1, (FSUB(t1, stc[0]))), dtot);
            bool clipped = false;
            if (FSUB(esd, esnow2) <= esdmin) {
                esd = 0.0f;
                ex = 0.0f;
                snomlt = 0.0f;
                flx3 = 0.0f;
                clipped = true;
            } else {
                esd = FSUB(esd, esnow2);
                real seh = FMUL(rch, (FSUB(t1, th2)));
                real t14 = FMUL(t1, t1);
                t14 = FMUL(t14, t14);
                flx3 = FSUB(FSUB(FSUB(FSUB(FSUB(FSUB(fdown, flx1), flx2), FMUL(FMUL(emissi, NSIGMA), t14)), ssoil_pac), seh), etanrg);
                real flx3_def = flx3;
                if (flx3 <= 0.0f) flx3 = 0.0f;
                if (flx3 != flx3_def) clipped = true;
                ex = FDIV(FMUL(flx3, 0.001f), NLSUBF);
                snomlt = FMUL(ex, dt);
                if (FSUB(esd, snomlt) >= esdmin) {
                    esd = FSUB(esd, snomlt);
                } else {
                    ex = FDIV(esd, dt);
                    flx3 = FMUL(FMUL(ex, 1000.0f), NLSUBF);
                    snomlt = esd;
                    esd = 0.0f;
                    clipped = true;
                }
            }
            ebal_case = clipped ? 3 : 2;
            prcp1 = FADD(prcp1, ex);
            real seh = FMUL(rch, (FSUB(t1, th2)));
            real t14 = FMUL(t1, t1);
            t14 = FMUL(t14, t14);

        }
        noah_smflx(smc, cmc, dt, prcp1, zsoil, swc, slope, kdt,
                   frzx, smcmax, bexp, smcwlt, dksat, dwsat, shdfac,
                   cmcmax, edir1k, ec1k, et1k, runoff1, runoff2,
                   runoff3, drip);
        real zz1 = 1.0f;
        real yy = FSUB(stc[0], FDIV(FMUL(FMUL(FMUL(0.5f, ssoil_pac), zsoil[0]), zz1), df1));
        real t11 = t1;
        real ssoil1;
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
        eta_kin = FSUB(FADD(esnow, etns), FMUL(1000.0f, dew));
    }

    real q1 = FADD(q2k, FDIV(FMUL(eta_kin, CP), rch));
    real sheat = FMUL(FDIV(-(FMUL(FMUL(chk, CP), sfcprs)), (FMUL(NR_SHEAT, t2v))), (FSUB(th2, t1)));

    // kinematic -> energy conversions (SFLX epilogue)
    real edir_e = FMUL((FMUL(edir1k, 1000.0f)), NLVH2O);
    real ec_e = FMUL((FMUL(ec1k, 1000.0f)), NLVH2O);
    real ett_e = FMUL((FMUL(ett1k, 1000.0f)), NLVH2O);
    real esnow_e = FMUL(esnow, NLSUBS);
    real etp_e = FMUL(etp, (FADD(FMUL((FSUB(1.0f, sncovr)), NLVH2O), FMUL(sncovr, NLSUBS))));
    real eta_e;
    if (etp_e > 0.0f) eta_e = FADD(FADD(FADD(edir_e, ec_e), ett_e), esnow_e);
    else eta_e = etp_e;
    if (etp_e == 0.0f) beta = 0.0f;
    else beta = FDIV(eta_e, etp_e);

    if (ebal_case == 0) {

    }

    real ssoil_out = FMUL(-1.0f, ssoil_pac);
    runoff3 = FDIV(runoff3, dt);
    runoff2 = FADD(runoff2, runoff3);
    real soilm = FMUL(FMUL(-1.0f, smc[0]), zsoil[0]);
    for (int k = 1; k < NSOIL; ++k)
        soilm = FADD(soilm, FMUL(smc[k], (FSUB(zsoil[k - 1], zsoil[k]))));
    real soilwm = FMUL(FMUL(-1.0f, (FSUB(smcmax, smcwlt))), zsoil[0]);
    real soilww = FMUL(FMUL(-1.0f, (FSUB(smc[0], smcwlt))), zsoil[0]);
    real smav[NSOIL];
    for (int k = 0; k < NSOIL; ++k)
        smav[k] = FDIV((FSUB(smc[k], smcwlt)), (FSUB(smcmax, smcwlt)));
    if (nroot >= 2) {
        for (int k = 1; k < nroot; ++k) {
            soilwm = FADD(soilwm, FMUL((FSUB(smcmax, smcwlt)), (FSUB(zsoil[k - 1], zsoil[k]))));
            soilww = FADD(soilww, FMUL((FSUB(smc[k], smcwlt)), (FSUB(zsoil[k - 1], zsoil[k]))));
        }
    }
    real soilw;
    if (soilwm < 1.0e-6f) {
        soilwm = 0.0f;
        soilw = 0.0f;
        soilm = 0.0f;
    } else {
        soilw = FDIV(soilww, soilwm);
    }

    // ================= driver post-SFLX updates ======================
    lai_a[idx] = xlai;
    canwat_a[idx] = FMUL(cmc, 1000.0f);
    snow_a[idx] = FMUL(sneqv, 1000.0f);
    snowh_a[idx] = snowhk;
    albedo_a[idx] = albedo;
    albbck_a[idx] = alb;
    z0_a[idx] = z0brd;
    emiss_a[idx] = emissi;
    znt_a[idx] = z0k;
#if NOAH_MOSAIC_UCM
    s.rural_t1=t1; s.rural_q1=q1;
#endif
    tsk_a[idx] = t1;
    hfx_a[idx] = sheat;
    potevp_a[idx] = FADD(potevp_a[idx], FMUL(etp_e, (FDIV(dt, (__uint_as_float(0x4f1502f9u) /* gfortran15.2 fold_016 */)))));
    qfx_a[idx] = eta_kin;
    lh_a[idx] = eta_e;
    grdflx_a[idx] = ssoil_out;
    snowc_a[idx] = sncovr;
    // WRF module_sf_noahdrv.F:3691: ordinary-land Noah makes the 2 m
    // thermal exchange coefficient identical to its moisture coefficient.
    chs2_a[idx] = cqs2_a[idx];
    snotime_a[idx] = snotime1;
    if (q1 > qsfc_a[idx]) cqs2_a[idx] = chs_a[idx];
    qsfc_a[idx] = FDIV(q1, (FSUB(1.0f, q1)));
    for (int k = 0; k < NSOIL; ++k) {
        smois_a[k * plane + idx] = smc[k];
        tslb_a[k * plane + idx] = stc[k];
        sh2o_a[k * plane + idx] = swc[k];
        smcrel_a[k * plane + idx] = smav[k];
    }
    real t14 = FMUL(t1, t1); t14 = FMUL(t14, t14);
    noahres_a[idx] = FSUB(FSUB(FSUB(FSUB(FSUB(FADD(FSUB((FADD(solnet, lwdn)), sheat), ssoil_out), eta_e), (FMUL(FMUL(emissi, NSTBOLT), t14))), flx1), flx2), flx3);
    chklowq_a[idx] = 1.0f;
    smstav_a[idx] = soilw;
    smstot_a[idx] = FMUL(soilm, 1000.0f);
    sfcrunoff_a[idx] = FADD(sfcrunoff_a[idx], FMUL(FMUL(runoff1, dt), 1000.0f));
    udrunoff_a[idx] = FADD(udrunoff_a[idx], FMUL(FMUL(runoff2, dt), 1000.0f));
    if (ffrozp > 0.5f)
        acsnow_a[idx] = FADD(acsnow_a[idx], FMUL(prcp, dt));
    if (snow_a[idx] > 0.0f) {
        acsnom_a[idx] = FADD(acsnom_a[idx], FMUL(snomlt, 1000.0f));
        snopcx_a[idx] = FSUB(snopcx_a[idx], FDIV(snomlt, (FDIV(dt, __uint_as_float(0x4da6e49cu) /* gfortran ROWLIW=ROW*ELIW */))));
    }
}
struct MosaicSums {
 real tsk;
 real znt;
 real z0;
 real area;
 real r1;
 real r2;
 real pe;
 real sm;
 real sx;
 real qsfc;
 real canwat;
 real snow;
 real snowh;
 real snowc;
 real albedo;
 real albbck;
 real emiss;
 real embck;
 real hfx;
 real qfx;
 real lh;
 real grdflx;
 real soil[3][4];
};
// module_sf_noahdrv.F:4139-4175, D1 corrected soil indices; D2 weighted increments.
static inline void noah_mosaic_accumulate(thread MosaicState& s,thread const MosaicState& tile,
 thread MosaicSums& avg,real area) {
 real t4=FMUL(tile.tsk, tile.tsk);t4=FMUL(t4, t4);avg.tsk=FADD(avg.tsk, FMUL((FMUL(tile.emiss, t4)), area));
 avg.qsfc=FADD(avg.qsfc, FMUL(tile.qsfc, area));
 avg.canwat=FADD(avg.canwat, FMUL(tile.canwat, area));
 avg.snow=FADD(avg.snow, FMUL(tile.snow, area));
 avg.snowh=FADD(avg.snowh, FMUL(tile.snowh, area));
 avg.snowc=FADD(avg.snowc, FMUL(tile.snowc, area));
 avg.albedo=FADD(avg.albedo, FMUL(tile.albedo, area));
 avg.albbck=FADD(avg.albbck, FMUL(tile.albbck, area));
 avg.emiss=FADD(avg.emiss, FMUL(tile.emiss, area));
 avg.embck=FADD(avg.embck, FMUL(tile.embck, area));
 avg.hfx=FADD(avg.hfx, FMUL(tile.hfx, area));
 avg.qfx=FADD(avg.qfx, FMUL(tile.qfx, area));
 avg.lh=FADD(avg.lh, FMUL(tile.lh, area));
 avg.grdflx=FADD(avg.grdflx, FMUL(tile.grdflx, area));
 avg.znt=FADD(avg.znt, FMUL(gfk_log(tile.znt), area));avg.z0=FADD(avg.z0, FMUL(gfk_log(tile.z0), area));avg.area=FADD(avg.area, area);
 for(int k=0;k<4;++k)avg.soil[0][k]=FADD(avg.soil[0][k], FMUL(tile.smois[k], area));
 for(int k=0;k<4;++k)avg.soil[1][k]=FADD(avg.soil[1][k], FMUL(tile.tslb[k], area));
 for(int k=0;k<4;++k)avg.soil[2][k]=FADD(avg.soil[2][k], FMUL(tile.sh2o[k], area));
 avg.r1=FADD(avg.r1, FMUL(tile.sfcrunoff, area));avg.r2=FADD(avg.r2, FMUL(tile.udrunoff, area));
 avg.pe=FADD(avg.pe, FMUL(tile.potevp, area));avg.sm=FADD(avg.sm, FMUL(tile.acsnom, area));avg.sx=FADD(avg.sx, FMUL((-tile.snopcx), area));
 s.cqs2=tile.cqs2;s.chs2=tile.chs2;s.lai=tile.lai;s.snotime=tile.snotime;
 s.smstav=tile.smstav;s.smstot=tile.smstot;s.noahres=tile.noahres;s.chklowq=tile.chklowq;
 for(int k=0;k<4;++k)s.smcrel[k]=tile.smcrel[k];
}
// module_sf_noahdrv.F:4181-4213; D2 adds corrected increments once per cell.
static inline void noah_mosaic_mean(thread MosaicState& s,thread const MosaicSums& avg,
 real dt,int frpcpn) {
s.qsfc=avg.qsfc;
 s.canwat=avg.canwat;
 s.snow=avg.snow;
 s.snowh=avg.snowh;
 s.snowc=avg.snowc;
 s.albedo=avg.albedo;
 s.albbck=avg.albbck;
 s.emiss=avg.emiss;
 s.embck=avg.embck;
 s.hfx=avg.hfx;
 s.qfx=avg.qfx;
 s.lh=avg.lh;
 s.grdflx=avg.grdflx;
 for(int k=0;k<4;++k)s.smois[k]=avg.soil[0][k];
 for(int k=0;k<4;++k)s.tslb[k]=avg.soil[1][k];
 for(int k=0;k<4;++k)s.sh2o[k]=avg.soil[2][k];
 s.tsk=gfk_pow(FDIV(avg.tsk, s.emiss),0.25f);s.znt=gfk_exp(FDIV(avg.znt, avg.area));s.z0=gfk_exp(FDIV(avg.z0, avg.area));
 s.sfcrunoff=FADD(s.sfcrunoff, avg.r1);s.udrunoff=FADD(s.udrunoff, avg.r2);s.potevp=FADD(s.potevp, avg.pe);
 s.acsnom=FADD(s.acsnom, avg.sm);s.snopcx=FSUB(s.snopcx, avg.sx);
 real frozen=frpcpn?s.sr:(s.sfctmp<=273.15f?1.0f:0.0f);
 if(frozen>0.5f)s.acsnow=FADD(s.acsnow, FMUL((FDIV(s.rainbl, dt)), dt));
}

#include "../generated/noah_mosaic_column_bindings.metalh"
kernel void noah_mosaic_column(device NoahMosaicColumnResources& r [[buffer(0)]], constant NoahMosaicColumnParams& p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const int* ivgtyp = r.ivgtyp;
    device const int* isltyp = r.isltyp;
    int ny=p.ny; int nx=p.nx; const size_t plane=(size_t)ny*nx;
    device float* state_fields=r.state_fields;
    device float* psfc_a=state_fields+0*plane;
    device float* sfcprs_a=state_fields+1*plane;
    device float* sfctmp_a=state_fields+2*plane;
    device float* qv1_a=state_fields+3*plane;
    device float* qgh_a=state_fields+4*plane;
    device float* dz8w1_a=state_fields+5*plane;
    device float* glw_a=state_fields+6*plane;
    device float* swdown_a=state_fields+7*plane;
    device float* rainbl_a=state_fields+8*plane;
    device float* sr_a=state_fields+9*plane;
    device float* chs_a=state_fields+10*plane;
    device float* cqs2_a=state_fields+11*plane;
    device float* chs2_a=state_fields+12*plane;
    device float* rib_a=state_fields+13*plane;
    device float* vegfra_a=state_fields+14*plane;
    device float* shdmin_a=state_fields+15*plane;
    device float* shdmax_a=state_fields+16*plane;
    device float* tmn_a=state_fields+17*plane;
    device float* xland_a=state_fields+18*plane;
    device float* xice_a=state_fields+19*plane;
    device float* snoalb_a=state_fields+20*plane;
    device float* embck_a=state_fields+21*plane;
    device float* tsk_a=state_fields+22*plane;
    device float* hfx_a=state_fields+23*plane;
    device float* qfx_a=state_fields+24*plane;
    device float* lh_a=state_fields+25*plane;
    device float* grdflx_a=state_fields+26*plane;
    device float* qsfc_a=state_fields+27*plane;
    device float* canwat_a=state_fields+28*plane;
    device float* snow_a=state_fields+29*plane;
    device float* snowc_a=state_fields+30*plane;
    device float* snowh_a=state_fields+31*plane;
    device float* albedo_a=state_fields+32*plane;
    device float* albbck_a=state_fields+33*plane;
    device float* emiss_a=state_fields+34*plane;
    device float* znt_a=state_fields+35*plane;
    device float* z0_a=state_fields+36*plane;
    device float* snotime_a=state_fields+37*plane;
    device float* lai_a=state_fields+38*plane;
    device float* smstav_a=state_fields+39*plane;
    device float* smstot_a=state_fields+40*plane;
    device float* sfcrunoff_a=state_fields+41*plane;
    device float* udrunoff_a=state_fields+42*plane;
    device float* acsnow_a=state_fields+43*plane;
    device float* acsnom_a=state_fields+44*plane;
    device float* snopcx_a=state_fields+45*plane;
    device float* potevp_a=state_fields+46*plane;
    device float* noahres_a=state_fields+47*plane;
    device float* chklowq_a=state_fields+48*plane;
    device float* smois_a=state_fields+(49)*plane;
    device float* tslb_a=state_fields+(53)*plane;
    device float* sh2o_a=state_fields+(57)*plane;
    device float* smcrel_a=state_fields+(61)*plane;
    device float* tsk_mosaic=r.tsk_mosaic;
    device float* qsfc_mosaic=r.qsfc_mosaic;
    device float* canwat_mosaic=r.canwat_mosaic;
    device float* snow_mosaic=r.snow_mosaic;
    device float* snowh_mosaic=r.snowh_mosaic;
    device float* snowc_mosaic=r.snowc_mosaic;
    device float* albedo_mosaic=r.albedo_mosaic;
    device float* albbck_mosaic=r.albbck_mosaic;
    device float* emiss_mosaic=r.emiss_mosaic;
    device float* embck_mosaic=r.embck_mosaic;
    device float* znt_mosaic=r.znt_mosaic;
    device float* z0_mosaic=r.z0_mosaic;
    device float* hfx_mosaic=r.hfx_mosaic;
    device float* qfx_mosaic=r.qfx_mosaic;
    device float* lh_mosaic=r.lh_mosaic;
    device float* grdflx_mosaic=r.grdflx_mosaic;
    device float* snotime_mosaic=r.snotime_mosaic;
    device float* tslb_mosaic=r.tslb_mosaic;
    device float* smois_mosaic=r.smois_mosaic;
    device float* sh2o_mosaic=r.sh2o_mosaic;
    device const int* mosaic_cat_index=r.mosaic_cat_index;
    device const float* landusef2=r.landusef2;
    device const int* lcz=r.lcz;
    device const real* vegtbl=r.vegtbl;
    device const real* soiltbl=r.soiltbl;
    device const real* genp=r.genp;
    device const real* dzs=r.dzs;
    int nlcz=p.nlcz;
    real dt=p.dt;
    int lucats=p.lucats;
    int slcats=p.slcats;
    int isurban=p.isurban;
    int isice=p.isice;
    real xice_threshold=p.xice_threshold;
    int itimestep=p.itimestep;
    int frpcpn=p.frpcpn;
    int usemonalb=p.usemonalb;
    int rdlai2d=p.rdlai2d;
    int opt_thcnd=p.opt_thcnd;
    int mosaic_cat=p.mosaic_cat;
 uint col=gid; if(col>=uint(ny)*uint(nx))return;
 const size_t idx=col;
 MosaicState s;
#if NOAH_MOSAIC_UCM
 s.urban_tile=false; s.natural=natural;
#endif
 s.ivgtyp=ivgtyp[idx];s.isltyp=isltyp[idx];
 s.psfc=psfc_a[idx];
 s.sfcprs=sfcprs_a[idx];
 s.sfctmp=sfctmp_a[idx];
 s.qv1=qv1_a[idx];
 s.qgh=qgh_a[idx];
 s.dz8w1=dz8w1_a[idx];
 s.glw=glw_a[idx];
 s.swdown=swdown_a[idx];
 s.rainbl=rainbl_a[idx];
 s.sr=sr_a[idx];
 s.chs=chs_a[idx];
 s.cqs2=cqs2_a[idx];
 s.chs2=chs2_a[idx];
 s.rib=rib_a[idx];
 s.vegfra=vegfra_a[idx];
 s.shdmin=shdmin_a[idx];
 s.shdmax=shdmax_a[idx];
 s.tmn=tmn_a[idx];
 s.xland=xland_a[idx];
 s.xice=xice_a[idx];
 s.snoalb=snoalb_a[idx];
 s.embck=embck_a[idx];
 s.tsk=tsk_a[idx];
 s.hfx=hfx_a[idx];
 s.qfx=qfx_a[idx];
 s.lh=lh_a[idx];
 s.grdflx=grdflx_a[idx];
 s.qsfc=qsfc_a[idx];
 s.canwat=canwat_a[idx];
 s.snow=snow_a[idx];
 s.snowc=snowc_a[idx];
 s.snowh=snowh_a[idx];
 s.albedo=albedo_a[idx];
 s.albbck=albbck_a[idx];
 s.emiss=emiss_a[idx];
 s.znt=znt_a[idx];
 s.z0=z0_a[idx];
 s.snotime=snotime_a[idx];
 s.lai=lai_a[idx];
 s.smstav=smstav_a[idx];
 s.smstot=smstot_a[idx];
 s.sfcrunoff=sfcrunoff_a[idx];
 s.udrunoff=udrunoff_a[idx];
 s.acsnow=acsnow_a[idx];
 s.acsnom=acsnom_a[idx];
 s.snopcx=snopcx_a[idx];
 s.potevp=potevp_a[idx];
 s.noahres=noahres_a[idx];
 s.chklowq=chklowq_a[idx];
 for(int k=0;k<4;++k)s.smois[k]=smois_a[k*plane+idx];
 for(int k=0;k<4;++k)s.tslb[k]=tslb_a[k*plane+idx];
 for(int k=0;k<4;++k)s.sh2o[k]=sh2o_a[k*plane+idx];
 for(int k=0;k<4;++k)s.smcrel[k]=smcrel_a[k*plane+idx];
 // module_sf_noahdrv.F:3118-3157
 if(itimestep==1) {
  if(s.xland>=1.5f) {
   s.smstav=1.0f;s.smstot=1.0f;
   for(int k=0;k<4;++k){s.smois[k]=1.0f;s.tslb[k]=273.16f;s.smcrel[k]=1.0f;}
  } else if(s.xice>=xice_threshold) {
   s.smstav=1.0f;s.smstot=1.0f;
   for(int k=0;k<4;++k){s.smois[k]=1.0f;s.smcrel[k]=1.0f;}
  }
 }
 s.chklowq=1.0f;
 if(s.xland<1.5f && s.xice<xice_threshold) {
 MosaicSums avg={};
 real carried_soilw=0.0f;
 for(int t=mosaic_cat-1;t>=0;--t){
 size_t off=t*plane+idx;
 MosaicState tile=s;
 tile.ivgtyp=mosaic_cat_index[off];
#if NOAH_MOSAIC_UCM
 int utype=tile.ivgtyp==isurban?(use_lcz?5:2):0;
 for(int l=0;l<nlcz;++l)if(tile.ivgtyp==lcz[l])utype=l+1;
 tile.urban_tile=utype!=0;
 // Preserve the original tile category for UTYPE; remap only VEGTYP.
#else
 for(int l=0;l<nlcz;++l)if(tile.ivgtyp==lcz[l])tile.ivgtyp=isurban;
#endif
 tile.tsk=tsk_mosaic[off];
 tile.qsfc=qsfc_mosaic[off];
 tile.canwat=canwat_mosaic[off];
 tile.snow=snow_mosaic[off];
 tile.snowh=snowh_mosaic[off];
 tile.snowc=snowc_mosaic[off];
 tile.albedo=albedo_mosaic[off];
 tile.albbck=albbck_mosaic[off];
 tile.emiss=emiss_mosaic[off];
 tile.embck=embck_mosaic[off];
 tile.znt=znt_mosaic[off];
 tile.z0=z0_mosaic[off];
 tile.hfx=hfx_mosaic[off];
 tile.qfx=qfx_mosaic[off];
 tile.lh=lh_mosaic[off];
 tile.grdflx=grdflx_mosaic[off];
 tile.snotime=snotime_mosaic[off];
 for(int k=0;k<4;++k)tile.tslb[k]=tslb_mosaic[(4*t+k)*plane+idx];
 for(int k=0;k<4;++k)tile.smois[k]=smois_mosaic[(4*t+k)*plane+idx];
 for(int k=0;k<4;++k)tile.sh2o[k]=sh2o_mosaic[(4*t+k)*plane+idx];
 tile.sfcrunoff=0.0f;tile.udrunoff=0.0f;tile.potevp=0.0f;tile.acsnom=0.0f;tile.snopcx=0.0f;tile.acsnow=0.0f;
 tile.smstav=carried_soilw;
#if NOAH_MOSAIC_UCM
 if(tile.urban_tile)tile.rural_initial=PLANE(urban_tiles,7)[off];
#endif
 if(tile.ivgtyp==isice)noah_mosaic_tile_glacial(tile,vegtbl,soiltbl,genp,dzs,dt,lucats,slcats,isurban,isice,xice_threshold,itimestep,frpcpn,usemonalb,rdlai2d,opt_thcnd);
 else noah_mosaic_tile(tile,vegtbl,soiltbl,genp,dzs,dt,lucats,slcats,isurban,isice,xice_threshold,itimestep,frpcpn,usemonalb,rdlai2d,opt_thcnd);
#if NOAH_MOSAIC_UCM
 if(tile.urban_tile) {
  // noahdrv.F:3733-4057. Roof/wall/road stacks tile; SMR/TGRL stay grid.
  UcmCol c; ucm_load_state(urban_state,idx,plane,c);
  c.tr=PLANE(urban_tiles,0)[off];c.tb=PLANE(urban_tiles,1)[off];
  c.tg=PLANE(urban_tiles,2)[off];c.tc=PLANE(urban_tiles,3)[off];
  c.qc=PLANE(urban_tiles,4)[off];c.uc=PLANE(urban_tiles,5)[off];
  for(int k=0;k<4;++k){size_t a=(4*t+k)*plane+idx;
   c.trl[k]=PLANE(urban_tiles,12)[a];c.tbl[k]=PLANE(urban_tiles,13)[a];c.tgl[k]=PLANE(urban_tiles,14)[a];}
  c.utype=utype;c.jmonth=jmonth;c.ta=tile.sfctmp;
  c.qa=FDIV(tile.qv1,FADD(1.0f,tile.qv1));
  c.u1=u1[idx];c.v1=v1[idx];c.ua=ucm_wind(c.u1,c.v1);
  c.ssg=tile.swdown;c.llg=tile.glw;c.rain=FMUL(FDIV(tile.rainbl,dt),3600.0f);
  c.rhoo=FDIV(tile.sfcprs,FMUL(FMUL(287.04f,c.ta),FADD(1.0f,FMUL(0.61f,c.qa))));
  c.za=FMUL(0.5f,tile.dz8w1);c.delt=dt;c.omg=hrang[idx];c.znt=tile.znt;
  ucm_floor_exchange(&tile.chs,&tile.chs2,&tile.cqs2,0);
  c.chs=tile.chs;c.chs2=tile.chs2;
  int code=ucm_urban(tab,glob,isw,c);
  if(code!=UCM_OK){ucm_flag(err,code);return;}
  real f=frc[idx],omf=FSUB(1.0f,f);
  tile.albedo=FADD(FMUL(f,c.alb),FMUL(omf,tile.albedo));
  tile.hfx=FADD(FMUL(f,c.sh),FMUL(omf,tile.hfx));
  tile.qfx=FADD(FMUL(f,c.lh_kin),FMUL(omf,tile.qfx));
  tile.lh=FADD(FMUL(f,c.lh),FMUL(omf,tile.lh));
  tile.grdflx=FADD(FMUL(f,c.g),FMUL(omf,tile.grdflx));
  tile.tsk=FADD(FMUL(f,c.ts),FMUL(omf,tile.rural_t1));
  real q1=FADD(FMUL(f,c.qs),FMUL(omf,tile.rural_q1));
  tile.qsfc=FDIV(q1,FSUB(1.0f,q1));
  ust[idx]=FADD(FMUL(f,c.ust),FMUL(omf,ust[idx]));
  tile.znt=gfk_exp(FADD(FMUL(f,gfk_log(c.znt)),FMUL(omf,gfk_log(tile.znt))));
  ucm_store_state(urban_state,idx,plane,c);
  PLANE(urban_tiles,0)[off]=c.tr;PLANE(urban_tiles,1)[off]=c.tb;
  PLANE(urban_tiles,2)[off]=c.tg;PLANE(urban_tiles,3)[off]=c.tc;
  PLANE(urban_tiles,4)[off]=c.qc;PLANE(urban_tiles,5)[off]=c.uc;
  PLANE(urban_tiles,6)[off]=c.ts;PLANE(urban_tiles,7)[off]=tile.rural_t1;
  PLANE(urban_tiles,8)[off]=c.sh;PLANE(urban_tiles,9)[off]=c.lh;
  PLANE(urban_tiles,10)[off]=c.g;PLANE(urban_tiles,11)[off]=c.rn;
  for(int k=0;k<4;++k){size_t a=(4*t+k)*plane+idx;
   PLANE(urban_tiles,12)[a]=c.trl[k];PLANE(urban_tiles,13)[a]=c.tbl[k];PLANE(urban_tiles,14)[a]=c.tgl[k];}
  s.chs=tile.chs;
 }
#endif
 carried_soilw=tile.smstav;
 real area=landusef2[off];
 tsk_mosaic[off]=tile.tsk;
 qsfc_mosaic[off]=tile.qsfc;
 canwat_mosaic[off]=tile.canwat;
 snow_mosaic[off]=tile.snow;
 snowh_mosaic[off]=tile.snowh;
 snowc_mosaic[off]=tile.snowc;
 albedo_mosaic[off]=tile.albedo;
 albbck_mosaic[off]=tile.albbck;
 emiss_mosaic[off]=tile.emiss;
 embck_mosaic[off]=tile.embck;
 znt_mosaic[off]=tile.znt;
 z0_mosaic[off]=tile.z0;
 hfx_mosaic[off]=tile.hfx;
 qfx_mosaic[off]=tile.qfx;
 lh_mosaic[off]=tile.lh;
 grdflx_mosaic[off]=tile.grdflx;
 snotime_mosaic[off]=tile.snotime;
 for(int k=0;k<4;++k)tslb_mosaic[(4*t+k)*plane+idx]=tile.tslb[k];
 for(int k=0;k<4;++k)smois_mosaic[(4*t+k)*plane+idx]=tile.smois[k];
 for(int k=0;k<4;++k)sh2o_mosaic[(4*t+k)*plane+idx]=tile.sh2o[k];
 noah_mosaic_accumulate(s,tile,avg,area);
 }
 noah_mosaic_mean(s,avg,dt,frpcpn);
 } else if(s.xland<1.5f && s.xice>=xice_threshold){
  for(int k=0;k<4;++k)s.sh2o[k]=1.0f;s.lai=0.01f;
 }
 psfc_a[idx]=s.psfc;
 sfcprs_a[idx]=s.sfcprs;
 sfctmp_a[idx]=s.sfctmp;
 qv1_a[idx]=s.qv1;
 qgh_a[idx]=s.qgh;
 dz8w1_a[idx]=s.dz8w1;
 glw_a[idx]=s.glw;
 swdown_a[idx]=s.swdown;
 rainbl_a[idx]=s.rainbl;
 sr_a[idx]=s.sr;
 chs_a[idx]=s.chs;
 cqs2_a[idx]=s.cqs2;
 chs2_a[idx]=s.chs2;
 rib_a[idx]=s.rib;
 vegfra_a[idx]=s.vegfra;
 shdmin_a[idx]=s.shdmin;
 shdmax_a[idx]=s.shdmax;
 tmn_a[idx]=s.tmn;
 xland_a[idx]=s.xland;
 xice_a[idx]=s.xice;
 snoalb_a[idx]=s.snoalb;
 embck_a[idx]=s.embck;
 tsk_a[idx]=s.tsk;
 hfx_a[idx]=s.hfx;
 qfx_a[idx]=s.qfx;
 lh_a[idx]=s.lh;
 grdflx_a[idx]=s.grdflx;
 qsfc_a[idx]=s.qsfc;
 canwat_a[idx]=s.canwat;
 snow_a[idx]=s.snow;
 snowc_a[idx]=s.snowc;
 snowh_a[idx]=s.snowh;
 albedo_a[idx]=s.albedo;
 albbck_a[idx]=s.albbck;
 emiss_a[idx]=s.emiss;
 znt_a[idx]=s.znt;
 z0_a[idx]=s.z0;
 snotime_a[idx]=s.snotime;
 lai_a[idx]=s.lai;
 smstav_a[idx]=s.smstav;
 smstot_a[idx]=s.smstot;
 sfcrunoff_a[idx]=s.sfcrunoff;
 udrunoff_a[idx]=s.udrunoff;
 acsnow_a[idx]=s.acsnow;
 acsnom_a[idx]=s.acsnom;
 snopcx_a[idx]=s.snopcx;
 potevp_a[idx]=s.potevp;
 noahres_a[idx]=s.noahres;
 chklowq_a[idx]=s.chklowq;
 for(int k=0;k<4;++k)smois_a[k*plane+idx]=s.smois[k];
 for(int k=0;k<4;++k)tslb_a[k*plane+idx]=s.tslb[k];
 for(int k=0;k<4;++k)sh2o_a[k*plane+idx]=s.sh2o[k];
 for(int k=0;k<4;++k)smcrel_a[k*plane+idx]=s.smcrel[k];
}

kernel void noah_mosaic_pack(device NoahMosaicPackResources& r [[buffer(0)]], constant NoahMosaicPackParams& p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 uint ncol=uint(p.ny)*uint(p.nx); if(gid>=ncol)return;
 r.state_fields[0*ncol+gid]=r.psfc_a[gid];
 r.state_fields[1*ncol+gid]=r.sfcprs_a[gid];
 r.state_fields[2*ncol+gid]=r.sfctmp_a[gid];
 r.state_fields[3*ncol+gid]=r.qv1_a[gid];
 r.state_fields[4*ncol+gid]=r.qgh_a[gid];
 r.state_fields[5*ncol+gid]=r.dz8w1_a[gid];
 r.state_fields[6*ncol+gid]=r.glw_a[gid];
 r.state_fields[7*ncol+gid]=r.swdown_a[gid];
 r.state_fields[8*ncol+gid]=r.rainbl_a[gid];
 r.state_fields[9*ncol+gid]=r.sr_a[gid];
 r.state_fields[10*ncol+gid]=r.chs_a[gid];
 r.state_fields[11*ncol+gid]=r.cqs2_a[gid];
 r.state_fields[12*ncol+gid]=r.chs2_a[gid];
 r.state_fields[13*ncol+gid]=r.rib_a[gid];
 r.state_fields[14*ncol+gid]=r.vegfra_a[gid];
 r.state_fields[15*ncol+gid]=r.shdmin_a[gid];
 r.state_fields[16*ncol+gid]=r.shdmax_a[gid];
 r.state_fields[17*ncol+gid]=r.tmn_a[gid];
 r.state_fields[18*ncol+gid]=r.xland_a[gid];
 r.state_fields[19*ncol+gid]=r.xice_a[gid];
 r.state_fields[20*ncol+gid]=r.snoalb_a[gid];
 r.state_fields[21*ncol+gid]=r.embck_a[gid];
 r.state_fields[22*ncol+gid]=r.tsk_a[gid];
 r.state_fields[23*ncol+gid]=r.hfx_a[gid];
 r.state_fields[24*ncol+gid]=r.qfx_a[gid];
 r.state_fields[25*ncol+gid]=r.lh_a[gid];
 r.state_fields[26*ncol+gid]=r.grdflx_a[gid];
 r.state_fields[27*ncol+gid]=r.qsfc_a[gid];
 r.state_fields[28*ncol+gid]=r.canwat_a[gid];
 r.state_fields[29*ncol+gid]=r.snow_a[gid];
 r.state_fields[30*ncol+gid]=r.snowc_a[gid];
 r.state_fields[31*ncol+gid]=r.snowh_a[gid];
 r.state_fields[32*ncol+gid]=r.albedo_a[gid];
 r.state_fields[33*ncol+gid]=r.albbck_a[gid];
 r.state_fields[34*ncol+gid]=r.emiss_a[gid];
 r.state_fields[35*ncol+gid]=r.znt_a[gid];
 r.state_fields[36*ncol+gid]=r.z0_a[gid];
 r.state_fields[37*ncol+gid]=r.snotime_a[gid];
 r.state_fields[38*ncol+gid]=r.lai_a[gid];
 r.state_fields[39*ncol+gid]=r.smstav_a[gid];
 r.state_fields[40*ncol+gid]=r.smstot_a[gid];
 r.state_fields[41*ncol+gid]=r.sfcrunoff_a[gid];
 r.state_fields[42*ncol+gid]=r.udrunoff_a[gid];
 r.state_fields[43*ncol+gid]=r.acsnow_a[gid];
 r.state_fields[44*ncol+gid]=r.acsnom_a[gid];
 r.state_fields[45*ncol+gid]=r.snopcx_a[gid];
 r.state_fields[46*ncol+gid]=r.potevp_a[gid];
 r.state_fields[47*ncol+gid]=r.noahres_a[gid];
 r.state_fields[48*ncol+gid]=r.chklowq_a[gid];
 r.state_fields[49*ncol+gid]=r.smois_a[0*ncol+gid];
 r.state_fields[50*ncol+gid]=r.smois_a[1*ncol+gid];
 r.state_fields[51*ncol+gid]=r.smois_a[2*ncol+gid];
 r.state_fields[52*ncol+gid]=r.smois_a[3*ncol+gid];
 r.state_fields[53*ncol+gid]=r.tslb_a[0*ncol+gid];
 r.state_fields[54*ncol+gid]=r.tslb_a[1*ncol+gid];
 r.state_fields[55*ncol+gid]=r.tslb_a[2*ncol+gid];
 r.state_fields[56*ncol+gid]=r.tslb_a[3*ncol+gid];
 r.state_fields[57*ncol+gid]=r.sh2o_a[0*ncol+gid];
 r.state_fields[58*ncol+gid]=r.sh2o_a[1*ncol+gid];
 r.state_fields[59*ncol+gid]=r.sh2o_a[2*ncol+gid];
 r.state_fields[60*ncol+gid]=r.sh2o_a[3*ncol+gid];
 r.state_fields[61*ncol+gid]=r.smcrel_a[0*ncol+gid];
 r.state_fields[62*ncol+gid]=r.smcrel_a[1*ncol+gid];
 r.state_fields[63*ncol+gid]=r.smcrel_a[2*ncol+gid];
 r.state_fields[64*ncol+gid]=r.smcrel_a[3*ncol+gid];
}

kernel void noah_mosaic_unpack(device NoahMosaicUnpackResources& r [[buffer(0)]], constant NoahMosaicUnpackParams& p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 uint ncol=uint(p.ny)*uint(p.nx); if(gid>=ncol)return;
 r.psfc_a[gid]=r.state_fields[0*ncol+gid];
 r.sfcprs_a[gid]=r.state_fields[1*ncol+gid];
 r.sfctmp_a[gid]=r.state_fields[2*ncol+gid];
 r.qv1_a[gid]=r.state_fields[3*ncol+gid];
 r.qgh_a[gid]=r.state_fields[4*ncol+gid];
 r.dz8w1_a[gid]=r.state_fields[5*ncol+gid];
 r.glw_a[gid]=r.state_fields[6*ncol+gid];
 r.swdown_a[gid]=r.state_fields[7*ncol+gid];
 r.rainbl_a[gid]=r.state_fields[8*ncol+gid];
 r.sr_a[gid]=r.state_fields[9*ncol+gid];
 r.chs_a[gid]=r.state_fields[10*ncol+gid];
 r.cqs2_a[gid]=r.state_fields[11*ncol+gid];
 r.chs2_a[gid]=r.state_fields[12*ncol+gid];
 r.rib_a[gid]=r.state_fields[13*ncol+gid];
 r.vegfra_a[gid]=r.state_fields[14*ncol+gid];
 r.shdmin_a[gid]=r.state_fields[15*ncol+gid];
 r.shdmax_a[gid]=r.state_fields[16*ncol+gid];
 r.tmn_a[gid]=r.state_fields[17*ncol+gid];
 r.xland_a[gid]=r.state_fields[18*ncol+gid];
 r.xice_a[gid]=r.state_fields[19*ncol+gid];
 r.snoalb_a[gid]=r.state_fields[20*ncol+gid];
 r.embck_a[gid]=r.state_fields[21*ncol+gid];
 r.tsk_a[gid]=r.state_fields[22*ncol+gid];
 r.hfx_a[gid]=r.state_fields[23*ncol+gid];
 r.qfx_a[gid]=r.state_fields[24*ncol+gid];
 r.lh_a[gid]=r.state_fields[25*ncol+gid];
 r.grdflx_a[gid]=r.state_fields[26*ncol+gid];
 r.qsfc_a[gid]=r.state_fields[27*ncol+gid];
 r.canwat_a[gid]=r.state_fields[28*ncol+gid];
 r.snow_a[gid]=r.state_fields[29*ncol+gid];
 r.snowc_a[gid]=r.state_fields[30*ncol+gid];
 r.snowh_a[gid]=r.state_fields[31*ncol+gid];
 r.albedo_a[gid]=r.state_fields[32*ncol+gid];
 r.albbck_a[gid]=r.state_fields[33*ncol+gid];
 r.emiss_a[gid]=r.state_fields[34*ncol+gid];
 r.znt_a[gid]=r.state_fields[35*ncol+gid];
 r.z0_a[gid]=r.state_fields[36*ncol+gid];
 r.snotime_a[gid]=r.state_fields[37*ncol+gid];
 r.lai_a[gid]=r.state_fields[38*ncol+gid];
 r.smstav_a[gid]=r.state_fields[39*ncol+gid];
 r.smstot_a[gid]=r.state_fields[40*ncol+gid];
 r.sfcrunoff_a[gid]=r.state_fields[41*ncol+gid];
 r.udrunoff_a[gid]=r.state_fields[42*ncol+gid];
 r.acsnow_a[gid]=r.state_fields[43*ncol+gid];
 r.acsnom_a[gid]=r.state_fields[44*ncol+gid];
 r.snopcx_a[gid]=r.state_fields[45*ncol+gid];
 r.potevp_a[gid]=r.state_fields[46*ncol+gid];
 r.noahres_a[gid]=r.state_fields[47*ncol+gid];
 r.chklowq_a[gid]=r.state_fields[48*ncol+gid];
 r.smois_a[0*ncol+gid]=r.state_fields[49*ncol+gid];
 r.smois_a[1*ncol+gid]=r.state_fields[50*ncol+gid];
 r.smois_a[2*ncol+gid]=r.state_fields[51*ncol+gid];
 r.smois_a[3*ncol+gid]=r.state_fields[52*ncol+gid];
 r.tslb_a[0*ncol+gid]=r.state_fields[53*ncol+gid];
 r.tslb_a[1*ncol+gid]=r.state_fields[54*ncol+gid];
 r.tslb_a[2*ncol+gid]=r.state_fields[55*ncol+gid];
 r.tslb_a[3*ncol+gid]=r.state_fields[56*ncol+gid];
 r.sh2o_a[0*ncol+gid]=r.state_fields[57*ncol+gid];
 r.sh2o_a[1*ncol+gid]=r.state_fields[58*ncol+gid];
 r.sh2o_a[2*ncol+gid]=r.state_fields[59*ncol+gid];
 r.sh2o_a[3*ncol+gid]=r.state_fields[60*ncol+gid];
 r.smcrel_a[0*ncol+gid]=r.state_fields[61*ncol+gid];
 r.smcrel_a[1*ncol+gid]=r.state_fields[62*ncol+gid];
 r.smcrel_a[2*ncol+gid]=r.state_fields[63*ncol+gid];
 r.smcrel_a[3*ncol+gid]=r.state_fields[64*ncol+gid];
}
