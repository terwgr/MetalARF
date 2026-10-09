// needs integration: packed-source SLUCM Noah-MP canopy body.
// Float inputs, the urban table and global arrays share float_bundle to keep
// the typed entry below the runtime's 64-resource ceiling. The canopy math is
// translated from urban_ucm.cu; central integration owns generated bindings.
#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_gfk_atan.metalh"
#include "ucm_noahmp_after_lsm_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

#define FADD(a,b) as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)))
#define FSUB(a,b) as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)))
#define FMUL(a,b) as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)))
#define FDIV(a,b) as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)))
#define FSQRT(a) as_type<float>(soft32_sqrt(as_type<uint>(a)))
static inline float gfk_exp(float x) { return as_type<float>(soft_gfk_exp(as_type<uint>(x))); }
static inline float gfk_log(float x) { return as_type<float>(soft_gfk_log(as_type<uint>(x))); }
static inline float gfk_pow(float x, float y) { return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x), as_type<uint>(y))); }

//
// Bitwise discipline (the reference is gfortran -O0 + glibc 2.43):
//  * every float op is an explicit rounding intrinsic (FADD/FSUB/FMUL/FDIV/
//    FSQRT from glibc_flt32.cuh), so nvcc cannot contract a*b+c into an FMA
//    gfortran -O0 never forms, and evaluation is Fortran's left to right;
//  * EXP/LOG/x**real are glibc's expf/logf/powf: gfk_exp/gfk_log/gfk_pow.
//    gfortran -O0 calls powf for EVERY real exponent, 2. 3. 4. included;
//    x**integer is libgcc's __powisf2 (u_powi);
//  * ATAN and LOG10: glibc 2.43's atanf and log10f are CORE-MATH, correctly
//    rounded (measured: double atan/log10 rounded once matched glibc 2.43 on
//    every normal input of a 1/97 sweep of all float32, while the fdlibm
//    atanf gpuwm carries for glibc 2.39 missed 217,462 of them);
//  * constant subexpressions gfortran folds at compile time are the folded
//    words, read off gfortran 15.2 itself (tools/urban_wrf471_oracle/
//    build_ucm.sh), never re-folded by nvcc.
//
// Two WRF undefined reads get defined behaviour (urban_ucm.py says why):
//  (a) ETR (:571) is read by SMFLX/SRT when the green-roof EPGR <= 0 on the
//      first iteration, before TRANSP wrote it: ETR = 0 at entry;
//  (b) tloc (:623-627) is written only when AHOPTION == 1 but read by the
//      IRI_SCHEME == 1 arm (:874-883): computed by the same formula
//      whenever either needs it.

#define UCM_NL 4

// ---------------------------------------------------------------------------
// folded constants (gfortran 15.2 -O0 compile-time words)
// ---------------------------------------------------------------------------
#define U_C_CC        as_type<float>(0x45A94801u)  // 2.5*10.**6./461.51
#define U_PIO2        as_type<float>(0x3FC90FD0u)  // PI/2., PI = 3.14159
#define U_TWO_ATAN1   as_type<float>(0x3FC90FDBu)  // 2.*ATAN(1.)
#define U_KARMANG     as_type<float>(0x407B22D2u)  // 0.4*9.81
#define U_19P6        as_type<float>(0x419CCCCDu)  // 9.8*2.
#define U_FR_C2       as_type<float>(0x4656DBFEu)  // 24.*3600./2./3.14159
#define U_WWST2       as_type<float>(0x3FB851ECu)  // WWST*WWST
#define U_ELFC        as_type<float>(0x3C6DDF15u)  // VKRM*BETA*G
#define U_WNEW        as_type<float>(0x3F59999Au)  // 1.-WOLD
#define U_PIHF        as_type<float>(0x3FC90FDBu)  // 3.14159265/2.
#define U_BTGH        as_type<float>(0x42112F68u)  // BTG*HPBL
#define U_SQVISC2     as_type<float>(0x478235A0u)  // SQVISC**2
#define U_TWO_THIRDS  as_type<float>(0x3F2AAAABu)  // 2./3.
#define U_49          as_type<float>(0x42440000u)  // 7.**2
#define U_1P1         as_type<float>(0x3F8CCCCDu)  // 0.55*2.0
#define U_EXP_M16     as_type<float>(0x3E4EBDF6u)  // EXP(-2.0*SHDFAC)
#define U_INV_4P1868  as_type<float>(0x3E749405u)  // 1.0/4.1868
#define U_RCP         as_type<float>(0x3E924925u)  // r_d/cp, module_model_constants
#define U_KARMAN      0.4f                          // module_model_constants KARMAN

// urban's own PARAMETERs (:326-338, :583-608)
#define U_CPCGS   0.24f
#define U_EL      583.0f
#define U_SIG     8.17e-11f
#define U_PI      3.14159f
#define U_SRATIO  0.75f
#define U_CPP     1004.5f
#define U_ELL     2.442e+06f
// green-roof PARAMETERs.  They are read from a MUTABLE static inline array, not
// written as literals, so that no expression over two of them can be
// folded at compile time.  WRF evaluates SMCMAX-SMCDRY, 1-SHDFAC and the
// rest at run time (they are DIREVAP/TRANSP/TDFCND arguments), and so must
// this kernel: NVRTC 12.9.86 -- the cupy-cuda12x of the default [gpu]
// extra -- folds __fadd_rn/__fsub_rn of two constants with round-toward-
// zero in about 2% of cases (measured: 4 of 400 adds and 9 of 400 subs;
// NVRTC 13.4.92: 0), which moved SMCMAX-SMCDRY by 1 ULP and the green-roof
// latent heat by up to 7 ULP against WRF.  An empty asm barrier does not
// stop it (ptxas folds the mov'd immediates); a load from this array does.
// Nothing writes it: its contents are these literals, in this order.
constant float u_gr_param[21] = {
    0.80f, 0.20f, 0.93f, 1.50f, 0.5e-3f, 0.329f, 0.066f, 0.084f, 0.439f, 5000.0f, 100.0f, 100.0f, 0.5f, 0.143e-4f, 3.38e-6f, 5.25f, 2.0f, -2.0f, 0.40f, 2.0e+6f, 36.0f};
#define GR_SHDFAC (u_gr_param[0])   // 0.80f
#define GR_ALBV   (u_gr_param[1])   // 0.20f
#define GR_EPSV   (u_gr_param[2])   // 0.93f
#define GR_LAI    (u_gr_param[3])   // 1.50f
#define GR_CMCMAX (u_gr_param[4])   // 0.5e-3f
#define GR_SMCREF (u_gr_param[5])   // 0.329f
#define GR_SMCDRY (u_gr_param[6])   // 0.066f
#define GR_SMCWLT (u_gr_param[7])   // 0.084f
#define GR_SMCMAX (u_gr_param[8])   // 0.439f
#define GR_RSMAX  (u_gr_param[9])   // 5000.0f
#define GR_RSMIN  (u_gr_param[10])   // 100.0f
#define GR_RGL    (u_gr_param[11])   // 100.0f
#define GR_CFACTR (u_gr_param[12])   // 0.5f
#define GR_DWSAT  (u_gr_param[13])   // 0.143e-4f
#define GR_DKSAT  (u_gr_param[14])   // 3.38e-6f
#define GR_BEXP   (u_gr_param[15])   // 5.25f
#define GR_FXEXP  (u_gr_param[16])   // 2.0f
#define GR_ZBOT   (u_gr_param[17])   // -2.0f
#define GR_QUARTZ (u_gr_param[18])   // 0.40f
#define GR_CSOIL  (u_gr_param[19])   // 2.0e+6f
#define GR_HS     (u_gr_param[20])   // 36.0f
#define GR_NROOT  2
#define GR_NGR    4

// ---------------------------------------------------------------------------
// packed parameter table: one row per UTYPE (gpuwm/core/urban_ucm.py
// UCM_TABLE_COLUMNS) and the globals (UCM_GLOBAL_LAYOUT)
// ---------------------------------------------------------------------------
#define PT_ZR 0
#define PT_Z0C 1
#define PT_Z0HC 2
#define PT_ZDC 3
#define PT_SVF 4
#define PT_R 5
#define PT_RW 6
#define PT_HGT 7
#define PT_AH 8
#define PT_ALH 9
#define PT_BETR 10
#define PT_BETB 11
#define PT_BETG 12
#define PT_CAPR 13
#define PT_CAPB 14
#define PT_CAPG 15
#define PT_AKSR 16
#define PT_AKSB 17
#define PT_AKSG 18
#define PT_ALBR 19
#define PT_ALBB 20
#define PT_ALBG 21
#define PT_EPSR 22
#define PT_EPSB 23
#define PT_EPSG 24
#define PT_Z0R 25
#define PT_Z0B 26
#define PT_Z0G 27
#define PT_Z0HB 28
#define PT_Z0HG 29
#define PT_TRLEND 30
#define PT_TBLEND 31
#define PT_TGLEND 32
#define PT_AKANDA 33
#define PT_NCOL 34

#define PG_DZR 0
#define PG_DZB 4
#define PG_DZG 8
#define PG_DZGR 12
#define PG_PORIMP 16
#define PG_DENGIMP 19
#define PG_AHDIUPRF 22
#define PG_ALHSEASON 46
#define PG_ALHDIUPRF 50
#define PG_FGR 98
#define PG_N 99

#define SW_BOUNDR 0
#define SW_BOUNDB 1
#define SW_BOUNDG 2
#define SW_CH 3
#define SW_TS 4
#define SW_AH 5
#define SW_ALH 6
#define SW_IMP 7
#define SW_IRI 8
#define SW_GR 9
#define SW_NUT 10
#define SW_N 11

// ucm_urban return codes; the launcher turns a nonzero into a refusal.
#define UCM_OK 0
#define UCM_ERR_ZA 1        // :825 ZDC+Z0C+2. >= ZA, WRF's fatal
#define UCM_ERR_UTYPE 2     // UTYPE outside the table

struct UcmCol {
    int utype, jmonth;
    float ta, qa, ua, u1, v1, ssg, llg, rain, rhoo, za, omg, delt, chs, chs2;
    float znt;
    float tr, tb, tg, tc, qc, uc;
    float trl[UCM_NL], tbl[UCM_NL], tgl[UCM_NL];
    float xxxr, xxxb, xxxg, xxxc;
    float cmr, chr, cmc, chc, cmgr, chgr;
    float cmcr, tgr, tgrl[UCM_NL], smr[UCM_NL];
    float drelr, drelb, drelg, flxhumr, flxhumb, flxhumg;
    float ts, qs, sh, lh, lh_kin, sw, alb, lw, g, rn, psim, psih, gz1oz0;
    float u10, v10, th2, q2, ust;
};

// ---------------------------------------------------------------------------
// libm and intrinsic mirrors
// ---------------------------------------------------------------------------
static inline float u_max(float a, float b) { return (a > b) ? a : b; }
static inline float u_min(float a, float b) { return (a < b) ? a : b; }

// libgcc __powisf2: gfortran -O0 lowers REAL**INTEGER to it.
static inline float u_powi(float x, int n)
{
    unsigned int m = (n < 0) ? (0u - (unsigned int)n) : (unsigned int)n;
    float y = (m % 2u) ? x : 1.0f;
    while (m >>= 1) {
        x = FMUL(x, x);
        if (m % 2u) y = FMUL(y, x);
    }
    return (n < 0) ? FDIV(1.0f, y) : y;
}

// glibc 2.43 atanf / log10f are correctly rounded; see the header.
static inline float u_atan(float x) { return as_type<float>(soft_gfk_atanf(as_type<uint>(x))); }
static inline float u_log10(float x) { return log10(x); }

// ES = 6.11*EXP( (2.5*10.**6./461.51)*(T-273.15)/(273.15*T) )
static inline float u_es(float t)
{
    return FMUL(6.11f, gfk_exp(FDIV(FMUL(U_C_CC, FSUB(t, 273.15f)), FMUL(273.15f, t))));
}

// ---------------------------------------------------------------------------
// SFCDIF_URB (3116-3324), called with 10 arguments: VEGFRAC and ZT_OUT absent,
// ILECH = 0 so only Paulson's functions.
// ---------------------------------------------------------------------------
static inline float u_pspmu(float xx)
{
    // -2.*log((XX+1.)*0.5) - log((XX*XX+1.)*0.5) + 2.*ATAN(XX) - PIHF
    float t = FSUB(0.0f, FMUL(2.0f, gfk_log(FMUL(FADD(xx, 1.0f), 0.5f))));
    t = FSUB(t, gfk_log(FMUL(FADD(FMUL(xx, xx), 1.0f), 0.5f)));
    t = FADD(t, FMUL(2.0f, u_atan(xx)));
    return FSUB(t, U_PIHF);
}
static inline float u_psphu(float xx)
{
    return FSUB(0.0f, FMUL(2.0f, gfk_log(FMUL(FADD(FMUL(xx, xx), 1.0f), 0.5f))));
}

static inline void u_sfcdif_urb(float zlm, float z0, float thz0, float thlm,
                             float sfcspd, float akanda, thread float& akms,
                             thread float& akhs, thread float& rlmo, thread float& cd)
{
    const float excm = 0.001f, wold = 0.15f, vkrm = 0.40f;
    const float epsu2 = 1.e-4f, epsust = 0.07f, ztmin = -5.0f, ztmax = 1.0f;
    float zu = z0;
    float rdz = FDIV(1.0f, zlm);
    float cxch = FMUL(excm, rdz);
    float dthv = FSUB(thlm, thz0);
    float du2 = u_max(FMUL(sfcspd, sfcspd), epsu2);
    float wstar2;
    float bah = FMUL(FMUL(U_BTGH, akhs), dthv);
    if (bah != 0.0f) {
        wstar2 = FMUL(U_WWST2, gfk_pow(fabs(bah), U_TWO_THIRDS));
    } else {
        wstar2 = 0.0f;
    }
    float ustar = u_max(FSQRT(FMUL(akms, FSQRT(FADD(du2, wstar2)))), epsust);
    float zt = FMUL(gfk_exp(FSUB(2.0f, FMUL(akanda,
                   gfk_pow(FMUL(FMUL(U_SQVISC2, ustar), z0), 0.25f)))), z0);
    float zslu = FADD(zlm, zu);
    float zslt = FADD(zlm, zt);
    float rlogu = gfk_log(FDIV(zslu, zu));
    float rlogt = gfk_log(FDIV(zslt, zt));
    rlmo = FDIV(FMUL(FMUL(U_ELFC, akhs), dthv), u_powi(ustar, 3));
    float simm = 0.0f, simh = 0.0f;
    for (int itr = 1; itr <= 5; ++itr) {
        float zetalt = u_max(FMUL(zslt, rlmo), ztmin);
        rlmo = FDIV(zetalt, zslt);
        float zetalu = FMUL(zslu, rlmo);
        float zetau = FMUL(zu, rlmo);
        float zetat = FMUL(zt, rlmo);
        if (rlmo < 0.0f) {
            float xlu4 = FSUB(1.0f, FMUL(16.0f, zetalu));
            float xlt4 = FSUB(1.0f, FMUL(16.0f, zetalt));
            float xu4 = FSUB(1.0f, FMUL(16.0f, zetau));
            float xt4 = FSUB(1.0f, FMUL(16.0f, zetat));
            float xlu = FSQRT(FSQRT(xlu4));
            float xlt = FSQRT(FSQRT(xlt4));
            float xu = FSQRT(FSQRT(xu4));
            float xt = FSQRT(FSQRT(xt4));
            float psmz = u_pspmu(xu);
            simm = FADD(FSUB(u_pspmu(xlu), psmz), rlogu);
            float pshz = u_psphu(xt);
            simh = FADD(FSUB(u_psphu(xlt), pshz), rlogt);
        } else {
            zetalu = u_min(zetalu, ztmax);
            zetalt = u_min(zetalt, ztmax);
            float psmz = FMUL(5.0f, zetau);
            simm = FADD(FSUB(FMUL(5.0f, zetalu), psmz), rlogu);
            float pshz = FMUL(5.0f, zetat);
            simh = FADD(FSUB(FMUL(5.0f, zetalt), pshz), rlogt);
        }
        ustar = u_max(FSQRT(FMUL(akms, FSQRT(FADD(du2, wstar2)))), epsust);
        zt = FMUL(gfk_exp(FSUB(2.0f, FMUL(akanda,
                gfk_pow(FMUL(FMUL(U_SQVISC2, ustar), z0), 0.25f)))), z0);
        zslt = FADD(zlm, zt);
        rlogt = gfk_log(FDIV(zslt, zt));
        float ustark = FMUL(ustar, vkrm);
        akms = u_max(FDIV(ustark, simm), cxch);
        akhs = u_max(FDIV(ustark, simh), cxch);
        bah = FMUL(FMUL(U_BTGH, akhs), dthv);
        if (bah != 0.0f) {
            wstar2 = FMUL(U_WWST2, gfk_pow(fabs(bah), U_TWO_THIRDS));
        } else {
            wstar2 = 0.0f;
        }
        float rlmn = FDIV(FMUL(FMUL(U_ELFC, akhs), dthv), u_powi(ustar, 3));
        float rlma = FADD(FMUL(rlmo, wold), FMUL(rlmn, U_WNEW));
        rlmo = rlma;
    }
    cd = FDIV(FMUL(ustar, ustar), u_powi(sfcspd, 2));
}

// ---------------------------------------------------------------------------
// mos (1698-1781), CH_SCHEME = 1
// ---------------------------------------------------------------------------
static inline void u_mos(thread float& xxx, thread float& alpha, thread float& cd, float b1,
                      thread float& rib, float z, float z0, float ua, float ta,
                      float tsf, float rho)
{
    float psim = 0.0f, psih = 0.0f;
    if (rib <= -15.0f) rib = -15.0f;
    float lzz = gfk_log(FDIV(FADD(z, z0), z0));
    if (rib < 0.0f) {
        for (int newt = 1; newt <= 10; ++newt) {
            if (xxx >= 0.0f) xxx = -1.e-3f;
            float xxx0 = FDIV(FMUL(xxx, z0), FADD(z, z0));
            float a = FSUB(1.0f, FMUL(16.0f, xxx));
            float a0 = FSUB(1.0f, FMUL(16.0f, xxx0));
            float x = gfk_pow(a, 0.25f);
            float x0 = gfk_pow(a0, 0.25f);
            // PSIM=ALOG((Z+Z0)/Z0) -ALOG((X+1.)**2.*(X**2.+1.)) +2.*ATAN(X)
            //      +ALOG((X0+1.)**2.*(X0**2.+1.)) -2.*ATAN(X0)
            float t = FSUB(lzz, gfk_log(FMUL(gfk_pow(FADD(x, 1.0f), 2.0f),
                                             FADD(gfk_pow(x, 2.0f), 1.0f))));
            t = FADD(t, FMUL(2.0f, u_atan(x)));
            t = FADD(t, gfk_log(FMUL(gfk_pow(FADD(x0, 1.0f), 2.0f),
                                     FADD(gfk_pow(x0, 2.0f), 1.0f))));
            psim = FSUB(t, FMUL(2.0f, u_atan(x0)));
            // PSIH=ALOG((Z+Z0)/Z0)+0.4*B1 -2.*ALOG(SQRT(1.-16.*XXX)+1.)
            //      +2.*ALOG(SQRT(1.-16.*XXX0)+1.)
            float h = FADD(lzz, FMUL(0.4f, b1));
            h = FSUB(h, FMUL(2.0f, gfk_log(FADD(FSQRT(a), 1.0f))));
            psih = FADD(h, FMUL(2.0f, gfk_log(FADD(FSQRT(a0), 1.0f))));
            float dpsim = FSUB(FDIV(gfk_pow(a, -0.25f), xxx),
                               FDIV(gfk_pow(a0, -0.25f), xxx));
            float dpsih = FSUB(FDIV(FDIV(1.0f, FSQRT(a)), xxx),
                               FDIV(FDIV(1.0f, FSQRT(a0)), xxx));
            float f = FSUB(FDIV(FMUL(rib, gfk_pow(psim, 2.0f)), psih), xxx);
            float inner = FSUB(FMUL(FMUL(FMUL(2.0f, dpsim), psim), psih),
                               FMUL(dpsih, gfk_pow(psim, 2.0f)));
            float df = FSUB(FDIV(FMUL(rib, inner), gfk_pow(psih, 2.0f)), 1.0f);
            float xxxp = xxx;
            xxx = FSUB(xxxp, FDIV(f, df));
            if (xxx <= -10.0f) xxx = -10.0f;
        }
    } else if (rib >= 0.142857f) {
        xxx = 0.714f;
        psim = FADD(lzz, FMUL(7.0f, xxx));
        psih = FADD(psim, FMUL(0.4f, b1));
    } else {
        float al = lzz;
        float xkb = FMUL(0.4f, b1);
        float dd = FADD(FSUB(0.0f, FMUL(FMUL(FMUL(FMUL(4.0f, rib), 7.0f), xkb), al)),
                        gfk_pow(FADD(al, xkb), 2.0f));
        if (dd <= 0.0f) dd = 0.0f;
        float num = FSUB(FSUB(FADD(al, xkb), FMUL(FMUL(FMUL(2.0f, rib), 7.0f), al)),
                         FSQRT(dd));
        xxx = FDIV(num, FMUL(2.0f, FSUB(FMUL(rib, U_49), 7.0f)));
        psim = FADD(lzz, FMUL(7.0f, u_min(xxx, 0.714f)));
        psih = FADD(psim, FMUL(0.4f, b1));
    }
    float us = FDIV(FMUL(0.4f, ua), psim);
    if (us <= 0.01f) us = 0.01f;
    cd = FDIV(FMUL(us, us), gfk_pow(ua, 2.0f));
    alpha = FDIV(FMUL(FMUL(FMUL(rho, U_CPCGS), 0.4f), us), psih);
}

// ---------------------------------------------------------------------------
// multi_layer (1861-1936), KM = 4
// ---------------------------------------------------------------------------
static inline void u_multi_layer(int bound, float g0, float cap, float aks,
                              thread float* tsl, device const float* dz, float delt,
                              float tslend)
{
    const int km = UCM_NL;
    float a[UCM_NL], b[UCM_NL], c[UCM_NL], d[UCM_NL], x[UCM_NL], p[UCM_NL], q[UCM_NL];
    float dzend = dz[km - 1];
    float two_aks = FMUL(2.0f, aks);
    a[0] = 0.0f;
    b[0] = FADD(FDIV(FMUL(cap, dz[0]), delt), FDIV(two_aks, FADD(dz[0], dz[1])));
    c[0] = FSUB(0.0f, FDIV(two_aks, FADD(dz[0], dz[1])));
    d[0] = FADD(FMUL(FDIV(FMUL(cap, dz[0]), delt), tsl[0]), g0);
    for (int k = 1; k < km - 1; ++k) {
        a[k] = FSUB(0.0f, FDIV(two_aks, FADD(dz[k - 1], dz[k])));
        b[k] = FADD(FADD(FDIV(FMUL(cap, dz[k]), delt),
                         FDIV(two_aks, FADD(dz[k - 1], dz[k]))),
                    FDIV(two_aks, FADD(dz[k], dz[k + 1])));
        c[k] = FSUB(0.0f, FDIV(two_aks, FADD(dz[k], dz[k + 1])));
        d[k] = FMUL(FDIV(FMUL(cap, dz[k]), delt), tsl[k]);
    }
    int n = km - 1;
    a[n] = FSUB(0.0f, FDIV(two_aks, FADD(dz[n - 1], dz[n])));
    if (bound == 1) {
        b[n] = FADD(FDIV(FMUL(cap, dz[n]), delt), FDIV(two_aks, FADD(dz[n - 1], dz[n])));
        c[n] = 0.0f;
        d[n] = FMUL(FDIV(FMUL(cap, dz[n]), delt), tsl[n]);
    } else {
        b[n] = FADD(FADD(FDIV(FMUL(cap, dz[n]), delt),
                         FDIV(two_aks, FADD(dz[n - 1], dz[n]))),
                    FDIV(two_aks, FADD(dz[n], dzend)));
        c[n] = 0.0f;
        d[n] = FADD(FMUL(FDIV(FMUL(cap, dz[n]), delt), tsl[n]),
                    FDIV(FMUL(two_aks, tslend), FADD(dz[n], dzend)));
    }
    p[0] = FSUB(0.0f, FDIV(c[0], b[0]));
    q[0] = FDIV(d[0], b[0]);
    for (int k = 1; k < km; ++k) {
        float den = FADD(FMUL(a[k], p[k - 1]), b[k]);
        p[k] = FSUB(0.0f, FDIV(c[k], den));
        q[k] = FDIV(FADD(FSUB(0.0f, FMUL(a[k], q[k - 1])), d[k]), den);
    }
    x[km - 1] = q[km - 1];
    for (int k = km - 2; k >= 0; --k) x[k] = FADD(FMUL(p[k], x[k + 1]), q[k]);
    for (int k = 0; k < km; ++k) tsl[k] = x[k];
}

// force_restore (3058-3069)
static inline float u_force_restore(float cap, float aks, float delt, float s,
                                 float r, float h, float le, float tslend,
                                 float tsp)
{
    float c2 = U_FR_C2;
    float c1 = FSQRT(FMUL(FMUL(FMUL(0.5f, c2), cap), aks));
    float bal = FSUB(FSUB(FADD(s, r), h), le);
    return FADD(tsp, FMUL(delt, FSUB(FDIV(bal, c1), FDIV(FSUB(tsp, tslend), c2))));
}

// ---------------------------------------------------------------------------
// green-roof hydrology and heat (3330-4176)
// ---------------------------------------------------------------------------
static inline float u_direvap(float etp, float smc, float shdfac, float smcmax,
                           float smcdry, float fxexp)
{
    float sratio = FDIV(FSUB(smc, smcdry), FSUB(smcmax, smcdry));
    float fx;
    if (sratio > 0.0f) {
        fx = gfk_pow(sratio, fxexp);
        fx = u_max(u_min(fx, 1.0f), 0.0f);
    } else {
        fx = 0.0f;
    }
    return FMUL(FMUL(FMUL(fx, FSUB(1.0f, shdfac)), etp), 0.001f);
}

// TRANSP (3355-3458), NROOT = 2, NSOIL = NGR = 4
static inline void u_transp(thread float& ett, thread float* et, thread float& ec, float shdfac,
                         float etp1, float cmc, float cfactr, float cmcmax,
                         float lai, float rsmin, float rsmax, float rgl,
                         float sx, float ts, float ta, float qa,
                         thread const float* smc, float smcwlt, float smcref,
                         float cpp, float ch, float epsv, float delt,
                         device const float* dzvr, thread const float* zsoil, float hs)
{
    float gx[GR_NROOT], part[GR_NROOT];
    float slv = 2.501e+6f;
    float sigma = 5.67e-8f;
    ett = 0.0f;
    for (int k = 0; k < GR_NGR; ++k) et[k] = 0.0f;
    float ff = FDIV(FMUL(FMUL(FMUL(U_1P1, sx), 697.7f), 60.0f), FMUL(rgl, lai));
    float rcs = FDIV(FADD(ff, FDIV(rsmin, rsmax)), FADD(1.0f, ff));
    rcs = u_max(rcs, 0.0001f);
    float rct = FSUB(1.0f, FMUL(0.0016f, gfk_pow(FSUB(298.0f, ta), 2.0f)));
    rct = u_max(rct, 0.0001f);
    float ea = u_es(ta);
    float ws = FDIV(FMUL(0.622f, ea), 1013.0f);
    float rcq = FDIV(1.0f, FADD(1.0f, FMUL(hs, FSUB(ws, qa))));
    rcq = u_max(rcq, 0.01f);
    float rcsoil = 0.0f;
    for (int k = 0; k < GR_NROOT; ++k) {
        gx[k] = FDIV(FSUB(smc[k], smcwlt), FSUB(smcref, smcwlt));
        if (gx[k] > 1.0f) gx[k] = 1.0f;
        if (gx[k] < 0.0f) gx[k] = 0.0f;
        part[k] = FMUL(FDIV(FSUB(0.0f, dzvr[k]), zsoil[2]), gx[k]);
    }
    float sgx = 0.0f;
    for (int k = 0; k < GR_NROOT; ++k) {
        sgx = FADD(sgx, gx[k]);
        rcsoil = FADD(rcsoil, part[k]);
    }
    sgx = FDIV(sgx, (float)GR_NROOT);
    rcsoil = u_max(rcsoil, 0.0001f);
    float rc = FDIV(rsmin, FMUL(FMUL(FMUL(FMUL(lai, rcs), rct), rcq), rcsoil));
    float desdt = FDIV(FDIV(FDIV(FDIV(FMUL(FMUL(0.622f, slv), ea), 461.51f), ta), ta), 1013.0f);
    float delta = FMUL(FDIV(slv, cpp), desdt);
    float rr = FADD(FDIV(FMUL(FDIV(FMUL(FMUL(FMUL(4.0f, epsv), sigma), 287.04f), cpp),
                              gfk_pow(ta, 4.0f)),
                         FMUL(ts, ch)), 1.0f);
    float pc = FDIV(FADD(rr, delta), FADD(FMUL(rr, FADD(1.0f, FMUL(rc, ch))), delta));
    float ett1;
    if (cmc != 0.0f) {
        ett1 = FMUL(FMUL(FMUL(FMUL(shdfac, pc), etp1),
                         FSUB(1.0f, gfk_pow(FDIV(cmc, cmcmax), cfactr))), 0.001f);
    } else {
        ett1 = FMUL(FMUL(FMUL(shdfac, pc), etp1), 0.001f);
    }
    float denom = 0.0f;
    for (int k = 0; k < GR_NROOT; ++k) {
        float rtx = FSUB(FADD(FDIV(FSUB(0.0f, dzvr[k]), zsoil[2]), gx[k]), sgx);
        gx[k] = FMUL(gx[k], u_max(rtx, 0.0f));
        denom = FADD(denom, gx[k]);
    }
    if (denom <= 0.0f) denom = 1.0f;
    for (int k = 0; k < GR_NROOT; ++k) {
        et[k] = FDIV(FMUL(ett1, gx[k]), denom);
        ett = FADD(ett, et[k]);
    }
    if (cmc > 0.0f) {
        ec = FMUL(FMUL(FMUL(shdfac, gfk_pow(FDIV(cmc, cmcmax), cfactr)), etp1), 0.001f);
    } else {
        ec = 0.0f;
    }
    float cmc2ms = FDIV(cmc, delt);
    ec = u_min(cmc2ms, ec);
}

static inline void u_wdfcnd(thread float& wdf, thread float& wcnd, float smc, float smcmax,
                         float bexp, float dksat, float dwsat)
{
    float factr2 = FDIV(smc, smcmax);
    float expon = FADD(bexp, 2.0f);
    wdf = FMUL(dwsat, gfk_pow(factr2, expon));
    expon = FADD(FMUL(2.0f, bexp), 3.0f);
    wcnd = FMUL(dksat, gfk_pow(factr2, expon));
}

// ROSR12 (3774-3807) over n equations
static inline void u_rosr12(thread float* p, thread const float* a, thread const float* b, thread float* c,
                         thread const float* d, thread float* delta, int n)
{
    c[n - 1] = 0.0f;
    p[0] = FSUB(0.0f, FDIV(c[0], b[0]));
    delta[0] = FDIV(d[0], b[0]);
    for (int k = 1; k < n; ++k) {
        float inv = FDIV(1.0f, FADD(b[k], FMUL(a[k], p[k - 1])));
        p[k] = FSUB(0.0f, FMUL(c[k], inv));
        delta[k] = FMUL(FSUB(d[k], FMUL(a[k], delta[k - 1])), inv);
    }
    p[n - 1] = delta[n - 1];
    for (int k = 2; k <= n; ++k) {
        int kk = n - k;          // 0-based NSOIL-K+1
        p[kk] = FADD(FMUL(p[kk], p[kk + 1]), delta[kk]);
    }
}

// SRT (3516-3627) with NSOIL = 4
static inline void u_srt(thread float* rhstt, float edir, thread const float* et, thread const float* smcp,
                      float pcpdrp, thread const float* zsoil, float dwsat, float dksat,
                      float smcmax, float bexp, thread float& runoff1, thread float& runoff2,
                      float dt, float smcwlt, thread float* ai, thread float* bi, thread float* ci)
{
    const int nsoil = GR_NGR;
    float pddum = pcpdrp;
    runoff1 = 0.0f;
    float par = 2.0e-6f;
    float wdf, wcnd;
    if (pcpdrp != 0.0f) {
        float ddmax[3];
        float smcav = FSUB(smcmax, smcwlt);
        ddmax[0] = FMUL(FSUB(0.0f, zsoil[0]), smcav);
        ddmax[0] = FMUL(ddmax[0], FSUB(1.0f, FDIV(FSUB(smcp[0], smcwlt), smcav)));
        ddmax[1] = FMUL(FSUB(zsoil[0], zsoil[1]), smcav);
        ddmax[1] = FMUL(ddmax[1], FSUB(1.0f, FDIV(FSUB(smcp[1], smcwlt), smcav)));
        ddmax[2] = FMUL(FSUB(zsoil[1], zsoil[2]), smcav);
        ddmax[2] = FMUL(ddmax[2], FSUB(1.0f, FDIV(FSUB(smcp[2], smcwlt), smcav)));
        float dd = FADD(FADD(ddmax[0], ddmax[1]), ddmax[2]);
        float dt1 = FDIV(dt, 86400.0f);
        float kdt = FDIV(FMUL(3.0f, dksat), par);
        float val = FSUB(1.0f, gfk_exp(FSUB(0.0f, FMUL(kdt, dt1))));
        float ddt = FMUL(dd, val);
        float px = FMUL(pcpdrp, dt);
        if (px < 0.0f) px = 0.0f;
        float infmax = FDIV(FMUL(px, FDIV(ddt, FADD(px, ddt))), dt);
        float mxsmc = smcp[0];
        u_wdfcnd(wdf, wcnd, mxsmc, smcmax, bexp, dksat, dwsat);
        infmax = u_max(infmax, wcnd);
        infmax = u_min(infmax, FDIV(px, dt));
        if (pcpdrp > infmax) {
            runoff1 = FSUB(pcpdrp, infmax);
            pddum = infmax;
        }
    }
    u_wdfcnd(wdf, wcnd, smcp[0], smcmax, bexp, dksat, dwsat);
    float ddz = FDIV(1.0f, FSUB(0.0f, FMUL(0.5f, zsoil[1])));
    ai[0] = 0.0f;
    bi[0] = FDIV(FMUL(wdf, ddz), FSUB(0.0f, zsoil[0]));
    ci[0] = FSUB(0.0f, bi[0]);
    float dsmdz = FDIV(FSUB(smcp[0], smcp[1]), FSUB(0.0f, FMUL(0.5f, zsoil[1])));
    rhstt[0] = FDIV(FADD(FADD(FSUB(FADD(FMUL(wdf, dsmdz), wcnd), pddum), edir), et[0]),
                    zsoil[0]);
    float ddz2 = 0.0f;
    float wdf2 = 0.0f, wcnd2 = 0.0f, dsmdz2 = 0.0f;
    for (int k = 2; k <= nsoil - 1; ++k) {         // Fortran K
        int i = k - 1;                              // 0-based
        float denom2 = FSUB(zsoil[i - 1], zsoil[i]);
        if (k != nsoil - 1) {
            float mxsmc2 = smcp[i];
            u_wdfcnd(wdf2, wcnd2, mxsmc2, smcmax, bexp, dksat, dwsat);
            float denom = FSUB(zsoil[i - 1], zsoil[i + 1]);
            dsmdz2 = FDIV(FSUB(smcp[i], smcp[i + 1]), FMUL(denom, 0.5f));
            ddz2 = FDIV(2.0f, denom);
            ci[i] = FSUB(0.0f, FDIV(FMUL(wdf2, ddz2), denom2));
        } else {
            u_wdfcnd(wdf2, wcnd2, smcp[nsoil - 2], smcmax, bexp, dksat, dwsat);
            dsmdz2 = 0.0f;
            ci[i] = 0.0f;
        }
        float numer = FADD(FSUB(FSUB(FMUL(wdf2, dsmdz2), FMUL(wdf, dsmdz)), wcnd), et[i]);
        rhstt[i] = FDIV(numer, FSUB(0.0f, denom2));
        ai[i] = FSUB(0.0f, FDIV(FMUL(wdf, ddz), denom2));
        bi[i] = FSUB(0.0f, FADD(ai[i], ci[i]));
        if (k == nsoil - 1) runoff2 = 0.0f;
        if (k != nsoil - 1) {
            wdf = wdf2;
            wcnd = wcnd2;
            dsmdz = dsmdz2;
            ddz = ddz2;
        }
    }
}

// SSTEP (3630-3716)
static inline void u_sstep(thread const float* smcp, thread float* smc, float cmcp, thread float& cmc,
                        thread float* rhstt, float rhsct, float dt, float smcmax,
                        float cmcmax, thread float& runoff3, thread const float* zsoil,
                        thread float* ai, thread float* bi, thread float* ci)
{
    const int n = GR_NGR - 1;
    float rhsttin[GR_NGR], ciin[GR_NGR];
    for (int k = 0; k < n; ++k) {
        rhstt[k] = FMUL(rhstt[k], dt);
        ai[k] = FMUL(ai[k], dt);
        bi[k] = FADD(1.0f, FMUL(bi[k], dt));
        ci[k] = FMUL(ci[k], dt);
    }
    for (int k = 0; k < n; ++k) rhsttin[k] = rhstt[k];
    for (int k = 0; k < n; ++k) ciin[k] = ci[k];
    u_rosr12(ci, ai, bi, ciin, rhsttin, rhstt, n);
    float wplus = 0.0f;
    runoff3 = 0.0f;
    float ddz = FSUB(0.0f, zsoil[0]);
    for (int k = 0; k < n; ++k) {
        if (k != 0) ddz = FSUB(zsoil[k - 1], zsoil[k]);
        float stot = FADD(FADD(smcp[k], ci[k]), FDIV(wplus, ddz));
        if (stot > smcmax) {
            if (k == 0) {
                ddz = FSUB(0.0f, zsoil[0]);
            } else {
                ddz = FADD(FSUB(0.0f, zsoil[k]), zsoil[k - 1]);
            }
            wplus = FMUL(FSUB(stot, smcmax), ddz);
        } else {
            wplus = 0.0f;
        }
        smc[k] = u_max(u_min(stot, smcmax), 0.066f);
    }
    runoff3 = wplus;
    cmc = FADD(cmcp, FMUL(dt, rhsct));
    if (cmc < 1.e-20f) cmc = 0.0f;
    cmc = u_min(cmc, cmcmax);
}

// SMFLX (3463-3513).  PRCP1 is mm/h.
static inline void u_smflx(thread const float* smcp, thread float* smc, float cmcp, thread float& cmc,
                        float dt, float prcp1, thread const float* zsoil, float smcmax,
                        float bexp, float smcwlt, float dksat, float dwsat,
                        float shdfac, float cmcmax, thread float& runoff1,
                        thread float& runoff2, thread float& runoff3, float edir, float ec,
                        thread const float* et, thread float& drip)
{
    float ai[GR_NGR], bi[GR_NGR], ci[GR_NGR], rhstt[GR_NGR];
    for (int k = 0; k < GR_NGR; ++k) { ai[k] = 0.0f; bi[k] = 0.0f; ci[k] = 0.0f; rhstt[k] = 0.0f; }
    float rhsct = FSUB(FDIV(FMUL(FMUL(shdfac, prcp1), 0.001f), 3600.0f), ec);
    drip = 0.0f;
    float trhsct = FMUL(dt, rhsct);
    float excess = FADD(cmcp, trhsct);
    if (excess > cmcmax) drip = FSUB(excess, cmcmax);
    float pcpdrp = FADD(FDIV(FMUL(FMUL(FSUB(1.0f, shdfac), prcp1), 0.001f), 3600.0f),
                        FDIV(drip, dt));
    u_srt(rhstt, edir, et, smcp, pcpdrp, zsoil, dwsat, dksat, smcmax, bexp,
          runoff1, runoff2, dt, smcwlt, ai, bi, ci);
    u_sstep(smcp, smc, cmcp, cmc, rhstt, rhsct, dt, smcmax, cmcmax, runoff3,
            zsoil, ai, bi, ci);
}

// TDFCND (4098-4176)
static inline float u_tdfcnd(float smc, float qz, float smcmax)
{
    float satratio = FDIV(smc, smcmax);
    float thkw = 0.57f;
    float thko = 2.0f;
    float thkqtz = 7.7f;
    float thks = FMUL(gfk_pow(thkqtz, qz), gfk_pow(thko, FSUB(1.0f, qz)));
    float thksat = FMUL(gfk_pow(thks, FSUB(1.0f, smcmax)), gfk_pow(thkw, smcmax));
    float gammd = FMUL(FSUB(1.0f, smcmax), 2700.0f);
    float thkdry = FDIV(FADD(FMUL(0.135f, gammd), 64.7f),
                        FSUB(2700.0f, FMUL(0.947f, gammd)));
    float ake;
    if (satratio > 0.1f) {
        ake = FADD(u_log10(satratio), 1.0f);
    } else {
        ake = 0.0f;
    }
    return FADD(FMUL(ake, FSUB(thksat, thkdry)), thkdry);
}

// HRT (3864-4006) + HSTEP (4009-4053) + SHFLX (3810-3855), NSOIL = 4.  The
// TBND calls feed only TBK, which no statement reads, so they are omitted.
static inline void u_shflx(thread float* stc, thread const float* smc, float smcmax, float yy,
                        float zz1, thread const float* zsoil, float tbot, float zbot,
                        float dt, float df1, float quartz, float csoil,
                        float capr)
{
    const int nsoil = GR_NGR;
    const float cair = 1004.0f, ch2o = 4.2e6f;
    float ai[GR_NGR], bi[GR_NGR], ci[GR_NGR], rhsts[GR_NGR];
    float hcpct = FADD(FADD(FMUL(smc[0], ch2o), FMUL(FSUB(1.0f, smcmax), csoil)),
                       FMUL(FSUB(smcmax, smc[0]), cair));
    float ddz = FDIV(1.0f, FSUB(0.0f, FMUL(0.5f, zsoil[1])));
    ai[0] = 0.0f;
    ci[0] = FDIV(FMUL(df1, ddz), FMUL(zsoil[0], hcpct));
    bi[0] = FADD(FSUB(0.0f, ci[0]),
                 FDIV(df1, FMUL(FMUL(FMUL(FMUL(0.5f, zsoil[0]), zsoil[0]), hcpct), zz1)));
    float dtsdz = FDIV(FSUB(stc[0], stc[1]), FSUB(0.0f, FMUL(0.5f, zsoil[1])));
    float ssoil = FDIV(FMUL(df1, FSUB(stc[0], yy)), FMUL(FMUL(0.5f, zsoil[0]), zz1));
    float denom = FMUL(zsoil[0], hcpct);
    rhsts[0] = FDIV(FSUB(FMUL(df1, dtsdz), ssoil), denom);
    float ddz2 = 0.0f;
    float df1n = df1;
    for (int k = 2; k <= nsoil; ++k) {
        int i = k - 1;
        float df1k, dtsdz2;
        if (k < nsoil - 1 || k == nsoil - 1) {
            hcpct = FADD(FADD(FMUL(smc[i], ch2o), FMUL(FSUB(1.0f, smcmax), csoil)),
                         FMUL(FSUB(smcmax, smc[i]), cair));
            df1k = u_tdfcnd(smc[i], quartz, smcmax);
            denom = FMUL(0.5f, FSUB(zsoil[i - 1], zsoil[i + 1]));
            dtsdz2 = FDIV(FSUB(stc[i], stc[i + 1]), denom);
            ddz2 = FDIV(2.0f, FSUB(zsoil[i - 1], zsoil[i + 1]));
            ci[i] = FSUB(0.0f, FDIV(FMUL(df1k, ddz2),
                                    FMUL(FSUB(zsoil[i - 1], zsoil[i]), hcpct)));
        } else {
            hcpct = FMUL(FMUL(capr, 4.1868f), 1.e6f);
            df1k = 3.24f;
            denom = FSUB(FMUL(0.5f, FADD(zsoil[i - 1], zsoil[i])), zbot);
            dtsdz2 = FDIV(FSUB(stc[i], tbot), denom);
            ci[i] = 0.0f;
        }
        denom = FMUL(FSUB(zsoil[i], zsoil[i - 1]), hcpct);
        rhsts[i] = FDIV(FSUB(FMUL(df1k, dtsdz2), FMUL(df1n, dtsdz)), denom);
        ai[i] = FSUB(0.0f, FDIV(FMUL(df1n, ddz), FMUL(FSUB(zsoil[i - 1], zsoil[i]), hcpct)));
        bi[i] = FSUB(0.0f, FADD(ai[i], ci[i]));
        df1n = df1k;
        dtsdz = dtsdz2;
        ddz = ddz2;
    }
    // HSTEP
    float rhstsin[GR_NGR], ciin[GR_NGR];
    for (int k = 0; k < nsoil; ++k) {
        rhsts[k] = FMUL(rhsts[k], dt);
        ai[k] = FMUL(ai[k], dt);
        bi[k] = FADD(1.0f, FMUL(bi[k], dt));
        ci[k] = FMUL(ci[k], dt);
    }
    for (int k = 0; k < nsoil; ++k) rhstsin[k] = rhsts[k];
    for (int k = 0; k < nsoil; ++k) ciin[k] = ci[k];
    u_rosr12(ci, ai, bi, ciin, rhstsin, rhsts, nsoil);
    for (int k = 0; k < nsoil; ++k) stc[k] = FADD(stc[k], ci[k]);
    // SHFLX's T1 and SSOIL are dead at the call site (:1221-1224).
}

// ---------------------------------------------------------------------------
// urban (305-1692)
// ---------------------------------------------------------------------------
static inline int ucm_urban(device const float*  tab,
                         device const float*  glob,
                         device const int*  isw, thread UcmCol& c)
{
    if (c.utype < 1 || c.utype > isw[SW_NUT]) return UCM_ERR_UTYPE;
    device const float* pt = tab + (uint)(c.utype - 1) * PT_NCOL;
    const int boundr = isw[SW_BOUNDR], boundb = isw[SW_BOUNDB], boundg = isw[SW_BOUNDG];
    const int ch_scheme = isw[SW_CH], ts_scheme = isw[SW_TS];
    const int ahoption = isw[SW_AH], alhoption = isw[SW_ALH];
    const int imp_scheme = isw[SW_IMP], iri_scheme = isw[SW_IRI];
    const int groption = isw[SW_GR];
    const float fgr = glob[PG_FGR];
    device const float* dzr = glob + PG_DZR;
    device const float* dzb = glob + PG_DZB;
    device const float* dzg = glob + PG_DZG;
    device const float* dzgr = glob + PG_DZGR;
    device const float* porimp = glob + PG_PORIMP;
    device const float* dengimp = glob + PG_DENGIMP;

    // ETR: WRF reads it undefined on the dew arm; defined as 0 (header (a)).
    float etr[UCM_NL];
    for (int k = 0; k < UCM_NL; ++k) etr[k] = 0.0f;

    // tloc/tloc2 (:623-633); tloc also for the irrigation arm (header (b)).
    int tloc = 0, tloc2 = 0;
    if (ahoption == 1 || iri_scheme == 1) {
        float h = FADD(FADD(FDIV(FMUL(FDIV(c.omg, U_PI), 180.0f), 15.0f), 12.0f), 0.5f);
        tloc = ((int)h) % 24;
        if (tloc < 0) tloc += 24;
        if (tloc == 0) tloc = 24;
    }
    if (alhoption == 1) {
        float h = FADD(FMUL(FADD(FDIV(FMUL(FDIV(c.omg, U_PI), 180.0f), 15.0f), 12.0f), 2.0f), 0.5f);
        tloc2 = ((int)h) % 48;
        if (tloc2 < 0) tloc2 += 48;
        if (tloc2 == 0) tloc2 = 48;
    }

    // read_param (:635-645)
    float zr = pt[PT_ZR], z0c = pt[PT_Z0C], z0hc = pt[PT_Z0HC], zdc = pt[PT_ZDC];
    float svf = pt[PT_SVF], r = pt[PT_R], rw = pt[PT_RW], hgt = pt[PT_HGT];
    float ah = pt[PT_AH], alh = pt[PT_ALH];
    float betr = pt[PT_BETR], betb = pt[PT_BETB], betg = pt[PT_BETG];
    float capr = pt[PT_CAPR], capb = pt[PT_CAPB], capg = pt[PT_CAPG];
    float aksr = pt[PT_AKSR], aksb = pt[PT_AKSB], aksg = pt[PT_AKSG];
    float albr = pt[PT_ALBR], albb = pt[PT_ALBB], albg = pt[PT_ALBG];
    float epsr = pt[PT_EPSR], epsb = pt[PT_EPSB], epsg = pt[PT_EPSG];
    float z0r = pt[PT_Z0R], z0b = pt[PT_Z0B], z0g = pt[PT_Z0G];
    float z0hb = pt[PT_Z0HB], z0hg = pt[PT_Z0HG];
    float trlend = pt[PT_TRLEND], tblend = pt[PT_TBLEND], tglend = pt[PT_TGLEND];
    float akanda = pt[PT_AKANDA];

    // :794-804
    if (ahoption == 1) ah = FMUL(ah, glob[PG_AHDIUPRF + tloc - 1]);
    int kalh = 0;
    if (alhoption == 1) {
        if (c.jmonth == 3 || c.jmonth == 4 || c.jmonth == 5) kalh = 1;
        if (c.jmonth == 6 || c.jmonth == 7 || c.jmonth == 8) kalh = 2;
        if (c.jmonth == 9 || c.jmonth == 10 || c.jmonth == 11) kalh = 3;
        if (c.jmonth == 12 || c.jmonth == 1 || c.jmonth == 2) kalh = 4;
        alh = FMUL(FMUL(alh, glob[PG_ALHDIUPRF + tloc2 - 1]),
                   glob[PG_ALHSEASON + kalh - 1]);
    }

    // :825 WRF's fatal
    if (FADD(FADD(zdc, z0c), 2.0f) >= c.za) return UCM_ERR_ZA;

    // :829-841
    float ssgd = FMUL(U_SRATIO, c.ssg);
    float ssgq = FSUB(c.ssg, ssgd);
    float w = FMUL(2.0f, hgt);                     // 2.*1.*HGT, 2.*1. folded
    float vfgs = svf;
    float vfgw = FSUB(1.0f, svf);
    float vfwg = FDIV(FMUL(FSUB(1.0f, svf), FSUB(1.0f, r)), w);
    float vfws = vfwg;
    float vfww = FSUB(1.0f, FMUL(2.0f, vfwg));

    // :848-871
    float sx = FDIV(FDIV(FADD(ssgd, ssgq), 697.7f), 60.0f);
    float rx = FDIV(FDIV(c.llg, 697.7f), 60.0f);
    float rho = FMUL(c.rhoo, 0.001f);
    float trp = c.tr, tbp = c.tb, tgp = c.tg, tcp = c.tc, qcp = c.qc;
    float flxhumrp = c.flxhumr, flxhumbp = c.flxhumb, flxhumgp = c.flxhumg;
    float drelrp = c.drelr, drelbp = c.drelb, drelgp = c.drelg;
    float tgrp = c.tgr, cmcrp = c.cmcr;
    float smrp[UCM_NL];
    for (int k = 0; k < UCM_NL; ++k) smrp[k] = c.smr[k];

    // :874-883 irrigation
    if (iri_scheme == 1) {
        if (tloc == 21 || tloc == 22) {
            if (c.jmonth == 5 || c.jmonth == 6 || c.jmonth == 7 ||
                c.jmonth == 8 || c.jmonth == 9) {
                smrp[0] = GR_SMCREF;
                smrp[1] = GR_SMCREF;
            }
        }
    }

    float tav = FMUL(c.ta, FADD(1.0f, FMUL(0.61f, c.qa)));
    float ps = FDIV(FMUL(FMUL(c.rhoo, 287.0f), tav), 100.0f);

    // canopy wind :892-903
    float ur, zc, xlb, bb;
    if (FADD(zr, 2.0f) < c.za) {
        ur = FDIV(FMUL(c.ua, gfk_log(FDIV(FSUB(zr, zdc), z0c))),
                  gfk_log(FDIV(FSUB(c.za, zdc), z0c)));
        zc = FMUL(0.7f, zr);
        xlb = FMUL(0.4f, FSUB(zr, zdc));
        bb = FDIV(FMUL(0.4f, zr), FMUL(xlb, gfk_log(FDIV(FSUB(zr, zdc), z0c))));
        c.uc = FMUL(ur, gfk_exp(FSUB(0.0f, FMUL(bb, FSUB(1.0f, FDIV(zc, zr))))));
    } else {
        zc = FDIV(c.za, 2.0f);
        c.uc = FDIV(c.ua, 2.0f);
    }

    // shortwave :909-987 (SHADOW = .false.)
    float sr, sg, sgr, sb, snet;
    if (c.ssg > 0.0f) {
        float sr1 = FMUL(sx, FSUB(1.0f, albr));
        float sgr1 = FMUL(sx, FSUB(1.0f, GR_ALBV));
        float sg1 = FMUL(FMUL(sx, vfgs), FSUB(1.0f, albg));
        float sb1 = FMUL(FMUL(sx, vfws), FSUB(1.0f, albb));
        float sg2 = FMUL(FMUL(FDIV(FMUL(sb1, albb), FSUB(1.0f, albb)), vfgw), FSUB(1.0f, albg));
        float sb2 = FADD(FMUL(FMUL(FDIV(FMUL(sg1, albg), FSUB(1.0f, albg)), vfwg), FSUB(1.0f, albb)),
                         FMUL(FMUL(sb1, albb), vfww));
        sr = sr1;
        sgr = sgr1;
        sg = FADD(sg1, sg2);
        sb = FADD(sb1, sb2);
        if (groption == 1) {
            snet = FADD(FADD(FADD(FMUL(FMUL(r, fgr), sgr),
                                  FMUL(FMUL(r, FSUB(1.0f, fgr)), sr)),
                             FMUL(w, sb)), FMUL(rw, sg));
        } else {
            snet = FADD(FADD(FMUL(r, sr), FMUL(w, sb)), FMUL(rw, sg));
        }
    } else {
        sr = 0.0f;
        sg = 0.0f;
        sgr = 0.0f;
        sb = 0.0f;
        snet = 0.0f;
    }

    // roof exchange :1004-1024 (distributed arm refused on the host)
    float qfac = FADD(1.0f, FMUL(0.61f, c.qa));
    float t1vr = FMUL(trp, qfac);
    float th2v = FMUL(FADD(c.ta, FMUL(0.0098f, c.za)), qfac);
    float rlmo_urb = 0.0f;
    float cdr;
    u_sfcdif_urb(c.za, z0r, t1vr, th2v, c.ua, akanda, c.cmr, c.chr, rlmo_urb, cdr);
    float alphar = FMUL(FMUL(rho, U_CPCGS), c.chr);
    float chr = FDIV(FDIV(FDIV(alphar, rho), U_CPCGS), c.ua);

    // :1027-1050
    float rain1 = FDIV(FMUL(c.rain, 0.001f), 3600.0f);
    if (imp_scheme == 1) {
        if (c.rain > 1.0f) betr = 0.7f;
    }
    if (imp_scheme == 2) {
        if (flxhumrp <= 0.0f) flxhumrp = 0.0f;
        c.drelr = FADD(drelrp, FDIV(FMUL(FSUB(rain1, FDIV(FMUL(flxhumrp, c.rhoo), 1000.0f)),
                                         c.delt), porimp[0]));
        if (c.rain > 0.0f && c.drelr < drelrp) c.drelr = drelrp;
        if (c.drelr <= 0.0f) {
            c.drelr = 0.0f;
            betr = 0.0f;
        } else if (c.drelr <= dengimp[0]) {
            betr = FMUL(FDIV(c.drelr, dengimp[0]), porimp[0]);
        } else {
            c.drelr = dengimp[0];
            betr = porimp[0];
        }
        if (betr < 1.e-5f) betr = 0.0f;
    }

    // roof energy :1052-1121
    float rr, hr, eler, g0r;
    if (ts_scheme == 1) {
        for (int it = 1; it <= 20; ++it) {
            float es = u_es(trp);
            float desdt = FDIV(FMUL(U_C_CC, es), gfk_pow(trp, 2.0f));
            float psm = FSUB(ps, FMUL(0.378f, es));
            float qs0r = FDIV(FMUL(0.622f, es), psm);
            float dqs0rdtr = FDIV(FMUL(FMUL(desdt, 0.622f), ps), gfk_pow(psm, 2.0f));
            rr = FMUL(epsr, FSUB(rx, FDIV(FMUL(U_SIG, gfk_pow(trp, 4.0f)), 60.0f)));
            hr = FMUL(FMUL(FMUL(FMUL(FMUL(rho, U_CPCGS), chr), c.ua), FSUB(trp, c.ta)), 100.0f);
            eler = FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho, U_EL), chr), c.ua), betr),
                             FSUB(qs0r, c.qa)), 100.0f);
            g0r = FDIV(FMUL(aksr, FSUB(trp, c.trl[0])), FDIV(dzr[0], 2.0f));
            float f = FSUB(FSUB(FSUB(FADD(sr, rr), hr), eler), g0r);
            float drrdtr = FDIV(FSUB(0.0f, FMUL(FMUL(FMUL(4.0f, epsr), U_SIG),
                                                gfk_pow(trp, 3.0f))), 60.0f);
            float dhrdtr = FMUL(FMUL(FMUL(FMUL(rho, U_CPCGS), chr), c.ua), 100.0f);
            float delerdtr = FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho, U_EL), chr), c.ua), betr),
                                       dqs0rdtr), 100.0f);
            float dg0rdtr = FDIV(FMUL(2.0f, aksr), dzr[0]);
            float dfdt = FSUB(FSUB(FSUB(drrdtr, dhrdtr), delerdtr), dg0rdtr);
            float dtr = FDIV(f, dfdt);
            c.tr = FSUB(trp, dtr);
            trp = c.tr;
            if (fabs(f) < 0.000001f && fabs(dtr) < 0.000001f) break;
        }
        u_multi_layer(boundr, g0r, capr, aksr, c.trl, dzr, c.delt, trlend);
    } else {
        float es = u_es(trp);
        float qs0r = FDIV(FMUL(0.622f, es), FSUB(ps, FMUL(0.378f, es)));
        rr = FMUL(epsr, FSUB(rx, FDIV(FMUL(U_SIG, gfk_pow(trp, 4.0f)), 60.0f)));
        hr = FMUL(FMUL(FMUL(FMUL(FMUL(rho, U_CPCGS), chr), c.ua), FSUB(trp, c.ta)), 100.0f);
        eler = FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho, U_EL), chr), c.ua), betr),
                         FSUB(qs0r, c.qa)), 100.0f);
        g0r = FSUB(FSUB(FADD(sr, rr), hr), eler);
        c.tr = u_force_restore(capr, aksr, c.delt, sr, rr, hr, eler, trlend, trp);
        trp = c.tr;
    }
    float flxthr = FDIV(FDIV(FDIV(hr, rho), U_CPCGS), 100.0f);
    c.flxhumr = FDIV(FDIV(FDIV(eler, rho), U_EL), 100.0f);

    // green roof :1127-1228
    float flxthgr, flxhumgr, cdgr = 0.0f, rgr = 0.0f, g0gr = 0.0f;
    if (groption == 1) {
        float t1vgr = FMUL(tgrp, qfac);
        rlmo_urb = 0.0f;
        u_sfcdif_urb(c.za, z0r, t1vgr, th2v, c.ua, akanda, c.cmgr, c.chgr, rlmo_urb, cdgr);
        float alphagr = FMUL(FMUL(rho, U_CPCGS), c.chgr);
        float chgr = FDIV(FDIV(FDIV(alphagr, rho), U_CPCGS), c.ua);
        float runoff1 = 0.0f, runoff2 = 0.0f, runoff3 = 0.0f;
        float zsoilr[UCM_NL];
        zsoilr[0] = FSUB(0.0f, dzgr[0]);
        for (int k = 1; k < GR_NGR; ++k) zsoilr[k] = FADD(FSUB(0.0f, dzgr[k]), zsoilr[k - 1]);
        float hgr = 0.0f, elegr = 0.0f, yy = 0.0f, zz1 = 0.0f, df1 = 0.0f;
        for (int it = 1; it <= 100; ++it) {
            float es = u_es(tgrp);
            float desdt = FDIV(FMUL(U_C_CC, es), gfk_pow(tgrp, 2.0f));
            float psm = FSUB(ps, FMUL(0.378f, es));
            float qs0gr = FDIV(FMUL(0.622f, es), psm);
            float dqs0grdtgr = FDIV(FMUL(FMUL(desdt, 0.622f), ps), gfk_pow(psm, 2.0f));
            float epgr = FMUL(FMUL(FMUL(c.rhoo, chgr), c.ua), FSUB(qs0gr, c.qa));
            float edir, ettr, ecr, drip;
            if (epgr > 0.0f) {
                edir = u_direvap(epgr, smrp[0], GR_SHDFAC, GR_SMCMAX, GR_SMCDRY, GR_FXEXP);
                u_transp(ettr, etr, ecr, GR_SHDFAC, epgr, cmcrp, GR_CFACTR, GR_CMCMAX,
                         GR_LAI, GR_RSMIN, GR_RSMAX, GR_RGL, sx, tgrp, c.ta, c.qa,
                         smrp, GR_SMCWLT, GR_SMCREF, U_CPP, chgr, GR_EPSV, c.delt,
                         dzgr, zsoilr, GR_HS);
                u_smflx(smrp, c.smr, cmcrp, c.cmcr, c.delt, c.rain, zsoilr, GR_SMCMAX,
                        GR_BEXP, GR_SMCWLT, GR_DKSAT, GR_DWSAT, GR_SHDFAC, GR_CMCMAX,
                        runoff1, runoff2, runoff3, edir, ecr, etr, drip);
            } else {
                float dew = FSUB(0.0f, epgr);
                float raindr = FADD(c.rain, FMUL(dew, 3600.0f));
                edir = 0.0f;
                ecr = 0.0f;
                ettr = 0.0f;
                u_smflx(smrp, c.smr, cmcrp, c.cmcr, c.delt, raindr, zsoilr, GR_SMCMAX,
                        GR_BEXP, GR_SMCWLT, GR_DKSAT, GR_DWSAT, GR_SHDFAC, GR_CMCMAX,
                        runoff1, runoff2, runoff3, edir, ecr, etr, drip);
            }
            edir = FMUL(edir, 1000.0f);
            ettr = FMUL(ettr, 1000.0f);
            ecr = FMUL(ecr, 1000.0f);
            float etar = FADD(FADD(edir, ettr), ecr);
            if (etar < 1.e-20f) etar = 0.0f;
            float betgr;
            if (epgr <= 0.0f) {
                betgr = 0.0f;
            } else {
                betgr = FDIV(etar, epgr);
            }
            elegr = FMUL(FDIV(FMUL(FMUL(etar, rho), U_EL), c.rhoo), 100.0f);
            df1 = u_tdfcnd(c.smr[0], GR_QUARTZ, GR_SMCMAX);
            df1 = FMUL(df1, U_EXP_M16);
            rgr = FMUL(GR_EPSV, FSUB(rx, FDIV(FMUL(U_SIG, gfk_pow(c.ta, 4.0f)), 60.0f)));
            float rgrr = FMUL(FMUL(FADD(sgr, rgr), 697.7f), 60.0f);
            float rch = FMUL(FMUL(c.rhoo, U_CPP), chgr);
            float rr1 = FADD(FDIV(FMUL(FMUL(GR_EPSV, u_powi(c.ta, 4)), 6.48e-8f),
                                  FMUL(ps, chgr)), 1.0f);
            float rr2;
            if (c.rain > 0.0f) {
                rr2 = FADD(rr1, FDIV(FMUL(FDIV(c.rain, 3600.0f), 4.218e+3f), rch));
            } else {
                rr2 = rr1;
            }
            yy = FADD(c.ta, FDIV(FSUB(FDIV(rgrr, rch),
                                      FDIV(FMUL(FMUL(betgr, epgr), U_ELL), rch)), rr2));
            zz1 = FADD(FDIV(df1, FSUB(0.0f, FMUL(FMUL(FMUL(0.5f, zsoilr[0]), rch), rr2))), 1.0f);
            hgr = FMUL(FMUL(FMUL(FMUL(FMUL(rho, U_CPCGS), chgr), c.ua), FSUB(tgrp, c.ta)), 100.0f);
            runoff3 = FDIV(runoff3, c.delt);
            runoff2 = FADD(runoff2, runoff3);
            g0gr = FDIV(FDIV(FDIV(FMUL(df1, FSUB(tgrp, c.tgrl[0])), FDIV(dzgr[0], 2.0f)),
                             697.7f), 60.0f);
            float fv = FSUB(FSUB(FSUB(FADD(sgr, rgr), hgr), elegr), g0gr);
            float drrdtgr = FDIV(FSUB(0.0f, FMUL(FMUL(FMUL(4.0f, GR_EPSV), U_SIG),
                                                 gfk_pow(tgrp, 3.0f))), 60.0f);
            float dhrdtgr = FMUL(FMUL(FMUL(FMUL(rho, U_CPCGS), chgr), c.ua), 100.0f);
            float delerdtgr = FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho, U_EL), chgr), c.ua), betgr),
                                        dqs0grdtgr), 100.0f);
            float dg0rdtgr = FMUL(FMUL(FDIV(FMUL(2.0f, df1), dzgr[0]), U_INV_4P1868), 1.e-4f);
            float dfdvt = FSUB(FSUB(FSUB(drrdtgr, dhrdtgr), delerdtgr), dg0rdtgr);
            float dtgr = FDIV(FDIV(fv, dfdvt), 6.0f);
            c.tgr = FSUB(tgrp, dtgr);
            tgrp = c.tgr;
            if (fabs(fv) < 0.0001f && fabs(dtgr) < 0.001f) break;
        }
        u_shflx(c.tgrl, c.smr, GR_SMCMAX, yy, zz1, zsoilr, trlend, GR_ZBOT, c.delt,
                df1, GR_QUARTZ, GR_CSOIL, capr);
        flxthgr = FDIV(FDIV(FDIV(hgr, rho), U_CPCGS), 100.0f);
        flxhumgr = FDIV(FDIV(FDIV(elegr, rho), U_EL), 100.0f);
    } else {
        flxthgr = 0.0f;
        flxhumgr = 0.0f;
    }

    // wall and road exchange :1245-1276
    float t1vc = FMUL(tcp, qfac);
    rlmo_urb = 0.0f;
    float cdc;
    u_sfcdif_urb(c.za, z0c, t1vc, th2v, c.ua, akanda, c.cmc, c.chc, rlmo_urb, cdc);
    float alphac = FMUL(FMUL(rho, U_CPCGS), c.chc);
    float alphab, alphag;
    if (ch_scheme == 1) {
        float z = zdc;
        float bhb = FDIV(gfk_log(FDIV(z0b, z0hb)), 0.4f);
        float bhg = FDIV(gfk_log(FDIV(z0g, z0hg)), 0.4f);
        float uc2 = FMUL(c.uc, c.uc);
        float ribb = FDIV(FMUL(FMUL(FDIV(U_19P6, FADD(tcp, tbp)), FSUB(tcp, tbp)), FADD(z, z0b)), uc2);
        float ribg = FDIV(FMUL(FMUL(FDIV(U_19P6, FADD(tcp, tgp)), FSUB(tcp, tgp)), FADD(z, z0g)), uc2);
        float cdb, cdg;
        u_mos(c.xxxb, alphab, cdb, bhb, ribb, z, z0b, c.uc, tcp, tbp, rho);
        u_mos(c.xxxg, alphag, cdg, bhg, ribg, z, z0g, c.uc, tcp, tgp, rho);
    } else {
        alphab = FDIV(FMUL(FMUL(rho, U_CPCGS), FADD(6.15f, FMUL(4.18f, c.uc))), 1200.0f);
        if (c.uc > 5.0f)
            alphab = FDIV(FMUL(FMUL(rho, U_CPCGS), FMUL(7.51f, gfk_pow(c.uc, 0.78f))), 1200.0f);
        alphag = FDIV(FMUL(FMUL(rho, U_CPCGS), FADD(6.15f, FMUL(4.18f, c.uc))), 1200.0f);
        if (c.uc > 5.0f)
            alphag = FDIV(FMUL(FMUL(rho, U_CPCGS), FMUL(7.51f, gfk_pow(c.uc, 0.78f))), 1200.0f);
    }
    float chb = FDIV(FDIV(FDIV(alphab, rho), U_CPCGS), c.uc);
    float chg = FDIV(FDIV(FDIV(alphag, rho), U_CPCGS), c.uc);

    // :1279-1317
    if (imp_scheme == 1) {
        betb = 0.0f;
        if (c.rain > 1.0f) betg = 0.7f;
    }
    if (imp_scheme == 2) {
        if (flxhumbp <= 0.0f) flxhumbp = 0.0f;
        if (flxhumgp <= 0.0f) flxhumgp = 0.0f;
        c.drelb = FADD(drelbp, FDIV(FMUL(FSUB(rain1, FDIV(FMUL(flxhumbp, c.rhoo), 1000.0f)),
                                         c.delt), porimp[1]));
        if (c.rain > 0.0f && c.drelb < drelbp) c.drelb = drelbp;
        c.drelg = FADD(drelgp, FDIV(FMUL(FSUB(rain1, FDIV(FMUL(flxhumgp, c.rhoo), 1000.0f)),
                                         c.delt), porimp[2]));
        if (c.rain > 0.0f && c.drelg < drelgp) c.drelg = drelgp;
        if (c.drelb <= 0.0f) {
            c.drelb = 0.0f;
            betb = 0.0f;
        } else if (c.drelb <= dengimp[1]) {
            betb = FMUL(FDIV(c.drelb, dengimp[1]), porimp[1]);
        } else {
            c.drelb = dengimp[1];
            betb = porimp[1];
        }
        if (c.drelg <= 0.0f) {
            c.drelg = 0.0f;
            betg = 0.0f;
        } else if (c.drelg <= dengimp[2]) {
            betg = FMUL(FDIV(c.drelg, dengimp[2]), porimp[2]);
        } else {
            c.drelg = dengimp[2];
            betg = porimp[2];
        }
        if (betg < 1.e-5f) betg = 0.0f;
        if (betb < 1.e-5f) betb = 0.0f;
    }

    // wall and road energy :1319-1535
    float rb, rg, hb, hg, eleb, eleg, g0b, g0g;
    const float s60 = 60.0f;
    if (ts_scheme == 1) {
        for (int it = 1; it <= 20; ++it) {
            float es = u_es(tbp);
            float desdt = FDIV(FMUL(U_C_CC, es), gfk_pow(tbp, 2.0f));
            float psm = FSUB(ps, FMUL(0.378f, es));
            float qs0b = FDIV(FMUL(0.622f, es), psm);
            float dqs0bdtb = FDIV(FMUL(FMUL(desdt, 0.622f), ps), gfk_pow(psm, 2.0f));
            es = u_es(tgp);
            desdt = FDIV(FMUL(U_C_CC, es), gfk_pow(tgp, 2.0f));
            psm = FSUB(ps, FMUL(0.378f, es));
            float qs0g = FDIV(FMUL(0.622f, es), psm);
            float dqs0gdtg = FDIV(FMUL(FMUL(desdt, 0.622f), ps), gfk_pow(psm, 2.0f));

            float tbp4 = gfk_pow(tbp, 4.0f), tgp4 = gfk_pow(tgp, 4.0f);
            float rg1 = FMUL(epsg, FSUB(FADD(FMUL(rx, vfgs),
                                             FDIV(FMUL(FMUL(FMUL(epsb, vfgw), U_SIG), tbp4), s60)),
                                        FDIV(FMUL(U_SIG, tgp4), s60)));
            float rb1 = FMUL(epsb, FSUB(FADD(FADD(FMUL(rx, vfws),
                                                  FDIV(FMUL(FMUL(FMUL(epsg, vfwg), U_SIG), tgp4), s60)),
                                             FDIV(FMUL(FMUL(FMUL(epsb, vfww), U_SIG), tbp4), s60)),
                                        FDIV(FMUL(U_SIG, tbp4), s60)));
            float omeb = FSUB(1.0f, epsb), omeg = FSUB(1.0f, epsg), omsvf = FSUB(1.0f, svf);
            float om2vfws = FSUB(1.0f, FMUL(2.0f, vfws));
            float rg2 = FMUL(epsg, FADD(FADD(
                FMUL(FMUL(FMUL(omeb, omsvf), vfws), rx),
                FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(omeb, omsvf), vfwg), epsg), U_SIG), tgp4), s60)),
                FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(epsb, omeb), omsvf), om2vfws), U_SIG), tbp4), s60)));
            float rb2 = FMUL(epsb, FADD(FADD(FADD(FADD(
                FMUL(FMUL(FMUL(omeg, vfwg), vfgs), rx),
                FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(omeg, epsb), vfgw), vfwg), U_SIG), tbp4), s60)),
                FMUL(FMUL(FMUL(omeb, vfws), om2vfws), rx)),
                FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(omeb, vfwg), om2vfws), U_SIG), epsg), tgp4), s60)),
                FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(epsb, omeb), om2vfws), om2vfws), U_SIG), tbp4), s60)));
            rg = FADD(rg1, rg2);
            rb = FADD(rb1, rb2);

            float tb3 = gfk_pow(c.tb, 3.0f), tg3 = gfk_pow(c.tg, 3.0f);
            float drbdtb1 = FDIV(FMUL(epsb, FSUB(FMUL(FMUL(FMUL(FMUL(4.0f, epsb), U_SIG), tb3), vfww),
                                                 FMUL(FMUL(4.0f, U_SIG), tb3))), s60);
            float drbdtg1 = FDIV(FMUL(epsb, FMUL(FMUL(FMUL(FMUL(4.0f, epsg), U_SIG), tg3), vfwg)), s60);
            float drbdtb2 = FDIV(FMUL(epsb, FADD(
                FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4.0f, omeg), epsb), U_SIG), tb3), vfgw), vfwg),
                FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4.0f, epsb), omeb), U_SIG), tb3), vfww), vfww))), s60);
            float drbdtg2 = FDIV(FMUL(epsb, FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4.0f, omeb), epsg), U_SIG), tg3),
                                                      vfwg), vfww)), s60);
            float drgdtb1 = FDIV(FMUL(epsg, FMUL(FMUL(FMUL(FMUL(4.0f, epsb), U_SIG), tb3), vfgw)), s60);
            float drgdtg1 = FDIV(FMUL(epsg, FSUB(0.0f, FMUL(FMUL(4.0f, U_SIG), tg3))), s60);
            float drgdtb2 = FDIV(FMUL(epsg, FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4.0f, epsb), omeb), U_SIG), tb3),
                                                      vfww), vfgw)), s60);
            float drgdtg2 = FDIV(FMUL(epsg, FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(4.0f, omeb), epsg), U_SIG), tg3),
                                                      vfwg), vfgw)), s60);
            float drbdtb = FADD(drbdtb1, drbdtb2);
            float drbdtg = FADD(drbdtg1, drbdtg2);
            float drgdtb = FADD(drgdtb1, drgdtb2);
            float drgdtg = FADD(drgdtg1, drgdtg2);

            float rcpb = FMUL(FMUL(FMUL(rho, U_CPCGS), chb), c.uc);
            float rcpg = FMUL(FMUL(FMUL(rho, U_CPCGS), chg), c.uc);
            hb = FMUL(FMUL(rcpb, FSUB(tbp, tcp)), 100.0f);
            hg = FMUL(FMUL(rcpg, FSUB(tgp, tcp)), 100.0f);
            float den_t = FADD(FADD(FMUL(rw, alphac), FMUL(rw, alphag)), FMUL(w, alphab));
            float dtcdtb = FDIV(FMUL(w, alphab), den_t);
            float dtcdtg = FDIV(FMUL(rw, alphag), den_t);
            float dhbdtb = FMUL(FMUL(rcpb, FSUB(1.0f, dtcdtb)), 100.0f);
            float dhbdtg = FMUL(FMUL(rcpb, FSUB(0.0f, dtcdtg)), 100.0f);
            float dhgdtg = FMUL(FMUL(rcpg, FSUB(1.0f, dtcdtg)), 100.0f);
            float dhgdtb = FMUL(FMUL(rcpg, FSUB(0.0f, dtcdtb)), 100.0f);
            float relb = FMUL(FMUL(FMUL(FMUL(rho, U_EL), chb), c.uc), betb);
            float relg = FMUL(FMUL(FMUL(FMUL(rho, U_EL), chg), c.uc), betg);
            eleb = FMUL(FMUL(relb, FSUB(qs0b, qcp)), 100.0f);
            eleg = FMUL(FMUL(relg, FSUB(qs0g, qcp)), 100.0f);
            float den_q = FADD(FADD(FMUL(rw, alphac), FMUL(FMUL(rw, alphag), betg)),
                               FMUL(FMUL(w, alphab), betb));
            float dqcdtb = FDIV(FMUL(FMUL(FMUL(w, alphab), betb), dqs0bdtb), den_q);
            float dqcdtg = FDIV(FMUL(FMUL(FMUL(rw, alphag), betg), dqs0gdtg), den_q);
            float delebdtb = FMUL(FMUL(relb, FSUB(dqs0bdtb, dqcdtb)), 100.0f);
            float delebdtg = FMUL(FMUL(relb, FSUB(0.0f, dqcdtg)), 100.0f);
            float delegdtg = FMUL(FMUL(relg, FSUB(dqs0gdtg, dqcdtg)), 100.0f);
            float delegdtb = FMUL(FMUL(relg, FSUB(0.0f, dqcdtb)), 100.0f);

            g0b = FDIV(FMUL(aksb, FSUB(tbp, c.tbl[0])), FDIV(dzb[0], 2.0f));
            g0g = FDIV(FMUL(aksg, FSUB(tgp, c.tgl[0])), FDIV(dzg[0], 2.0f));
            float dg0bdtb = FDIV(FMUL(2.0f, aksb), dzb[0]);
            float dg0bdtg = 0.0f;
            float dg0gdtg = FDIV(FMUL(2.0f, aksg), dzg[0]);
            float dg0gdtb = 0.0f;

            float f = FSUB(FSUB(FSUB(FADD(sb, rb), hb), eleb), g0b);
            float fx = FSUB(FSUB(FSUB(drbdtb, dhbdtb), delebdtb), dg0bdtb);
            float fy = FSUB(FSUB(FSUB(drbdtg, dhbdtg), delebdtg), dg0bdtg);
            float gf = FSUB(FSUB(FSUB(FADD(sg, rg), hg), eleg), g0g);
            float gx = FSUB(FSUB(FSUB(drgdtb, dhgdtb), delegdtb), dg0gdtb);
            float gy = FSUB(FSUB(FSUB(drgdtg, dhgdtg), delegdtg), dg0gdtg);
            float dtb = FDIV(FSUB(FMUL(gf, fy), FMUL(f, gy)), FSUB(FMUL(fx, gy), FMUL(gx, fy)));
            float dtg = FSUB(0.0f, FDIV(FADD(gf, FMUL(gx, dtb)), gy));
            c.tb = FADD(tbp, dtb);
            c.tg = FADD(tgp, dtg);
            tbp = c.tb;
            tgp = c.tg;
            float tc1 = FADD(FADD(FMUL(rw, alphac), FMUL(rw, alphag)), FMUL(w, alphab));
            float tc2 = FADD(FADD(FMUL(FMUL(rw, alphac), c.ta), FMUL(FMUL(rw, alphag), tgp)),
                             FMUL(FMUL(w, alphab), tbp));
            c.tc = FDIV(tc2, tc1);
            float qc1 = FADD(FADD(FMUL(rw, alphac), FMUL(FMUL(rw, alphag), betg)),
                             FMUL(FMUL(w, alphab), betb));
            float qc2 = FADD(FADD(FMUL(FMUL(rw, alphac), c.qa),
                                  FMUL(FMUL(FMUL(rw, alphag), betg), qs0g)),
                             FMUL(FMUL(FMUL(w, alphab), betb), qs0b));
            c.qc = FDIV(qc2, qc1);
            float dtc = FSUB(tcp, c.tc);
            tcp = c.tc;
            qcp = c.qc;
            if (fabs(f) < 0.000001f && fabs(dtb) < 0.000001f &&
                fabs(gf) < 0.000001f && fabs(dtg) < 0.000001f &&
                fabs(dtc) < 0.000001f) break;
        }
        u_multi_layer(boundb, g0b, capb, aksb, c.tbl, dzb, c.delt, tblend);
        u_multi_layer(boundg, g0g, capg, aksg, c.tgl, dzg, c.delt, tglend);
    } else {
        float es = u_es(tbp);
        float qs0b = FDIV(FMUL(0.622f, es), FSUB(ps, FMUL(0.378f, es)));
        es = u_es(tgp);
        float qs0g = FDIV(FMUL(0.622f, es), FSUB(ps, FMUL(0.378f, es)));
        float tbp4 = gfk_pow(tbp, 4.0f), tgp4 = gfk_pow(tgp, 4.0f);
        float rg1 = FMUL(epsg, FSUB(FADD(FMUL(rx, vfgs),
                                         FDIV(FMUL(FMUL(FMUL(epsb, vfgw), U_SIG), tbp4), s60)),
                                    FDIV(FMUL(U_SIG, tgp4), s60)));
        float rb1 = FMUL(epsb, FSUB(FADD(FADD(FMUL(rx, vfws),
                                              FDIV(FMUL(FMUL(FMUL(epsg, vfwg), U_SIG), tgp4), s60)),
                                         FDIV(FMUL(FMUL(FMUL(epsb, vfww), U_SIG), tbp4), s60)),
                                    FDIV(FMUL(U_SIG, tbp4), s60)));
        float omeb = FSUB(1.0f, epsb), omeg = FSUB(1.0f, epsg), omsvf = FSUB(1.0f, svf);
        float om2vfws = FSUB(1.0f, FMUL(2.0f, vfws));
        float rg2 = FMUL(epsg, FADD(FADD(
            FMUL(FMUL(FMUL(omeb, omsvf), vfws), rx),
            FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(omeb, omsvf), vfwg), epsg), U_SIG), tgp4), s60)),
            FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(epsb, omeb), omsvf), om2vfws), U_SIG), tbp4), s60)));
        float rb2 = FMUL(epsb, FADD(FADD(FADD(FADD(
            FMUL(FMUL(FMUL(omeg, vfwg), vfgs), rx),
            FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(omeg, epsb), vfgw), vfwg), U_SIG), tbp4), s60)),
            FMUL(FMUL(FMUL(omeb, vfws), om2vfws), rx)),
            FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(omeb, vfwg), om2vfws), U_SIG), epsg), tgp4), s60)),
            FDIV(FMUL(FMUL(FMUL(FMUL(FMUL(epsb, omeb), om2vfws), om2vfws), U_SIG), tbp4), s60)));
        rg = FADD(rg1, rg2);
        rb = FADD(rb1, rb2);
        hb = FMUL(FMUL(FMUL(FMUL(FMUL(rho, U_CPCGS), chb), c.uc), FSUB(tbp, tcp)), 100.0f);
        eleb = FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho, U_EL), chb), c.uc), betb), FSUB(qs0b, qcp)), 100.0f);
        g0b = FSUB(FSUB(FADD(sb, rb), hb), eleb);
        hg = FMUL(FMUL(FMUL(FMUL(FMUL(rho, U_CPCGS), chg), c.uc), FSUB(tgp, tcp)), 100.0f);
        eleg = FMUL(FMUL(FMUL(FMUL(FMUL(FMUL(rho, U_EL), chg), c.uc), betg), FSUB(qs0g, qcp)), 100.0f);
        g0g = FSUB(FSUB(FADD(sg, rg), hg), eleg);
        c.tb = u_force_restore(capb, aksb, c.delt, sb, rb, hb, eleb, tblend, tbp);
        c.tg = u_force_restore(capg, aksg, c.delt, sg, rg, hg, eleg, tglend, tgp);
        tbp = c.tb;
        tgp = c.tg;
        float tc1 = FADD(FADD(FMUL(rw, alphac), FMUL(rw, alphag)), FMUL(w, alphab));
        float tc2 = FADD(FADD(FMUL(FMUL(rw, alphac), c.ta), FMUL(FMUL(rw, alphag), tgp)),
                         FMUL(FMUL(w, alphab), tbp));
        c.tc = FDIV(tc2, tc1);
        float qc1 = FADD(FADD(FMUL(rw, alphac), FMUL(FMUL(rw, alphag), betg)),
                         FMUL(FMUL(w, alphab), betb));
        float qc2 = FADD(FADD(FMUL(FMUL(rw, alphac), c.qa),
                              FMUL(FMUL(FMUL(rw, alphag), betg), qs0g)),
                         FMUL(FMUL(FMUL(w, alphab), betb), qs0b));
        c.qc = FDIV(qc2, qc1);
        tcp = c.tc;
        qcp = c.qc;
    }
    float flxthb = FDIV(FDIV(FDIV(hb, rho), U_CPCGS), 100.0f);
    c.flxhumb = FDIV(FDIV(FDIV(eleb, rho), U_EL), 100.0f);
    float flxthg = FDIV(FDIV(FDIV(hg, rho), U_CPCGS), 100.0f);
    c.flxhumg = FDIV(FDIV(FDIV(eleg, rho), U_EL), 100.0f);

    // totals :1541-1573
    float flxth, flxhum, flxuv, flxg, lnet;
    if (groption == 1) {
        float omfgr = FSUB(1.0f, fgr);
        float th = FADD(FADD(FADD(FMUL(FMUL(omfgr, r), flxthr), FMUL(FMUL(fgr, r), flxthgr)),
                             FMUL(w, flxthb)), FMUL(rw, flxthg));
        flxth = (ahoption == 1) ? FADD(th, FDIV(FDIV(ah, c.rhoo), U_CPP)) : th;
        float hu = FADD(FADD(FADD(FMUL(FMUL(omfgr, r), c.flxhumr), FMUL(FMUL(fgr, r), flxhumgr)),
                             FMUL(w, c.flxhumb)), FMUL(rw, c.flxhumg));
        flxhum = (alhoption == 1) ? FADD(hu, FDIV(FDIV(alh, c.rhoo), U_ELL)) : hu;
        flxuv = FMUL(FMUL(FADD(FADD(FMUL(FMUL(omfgr, r), cdr), FMUL(FMUL(fgr, r), cdgr)),
                               FMUL(rw, cdc)), c.ua), c.ua);
        flxg = FADD(FADD(FADD(FMUL(FMUL(omfgr, r), g0r), FMUL(FMUL(fgr, r), g0gr)),
                         FMUL(w, g0b)), FMUL(rw, g0g));
        lnet = FADD(FADD(FADD(FMUL(FMUL(omfgr, r), rr), FMUL(FMUL(fgr, r), rgr)),
                         FMUL(w, rb)), FMUL(rw, rg));
    } else {
        float th = FADD(FADD(FMUL(r, flxthr), FMUL(w, flxthb)), FMUL(rw, flxthg));
        flxth = (ahoption == 1) ? FADD(th, FDIV(FDIV(ah, c.rhoo), U_CPP)) : th;
        float hu = FADD(FADD(FMUL(r, c.flxhumr), FMUL(w, c.flxhumb)), FMUL(rw, c.flxhumg));
        flxhum = (alhoption == 1) ? FADD(hu, FDIV(FDIV(alh, c.rhoo), U_ELL)) : hu;
        flxuv = FMUL(FMUL(FADD(FMUL(r, cdr), FMUL(rw, cdc)), c.ua), c.ua);
        flxg = FADD(FADD(FMUL(r, g0r), FMUL(w, g0b)), FMUL(rw, g0g));
        lnet = FADD(FADD(FMUL(r, rr), FMUL(w, rb)), FMUL(rw, rg));
    }

    // :1579-1591
    c.sh = FMUL(FMUL(flxth, c.rhoo), U_CPP);
    c.lh = FMUL(FMUL(flxhum, c.rhoo), U_ELL);
    c.lh_kin = FMUL(flxhum, c.rhoo);
    c.lw = FSUB(c.llg, FMUL(FMUL(lnet, 697.7f), 60.0f));
    c.sw = FSUB(c.ssg, FMUL(FMUL(snet, 697.7f), 60.0f));
    c.alb = 0.0f;
    if (fabs(c.ssg) > 0.0001f) c.alb = FDIV(c.sw, c.ssg);
    c.g = FSUB(0.0f, FMUL(FMUL(flxg, 697.7f), 60.0f));
    c.rn = FMUL(FMUL(FADD(snet, lnet), 697.7f), 60.0f);
    c.ust = FSQRT(flxuv);
    float tst = FSUB(0.0f, FDIV(flxth, c.ust));

    // diagnostics :1597-1688
    float z0 = z0c, z0h = z0hc;
    float z = FSUB(c.za, zdc);
    c.znt = z0;
    float xxx = FDIV(FDIV(FDIV(FMUL(FMUL(U_KARMANG, z), tst), c.ta), c.ust), c.ust);
    if (xxx >= 1.0f) xxx = 1.0f;
    if (xxx <= -5.0f) xxx = -5.0f;
    if (xxx > 0.0f) {
        c.psim = FSUB(0.0f, FMUL(5.0f, xxx));
        c.psih = FSUB(0.0f, FMUL(5.0f, xxx));
    } else {
        float x = gfk_pow(FSUB(1.0f, FMUL(16.0f, xxx)), 0.25f);
        float t = FADD(FMUL(2.0f, gfk_log(FDIV(FADD(1.0f, x), 2.0f))),
                       gfk_log(FDIV(FADD(1.0f, FMUL(x, x)), 2.0f)));
        t = FSUB(t, FMUL(2.0f, u_atan(x)));
        c.psim = FADD(t, U_PIO2);
        c.psih = FMUL(2.0f, gfk_log(FDIV(FADD(1.0f, FMUL(x, x)), 2.0f)));
    }
    c.gz1oz0 = gfk_log(FDIV(z, z0));
    c.ts = FADD(c.ta, FDIV(flxth, c.chs));
    c.qs = FADD(c.qa, FDIV(flxhum, c.chs));

    float xxx2 = FMUL(FDIV(2.0f, z), xxx);
    if (xxx2 >= 1.0f) xxx2 = 1.0f;
    if (xxx2 <= -5.0f) xxx2 = -5.0f;
    float psim2, psih2;
    if (xxx2 > 0.0f) {
        psim2 = FSUB(0.0f, FMUL(5.0f, xxx2));
        psih2 = FSUB(0.0f, FMUL(5.0f, xxx2));
    } else {
        float x = gfk_pow(FSUB(1.0f, FMUL(16.0f, xxx2)), 0.25f);
        float t = FADD(FMUL(2.0f, gfk_log(FDIV(FADD(1.0f, x), 2.0f))),
                       gfk_log(FDIV(FADD(1.0f, FMUL(x, x)), 2.0f)));
        t = FSUB(t, FMUL(2.0f, u_atan(x)));
        psim2 = FADD(t, U_TWO_ATAN1);
        psih2 = FMUL(2.0f, gfk_log(FDIV(FADD(1.0f, FMUL(x, x)), 2.0f)));
    }
    float xxx10 = FMUL(FDIV(10.0f, z), xxx);
    if (xxx10 >= 1.0f) xxx10 = 1.0f;
    if (xxx10 <= -5.0f) xxx10 = -5.0f;
    float psim10;
    if (xxx10 > 0.0f) {
        psim10 = FSUB(0.0f, FMUL(5.0f, xxx10));
    } else {
        float x = gfk_pow(FSUB(1.0f, FMUL(16.0f, xxx10)), 0.25f);
        float t = FADD(FMUL(2.0f, gfk_log(FDIV(FADD(1.0f, x), 2.0f))),
                       gfk_log(FDIV(FADD(1.0f, FMUL(x, x)), 2.0f)));
        t = FSUB(t, FMUL(2.0f, u_atan(x)));
        psim10 = FADD(t, U_TWO_ATAN1);
    }
    float psix = FSUB(gfk_log(FDIV(z, z0)), c.psim);
    float psit = FSUB(gfk_log(FDIV(z, z0h)), c.psih);
    float psit2 = FSUB(gfk_log(FDIV(2.0f, z0h)), psih2);
    float psix10 = FSUB(gfk_log(FDIV(10.0f, z0)), psim10);
    c.u10 = FMUL(c.u1, FDIV(psix10, psix));
    c.v10 = FMUL(c.v1, FDIV(psix10, psix));
    c.th2 = FADD(c.ts, FMUL(FSUB(c.ta, c.ts), FDIV(c.chs, c.chs2)));
    c.q2 = FADD(c.qs, FMUL(FSUB(c.qa, c.qs), FDIV(psit2, psit)));
    return UCM_OK;
}

// ---------------------------------------------------------------------------
// state planes.  The launcher hands one device array of pointers, in the
// order of gpuwm/core/urban_ucm.py UCM_STATE_PLANES; layered entries point at
// a (4, ny, nx) block.
// ---------------------------------------------------------------------------


static inline float ucm_metal_wind(float u, float v) {
    float speed = FSQRT(FADD(gfk_pow(u, 2.0f), gfk_pow(v, 2.0f)));
    return speed < 1.0f ? 1.0f : speed;
}

kernel void ucm_noahmp_after_lsm(
    constant UcmNoahmpAfterLsmResources &r [[buffer(0)]],
    constant UcmNoahmpAfterLsmParams &q [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    const uint plane = uint(q.ny) * uint(q.nx);
    if (index >= plane || r.urban[index] == 0) return;
    const device float *forcing = r.float_bundle;
    const float frc = forcing[index];
    const float u1 = forcing[plane + index];
    const float v1 = forcing[2u * plane + index];
    const float t3d1 = forcing[3u * plane + index];
    const float qv1 = forcing[4u * plane + index];
    const float p8w1 = forcing[5u * plane + index];
    const float p8w2 = forcing[6u * plane + index];
    const float dz8w1 = forcing[7u * plane + index];
    const float hrang = forcing[8u * plane + index];
    const device float *tab = forcing + 9u * plane;
    const device float *glob = tab + uint(q.n_utype) * PT_NCOL;

    UcmCol c;
    c.utype = r.utype[index];
    c.jmonth = q.jmonth;
    c.ta = t3d1;
    c.qa = FDIV(qv1, FADD(1.0f, qv1));
    c.ua = ucm_metal_wind(u1, v1);
    c.u1 = u1;
    c.v1 = v1;
    c.ssg = r.field_swdown[index];
    c.llg = r.field_glw[index];
    c.rain = FMUL(FDIV(r.field_rainbl[index], q.dt), 3600.0f);
    c.rhoo = FDIV(FMUL(FADD(p8w2, p8w1), 0.5f),
        FMUL(FMUL(287.04f, c.ta), FADD(1.0f, FMUL(0.61f, c.qa))));
    c.za = FMUL(0.5f, dz8w1);
    c.delt = q.dt;
    c.omg = hrang;
    c.znt = r.field_znt[index];
    if (r.field_chs[index] < 1.0e-02f) r.field_chs[index] = 1.0e-02f;
    if (r.field_chs2[index] < 1.0e-02f) r.field_chs2[index] = 1.0e-02f;
    if (r.field_cqs2[index] < 1.0e-02f) r.field_cqs2[index] = 1.0e-02f;
    c.chs = r.field_chs[index];
    r.field_chs2[index] = r.field_cqs2[index];
    c.chs2 = r.field_chs2[index];
    c.tr = r.state_tr_urb2d[index];
    c.tb = r.state_tb_urb2d[index];
    c.tg = r.state_tg_urb2d[index];
    c.tc = r.state_tc_urb2d[index];
    c.qc = r.state_qc_urb2d[index];
    c.uc = r.state_uc_urb2d[index];
    c.xxxr = r.state_xxxr_urb2d[index];
    c.xxxb = r.state_xxxb_urb2d[index];
    c.xxxg = r.state_xxxg_urb2d[index];
    c.xxxc = r.state_xxxc_urb2d[index];
    c.cmr = r.state_cmr_sfcdif[index];
    c.chr = r.state_chr_sfcdif[index];
    c.cmc = r.state_cmc_sfcdif[index];
    c.chc = r.state_chc_sfcdif[index];
    c.cmgr = r.state_cmgr_sfcdif[index];
    c.chgr = r.state_chgr_sfcdif[index];
    c.cmcr = r.state_cmcr_urb2d[index];
    c.tgr = r.state_tgr_urb2d[index];
    c.drelr = r.state_drelr_urb2d[index];
    c.drelb = r.state_drelb_urb2d[index];
    c.drelg = r.state_drelg_urb2d[index];
    c.flxhumr = r.state_flxhumr_urb2d[index];
    c.flxhumb = r.state_flxhumb_urb2d[index];
    c.flxhumg = r.state_flxhumg_urb2d[index];
    for (uint level = 0; level < UCM_NL; ++level) {
        c.trl[level] = r.state_trl_urb3d[level * plane + index];
        c.tbl[level] = r.state_tbl_urb3d[level * plane + index];
        c.tgl[level] = r.state_tgl_urb3d[level * plane + index];
        c.tgrl[level] = r.state_tgrl_urb3d[level * plane + index];
        c.smr[level] = r.state_smr_urb3d[level * plane + index];
    }
    const int code = ucm_urban(tab, glob, r.switches, c);
    if (code != UCM_OK) {
        atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.status),
            1u << uint(code - 1), memory_order_relaxed);
        return;
    }
    const float omf = FSUB(1.0f, frc);
    r.field_albedo[index] = FADD(FMUL(frc, c.alb), FMUL(omf, r.field_albedo[index]));
    r.field_hfx[index] = FADD(FMUL(frc, c.sh), FMUL(omf, r.field_hfx[index]));
    r.field_qfx[index] = FADD(FMUL(frc, c.lh_kin), FMUL(omf, r.field_qfx[index]));
    r.field_lh[index] = FADD(FMUL(frc, c.lh), FMUL(omf, r.field_lh[index]));
    r.field_grdflx[index] = FADD(FMUL(frc, FMUL(c.g, -1.0f)), FMUL(omf, r.field_grdflx[index]));
    r.field_tsk[index] = FADD(FMUL(frc, c.ts), FMUL(omf, r.field_tsk[index]));
    r.field_qsfc[index] = FADD(FMUL(frc, c.qs), FMUL(omf, r.field_qsfc[index]));
    r.field_ust[index] = FADD(FMUL(frc, c.ust), FMUL(omf, r.field_ust[index]));
    r.state_ts_urb2d[index] = c.ts;
    r.state_tr_urb2d[index] = c.tr;
    r.state_tb_urb2d[index] = c.tb;
    r.state_tg_urb2d[index] = c.tg;
    r.state_tc_urb2d[index] = c.tc;
    r.state_qc_urb2d[index] = c.qc;
    r.state_uc_urb2d[index] = c.uc;
    r.state_tgr_urb2d[index] = c.tgr;
    r.state_cmcr_urb2d[index] = c.cmcr;
    r.state_flxhumr_urb2d[index] = c.flxhumr;
    r.state_flxhumb_urb2d[index] = c.flxhumb;
    r.state_flxhumg_urb2d[index] = c.flxhumg;
    r.state_drelr_urb2d[index] = c.drelr;
    r.state_drelb_urb2d[index] = c.drelb;
    r.state_drelg_urb2d[index] = c.drelg;
    r.state_xxxr_urb2d[index] = c.xxxr;
    r.state_xxxb_urb2d[index] = c.xxxb;
    r.state_xxxg_urb2d[index] = c.xxxg;
    r.state_xxxc_urb2d[index] = c.xxxc;
    r.state_sh_urb2d[index] = c.sh;
    r.state_lh_urb2d[index] = c.lh;
    r.state_g_urb2d[index] = c.g;
    r.state_rn_urb2d[index] = c.rn;
    r.state_psim_urb2d[index] = c.psim;
    r.state_psih_urb2d[index] = c.psih;
    r.state_gz1oz0_urb2d[index] = c.gz1oz0;
    r.state_u10_urb2d[index] = c.u10;
    r.state_v10_urb2d[index] = c.v10;
    r.state_th2_urb2d[index] = c.th2;
    r.state_q2_urb2d[index] = c.q2;
    r.state_ust_urb2d[index] = c.ust;
    r.state_cmr_sfcdif[index] = c.cmr;
    r.state_chr_sfcdif[index] = c.chr;
    r.state_cmgr_sfcdif[index] = c.cmgr;
    r.state_chgr_sfcdif[index] = c.chgr;
    r.state_cmc_sfcdif[index] = c.cmc;
    r.state_chc_sfcdif[index] = c.chc;
    r.state_akms_urb2d[index] = FDIV(FMUL(U_KARMAN, c.ust), FSUB(c.gz1oz0, c.psim));
    for (uint level = 0; level < UCM_NL; ++level) r.state_trl_urb3d[level * plane + index] = c.trl[level];
    for (uint level = 0; level < UCM_NL; ++level) r.state_smr_urb3d[level * plane + index] = c.smr[level];
    for (uint level = 0; level < UCM_NL; ++level) r.state_tgrl_urb3d[level * plane + index] = c.tgrl[level];
    for (uint level = 0; level < UCM_NL; ++level) r.state_tbl_urb3d[level * plane + index] = c.tbl[level];
    for (uint level = 0; level < UCM_NL; ++level) r.state_tgl_urb3d[level * plane + index] = c.tgl[level];
}
