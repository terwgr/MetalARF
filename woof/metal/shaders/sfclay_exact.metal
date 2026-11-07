#include <metal_stdlib>
#include "sfclay_column_exact_bindings.metalh"
#include "sfclay_preflight_exact_bindings.metalh"
#include "soft_word32.metalh"
#include "soft_gfk_atan.metalh"
using namespace metal;
static inline F operator/(F a,F b) { return aa_div_impl(a,b); }
static inline thread F &operator*=(thread F &a,F b) { a=a*b; return a; }
static inline F sf_min(F a,F b) { if (soft32_is_nan(a.bits)) return b; if (soft32_is_nan(b.bits)) return a; if (((a.bits|b.bits)&0x7fffffffu)==0) return f32_word(a.bits|b.bits); return a<b?a:b; }
static inline F sf_max(F a,F b) { if (soft32_is_nan(a.bits)) return b; if (soft32_is_nan(b.bits)) return a; if (((a.bits|b.bits)&0x7fffffffu)==0) return f32_word(a.bits&b.bits); return a>b?a:b; }
static inline F sf_sqrt(F a) { return f32_word(soft32_sqrt(a.bits)); }
static __attribute__((noinline)) F sf_log(F a) { return gfk_log(a); }
static __attribute__((noinline)) F sf_exp(F a) { return gfk_exp(a); }
static __attribute__((noinline)) F sf_pow(F a,F b) { return gfk_pow(a,b); }
static __attribute__((noinline)) F sf_atan(F a) { return f32_word(soft_gfk_atanf(a.bits)); }
static inline F sf_copysign(F a,F b) { return f32_word((a.bits&0x7fffffffu)|(b.bits&0x80000000u)); }
static inline bool surface_le_zero(F a) { return a <= F(0.0f); }
static inline bool surface_lt_zero(F a) { return a < F(0.0f); }
static inline F sf_log_zratio(F num,F z0) { return sf_log(num / z0); }
#define SF_TWO_ATAN1 F(1.57079637050628662109375f)
#define SF_SQRT3 F(1.73205077648162841796875f)
#define SF_FOUR_ATAN1_OVER_SQRT3 F(1.81379950046539306640625f)
#define SF_INV_1P1 F(9.0909087657928466796875e-1f)
#define SF_INV_2P5 F(4.000000059604644775390625e-1f)
#define SF_SQRT_0P71 F(8.42614948749542236328125e-1f)
#define SF_SQRT_0P60 F(7.74596691131591796875e-1f)
#define G F(9.81f)
#define RD F(287.0f)
#define RV F(461.6f)
#define CP F(1004.5f)
#define P0 F(100000.0f)
#define RCP F(0.2857142857142857f)
#define XLV F(2500000.0f)
#define SVP1 F(0.6112f)
#define SVP2 F(17.67f)
#define SVP3 F(29.65f)
#define SVPT0 F(273.15f)
#define EP2 F(0.6217504332755632f)
static __attribute__((noinline)) F sf_psim_classic_full(F z)
{
    // module_sf_sfclay.F:961-963 (sfclayinit).
    F x = sf_pow(1.0f - 16.0f * z, 0.25f);
    return 2.0f * sf_log(0.5f * (1.0f + x))
         + sf_log(0.5f * (1.0f + x * x)) - 2.0f * sf_atan(x)
         + SF_TWO_ATAN1;
}

static __attribute__((noinline)) F sf_psih_classic_full(F z)
{
    // module_sf_sfclay.F:964: (1-16*ZOLN)**0.5 is a powf call, not sqrtf.
    F y = sf_pow(1.0f - 16.0f * z, 0.5f);
    return 2.0f * sf_log(0.5f * (1.0f + y));
}

static __attribute__((noinline)) F sf_classic_table(F z, bool heat)
{
    F x = sf_min(sf_max(-z, 0.0f), 9.9999f) * 100.0f;
    int n = (int)x;
    F r = x - F(float(n));
    F z0 = -0.01f * F(float(n)), z1 = -0.01f * F(float(n + 1));
    F f0 = heat ? sf_psih_classic_full(z0) : sf_psim_classic_full(z0);
    F f1 = heat ? sf_psih_classic_full(z1) : sf_psim_classic_full(z1);
    return f0 + r * (f1 - f0);
}

static __attribute__((noinline)) F sf_psim_stable_full(F z)
{
    // sf_sfclayrev.F90:991.
    return -6.1f * sf_log(z + sf_pow(1.0f + sf_pow(z, 2.5f), SF_INV_2P5));
}

static __attribute__((noinline)) F sf_psih_stable_full(F z)
{
    // sf_sfclayrev.F90:999.
    return -5.3f * sf_log(z + sf_pow(1.0f + sf_pow(z, 1.1f), SF_INV_1P1));
}

static __attribute__((noinline)) F sf_psim_unstable_full(F z)
{
    // sf_sfclayrev.F90:1008-1014.  ym**2. and zolf**2. are products: the
    // strict -O2 build folds powf(x, 2.0) into x*x (header note); zolf**2
    // (integer power) is a product at any optimisation level.
    F x = sf_pow(1.0f - 16.0f * z, 0.25f);
    F psimk = 2.0f * sf_log(0.5f * (1.0f + x))
               + sf_log(0.5f * (1.0f + x * x)) - 2.0f * sf_atan(x)
               + SF_TWO_ATAN1;
    F ym = sf_pow(1.0f - 10.0f * z, 0.33f); // file literal .33
    F psimc = 1.5f * sf_log(__fdiv_rn((ym * ym + ym + 1.0f), 3.0f))
                - SF_SQRT3 * sf_atan(__fdiv_rn((2.0f * ym + 1.0f), SF_SQRT3))
                + SF_FOUR_ATAN1_OVER_SQRT3;
    return (psimk + z * z * psimc) / (1.0f + z * z);
}

static __attribute__((noinline)) F sf_psih_unstable_full(F z)
{
    // sf_sfclayrev.F90:1023-1029.  (1.-16.*zolf)**.5 is a powf call; yh**2.
    // and zolf**2. are products, as in psim above.
    F y = sf_pow(1.0f - 16.0f * z, 0.5f);
    F psihk = 2.0f * sf_log((1.0f + y) / 2.0f);
    F yh = sf_pow(1.0f - 34.0f * z, 0.33f);
    F psihc = 1.5f * sf_log(__fdiv_rn((yh * yh + yh + 1.0f), 3.0f))
                - SF_SQRT3 * sf_atan(__fdiv_rn((2.0f * yh + 1.0f), SF_SQRT3))
                + SF_FOUR_ATAN1_OVER_SQRT3;
    return (psihk + z * z * psihc) / (1.0f + z * z);
}

static __attribute__((noinline)) F sf_rev_table(F z, int which)
{
    // which: 0 psim stable, 1 psih stable, 2 psim unstable, 3 psih unstable
    bool unstable = which >= 2;
    z = unstable ? sf_min(z, 0.0f) : sf_max(z, 0.0f);
    F x = (unstable ? -z : z) * 100.0f;
    int n = (int)x;
    if (n + 1 >= 1000) {
        if (which == 0) return sf_psim_stable_full(z);
        if (which == 1) return sf_psih_stable_full(z);
        if (which == 2) return sf_psim_unstable_full(z);
        return sf_psih_unstable_full(z);
    }
    F r = x - F(float(n));
    F sign = unstable ? -1.0f : 1.0f;
    F z0 = sign * 0.01f * F(float(n));
    F z1 = sign * 0.01f * F(float(n + 1));
    F f0, f1;
    if (which == 0) { f0 = sf_psim_stable_full(z0); f1 = sf_psim_stable_full(z1); }
    else if (which == 1) { f0 = sf_psih_stable_full(z0); f1 = sf_psih_stable_full(z1); }
    else if (which == 2) { f0 = sf_psim_unstable_full(z0); f1 = sf_psim_unstable_full(z1); }
    else { f0 = sf_psih_unstable_full(z0); f1 = sf_psih_unstable_full(z1); }
    return f0 + r * (f1 - f0);
}

static __attribute__((noinline)) F sf_psim_stable(F z) { return sf_rev_table(z, 0); }
static __attribute__((noinline)) F sf_psih_stable(F z) { return sf_rev_table(z, 1); }
static __attribute__((noinline)) F sf_psim_unstable(F z) { return sf_rev_table(z, 2); }
static __attribute__((noinline)) F sf_psih_unstable(F z) { return sf_rev_table(z, 3); }

static __attribute__((noinline)) F sf_zolri_residual(thread F &zeta, F ri,
                                                   F z, F z0)
{
    if (zeta * ri < 0.0f) zeta = 0.0f;
    F zeta0 = zeta * z0 / z;
    F zeta3 = zeta + zeta0;
    F fm, fh;
    // sf_log_zratio, not bare logf: a degenerate z0 otherwise turns the
    // residual into zeta*Inf/Inf = NaN and poisons the whole zol solve.
    if (ri < 0.0f) {
        fm = sf_log_zratio(z + z0, z0)
           - (sf_psim_unstable(zeta3) - sf_psim_unstable(zeta0));
        fh = sf_log_zratio(z + z0, z0)
           - (sf_psih_unstable(zeta3) - sf_psih_unstable(zeta0));
    } else {
        fm = sf_log_zratio(z + z0, z0)
           - (sf_psim_stable(zeta3) - sf_psim_stable(zeta0));
        fh = sf_log_zratio(z + z0, z0)
           - (sf_psih_stable(zeta3) - sf_psih_stable(zeta0));
    }
    return zeta * fh / (fm * fm) - ri;
}

static __attribute__((noinline)) F sf_zolri(F ri, F z, F z0)
{
    F x1 = ri < 0.0f ? -5.0f : 0.0f;
    F x2 = ri < 0.0f ? 0.0f : 5.0f;
    F fx1 = sf_zolri_residual(x1, ri, z, z0);
    F fx2 = sf_zolri_residual(x2, ri, z, z0);
    F result = fabs(fx1) < fabs(fx2) ? x1 : x2;
    int iter = 0;
    while (fabs(x1 - x2) > 0.01f) {
        if (iter == 10 || fx1 == fx2) return result;
        if (fabs(fx2) < fabs(fx1)) {
            x1 = x1 - fx1 / (fx2 - fx1) * (x2 - x1);
            fx1 = sf_zolri_residual(x1, ri, z, z0);
            result = x1;
        } else {
            x2 = x2 - fx2 / (fx2 - fx1) * (x2 - x1);
            fx2 = sf_zolri_residual(x2, ri, z, z0);
            result = x2;
        }
        ++iter;
    }
    return result;
}

static __attribute__((noinline)) F sf_rev_heat_psi(F zol, F za,
                                                 F rough, F height)
{
    F zh = zol * (height + rough) / za;
    F z0 = zol * rough / za;
    // sf_sfclayrev.F90:545-559's test order: > 0, then == 0, else unstable.
    if (zol > 0.0f) return sf_psih_stable(zh) - sf_psih_stable(z0);
    if (zol == 0.0f) return 0.0f;
    return sf_psih_unstable(zh) - sf_psih_unstable(z0);
}


kernel void sfclay_preflight_exact(constant SfclayPreflightExactResources &r [[buffer(0)]],constant SfclayPreflightExactParams &p [[buffer(1)]],uint idx [[thread_position_in_grid]]) {
    if (idx>=uint(p.n)) return;
    F z0=F(r.znt[idx]), za=F(0.5f)*F(r.dz8w[idx]);
    F nums[5]={za+z0,F(2.f)+z0,F(10.f)+z0,za,F(2.f)};
    bool reject=soft32_is_nan(z0.bits);
    for (uint i=0;i<5;i++) {
        uint b=sf_log(nums[i]/z0).bits;
        reject = reject || soft32_is_nan(nums[i].bits) || ((b & 0x7f800000u)==0x7f800000u);
    }
    r.status[idx]=reject?1.0f:0.0f;
}

kernel void sfclay_column_exact(constant SfclayColumnExactResources &r [[buffer(0)]],constant SfclayColumnExactParams &p [[buffer(1)]],uint idx [[thread_position_in_grid]]) {
    if (idx>=uint(p.n)) return;
    const int option=1, isfflx=p.isfflx, isftcflx=p.isftcflx, iz0tlnd=p.iz0tlnd;
    const F dx=F(p.dx);
    const F karman = 0.4f, ep1 = RV / RD - 1.0f, xka = 2.4e-5f;
    bool land = F(r.xland[idx]) < 1.5f;
    F uu = F(r.u[idx]), vv = F(r.v[idx]), temp = F(r.t[idx]), qvx = F(r.qv[idx]);
    F press = F(r.p[idx]), ps = F(r.psfc[idx]), ground_t = F(r.tsk[idx]);
    F z0 = F(r.znt[idx]), old_ust = F(r.ust[idx]), old_mol = F(r.mol[idx]);
    F old_ustm = F(r.ustm[idx]);
    F old_zol = F(r.zol[idx]);
    F old_hfx = F(r.hfx[idx]), old_qfx = F(r.qfx[idx]), qs = F(r.qsfc[idx]);

    // Both schemes: THGB=TSK*(P1000mb/PSFCPA)**ROVCP, and the air's Exner
    // factor from pressure in cb, THCON=(P1000mb*0.001/PL)**ROVCP with
    // PL=P/1000. (sf_sfclayrev.F90:230,255-259; module_sf_sfclay.F:405,
    // 430-434).  P1000mb*0.001 rounds to 100.0000076 in float32, so this is
    // not (P0/p)**ROVCP.
    F thgb = ground_t * sf_pow(P0 / ps, RCP);
    F pl = __fdiv_rn(press, 1000.0f);
    F thx = temp * sf_pow(__fdiv_rn(__fmul_rn(P0, 0.001f), pl), RCP);
    F thvx = thx * (1.0f + ep1 * qvx);
    F tv = temp * (1.0f + ep1 * qvx);
    F cpm = CP * (1.0f + 0.8f * qvx);
    F psfc_cb = __fdiv_rn(ps, 1000.0f);
    F es = SVP1 * sf_exp(SVP2 * (ground_t - SVPT0) / (ground_t - SVP3));
    if (!land && F(r.lakemask[idx]) == 0.0f) es *= 0.98f;
    if (!land || surface_le_zero(qs)) qs = EP2 * es / (psfc_cb - es);
    F es_air = SVP1 * sf_exp(SVP2 * (temp - SVPT0) / (temp - SVP3));
    F qgh = EP2 * es_air / (pl - es_air);
    // RHOX=PSFC*1000./(R*SCR4) with PSFC=PSFCPA/1000. (sf_sfclayrev.F90:221,
    // 300; module_sf_sfclay.F:396,475): the cb round trip is WRF's word.
    F rho = __fmul_rn(psfc_cb, 1000.0f) / (RD * tv);
    F za = 0.5f * F(r.dz8w[idx]);
    F gz1, gz2, gz10;
    if (option == 91) {
        gz1 = sf_log_zratio(za, z0);
        gz2 = sf_log_zratio(2.0f, z0);
        gz10 = sf_log_zratio(10.0f, z0);
    } else {
        gz1 = sf_log_zratio(za + z0, z0);
        gz2 = sf_log_zratio(2.0f + z0, z0);
        gz10 = sf_log_zratio(10.0f + z0, z0);
    }

    F tskv = thgb * (1.0f + ep1 * qs);
    F dthv = thvx - tskv;
    F wspd0 = sf_sqrt(uu * uu + vv * vv), vconv;   // WRF's SQRT, not hypot
    if (land) {
        F fluxc = sf_max(__fdiv_rn(old_hfx / rho, CP) + ep1 * tskv * old_qfx / rho,
                           0.0f);
        vconv = sf_pow(G / ground_t * F(r.pblh[idx]) * fluxc, 0.33f);
    } else {
        vconv = sf_sqrt(sf_max(-dthv, 0.0f));
    }
    F vsgd = 0.32f * sf_pow(sf_max(__fdiv_rn(dx, 5000.0f) - 1.0f, 0.0f), 0.33f);
    F wspd = sf_max(sf_sqrt(wspd0 * wspd0 + vconv * vconv + vsgd * vsgd),
                       0.1f);
    F br = G / thx * za * dthv / (wspd * wspd);
    if (surface_lt_zero(old_mol)) br = sf_min(br, 0.0f);

    F psim = 0.0f, psih = 0.0f, psim10 = 0.0f, psih10 = 0.0f;
    F psim2 = 0.0f, psih2 = 0.0f, pq = 0.0f, pq2 = 0.0f, pq10 = 0.0f;
    F zol = old_zol, regime, rmol;
    if (option == 91) {
        if (br >= 0.2f) {
            regime = 1.0f;
            psim = sf_max(-10.0f * gz1, -10.0f); psih = psim;
            psim10 = sf_max(10.0f / za * psim, -10.0f); psih10 = psim10;
            psim2 = sf_max(2.0f / za * psim, -10.0f); psih2 = psim2;
            F za_over_l = old_ust < 0.01f ? br * gz1
                  : karman * G / thx * za * old_mol / (old_ust * old_ust);
            rmol = sf_min(za_over_l, 9.999f) / za;
        } else if (br > 0.0f) {
            regime = 2.0f;
            psim = sf_max(-5.0f * br * gz1 / (1.1f - 5.0f * br), -10.0f);
            psih = psim;
            psim10 = sf_max(10.0f / za * psim, -10.0f); psih10 = psim10;
            psim2 = sf_max(2.0f / za * psim, -10.0f); psih2 = psim2;
            zol = br * gz1 / (1.00001f - 5.0f * br);
            if (zol > 0.5f) {
                zol = (1.89f * gz1 + 44.2f) * br * br
                    + (1.18f * gz1 - 1.37f) * br;
                zol = sf_min(zol, 9.999f);
            }
            rmol = zol / za;
        } else if (br == 0.0f) {
            regime = 3.0f;
            zol = old_ust < 0.01f ? br * gz1
                  : karman * G / thx * za * old_mol / (old_ust * old_ust);
            rmol = zol / za;
        } else {
            regime = 4.0f;
            zol = old_ust < 0.01f ? br * gz1
                  : karman * G / thx * za * old_mol / (old_ust * old_ust);
            F zol10 = sf_max(sf_min(10.0f / za * zol, 0.0f), -9.9999f);
            F zol2 = sf_max(sf_min(2.0f / za * zol, 0.0f), -9.9999f);
            zol = sf_max(sf_min(zol, 0.0f), -9.9999f);
            psim = sf_classic_table(zol, false); psih = sf_classic_table(zol, true);
            psim10 = sf_classic_table(zol10, false); psih10 = sf_classic_table(zol10, true);
            psim2 = sf_classic_table(zol2, false); psih2 = sf_classic_table(zol2, true);
            psih = sf_min(psih, 0.9f * gz1); psim = sf_min(psim, 0.9f * gz1);
            psih2 = sf_min(psih2, 0.9f * gz2);
            psim10 = sf_min(psim10, 0.9f * gz10);
            psih10 = sf_min(psih10, 0.9f * gz10);
            rmol = zol / za;
        }
    } else {
        zol = 0.0f;
        if (br > 0.0f) zol = sf_zolri(sf_min(br, 250.0f), za, z0);
        else if (br < 0.0f)
            zol = old_ust < 0.001f ? br * gz1
                  : sf_zolri(sf_max(br, -250.0f), za, z0);
        F zz = zol * (za + z0) / za, z10 = zol * (10.0f + z0) / za;
        F z2 = zol * (2.0f + z0) / za, zz0 = zol * z0 / za;
        // ZL=(0.01)/ZA*ZOL over land (sf_sfclayrev.F90:413), WRF's order.
        F scalar_z = land ? __fdiv_rn(0.01f, za) * zol : zz0;
        if (br > 0.0f) {
            regime = 1.0f;
            psim = sf_psim_stable(zz) - sf_psim_stable(zz0);
            psih = sf_psih_stable(zz) - sf_psih_stable(zz0);
            psim10 = sf_psim_stable(z10) - sf_psim_stable(zz0);
            psih10 = sf_psih_stable(z10) - sf_psih_stable(zz0);
            psim2 = sf_psim_stable(z2) - sf_psim_stable(zz0);
            psih2 = sf_psih_stable(z2) - sf_psih_stable(zz0);
            pq = sf_psih_stable(zol) - sf_psih_stable(scalar_z);
            pq2 = sf_psih_stable(2.0f / za * zol) - sf_psih_stable(scalar_z);
            pq10 = sf_psih_stable(10.0f / za * zol) - sf_psih_stable(scalar_z);
        } else if (br == 0.0f) {
            regime = 3.0f; zol = 0.0f;
        } else {
            regime = 4.0f;
            psim = sf_psim_unstable(zz) - sf_psim_unstable(zz0);
            psih = sf_psih_unstable(zz) - sf_psih_unstable(zz0);
            psim10 = sf_psim_unstable(z10) - sf_psim_unstable(zz0);
            psih10 = sf_psih_unstable(z10) - sf_psih_unstable(zz0);
            psim2 = sf_psim_unstable(z2) - sf_psim_unstable(zz0);
            psih2 = sf_psih_unstable(z2) - sf_psih_unstable(zz0);
            pq = sf_psih_unstable(zol) - sf_psih_unstable(scalar_z);
            pq2 = sf_psih_unstable(2.0f / za * zol) - sf_psih_unstable(scalar_z);
            pq10 = sf_psih_unstable(10.0f / za * zol) - sf_psih_unstable(scalar_z);
            psih = sf_min(psih, 0.9f * gz1); psim = sf_min(psim, 0.9f * gz1);
            psih2 = sf_min(psih2, 0.9f * gz2);
            psim10 = sf_min(psim10, 0.9f * gz10);
            psih10 = sf_min(psih10, 0.9f * gz10);
        }
        rmol = zol / za;
    }

    F dtg = thx - thgb;
    F psix = gz1 - psim, psix10 = gz10 - psim10;
    F psit = option == 91 ? sf_max(gz1 - psih, 2.0f) : gz1 - psih;
    F psit2 = gz2 - psih2;
    F zl = land ? 0.01f : z0;
    F psiq = sf_log(__fdiv_rn(karman * old_ust * za, xka) + __fdiv_rn(za, zl))
              - (option == 91 ? psih : pq);
    F psiq2 = sf_log(__fdiv_rn(karman * old_ust * 2.0f, xka) + __fdiv_rn(2.0f, zl))
               - (option == 91 ? psih2 : pq2);
    F psiq10 = sf_log(__fdiv_rn(karman * old_ust * 10.0f, xka) + __fdiv_rn(10.0f, zl))
                - (option == 91 ? psih10 : pq10);

    if (!land) {
        F visc = (1.32f + 0.009f * (temp - 273.15f)) * 1.0e-5f;
        F restar = old_ust * z0 / visc;
        // Z0T=MIN(Z0T,1.0E-4) then MAX(Z0T,2.0E-9), WRF's order.
        F z0t = sf_max(sf_min(5.5e-5f * sf_pow(restar, -0.60f), 1.0e-4f),
                         2.0e-9f);
        if (option == 91) {
            psiq = sf_max(sf_log((za + z0t) / z0t) - psih, 2.0f);
            psit = sf_max(sf_log((za + z0t) / z0t) - psih, 2.0f);
            psiq2 = sf_max(sf_log((2.0f + z0t) / z0t) - psih2, 2.0f);
            psit2 = sf_max(sf_log((2.0f + z0t) / z0t) - psih2, 2.0f);
            psiq10 = sf_max(sf_log((10.0f + z0t) / z0t) - psih10, 2.0f);
        } else {
            psih = sf_rev_heat_psi(zol, za, z0t, za);
            psih2 = sf_rev_heat_psi(zol, za, z0t, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0t, 10.0f);
            psit = sf_log((za + z0t) / z0t) - psih;
            psit2 = sf_log((2.0f + z0t) / z0t) - psih2;
            psiq = psit; psiq2 = psit2;
            psiq10 = sf_log((10.0f + z0t) / z0t) - psih10;
        }
    }

    if (isftcflx == 1 && !land) {
        F z0q = 1.0e-4f;
        if (option == 91) {
            psiq = sf_log(__fdiv_rn(za, z0q)) - psih;
            psiq2 = sf_log(2.0f / z0q) - psih2;
            psiq10 = sf_log(10.0f / z0q) - psih10;
        } else {
            psih = sf_rev_heat_psi(zol, za, z0q, za);
            psih2 = sf_rev_heat_psi(zol, za, z0q, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0q, 10.0f);
            psiq = sf_log(__fdiv_rn((za + z0q), z0q)) - psih;
            psiq2 = sf_log((2.0f + z0q) / z0q) - psih2;
            psiq10 = sf_log((10.0f + z0q) / z0q) - psih10;
        }
        psit = psiq; psit2 = psiq2;
    } else if (isftcflx == 2 && !land) {
        F visc = (1.32f + 0.009f * (temp - 273.15f)) * 1.0e-5f;
        F restar = old_ust * z0 / visc;
        // 0.40*(7.3*SQRT(SQRT(RESTAR))*SQRT(0.71)-5.): two square roots,
        // not a quarter power (sf_sfclayrev.F90:636,668;
        // module_sf_sfclay.F:761-762).
        F gz0t = 0.4f * (7.3f * sf_sqrt(sf_sqrt(restar)) * SF_SQRT_0P71 - 5.0f);
        F gz0q = 0.4f * (7.3f * sf_sqrt(sf_sqrt(restar)) * SF_SQRT_0P60 - 5.0f);
        if (option == 91) {
            psit = gz1 - psih + gz0t; psiq = gz1 - psih + gz0q;
            psit2 = gz2 - psih2 + gz0t; psiq2 = gz2 - psih2 + gz0q;
            psiq10 = gz10 - psih + gz0q;
        } else {
            F z0t = z0 / sf_exp(gz0t), z0q = z0 / sf_exp(gz0q);
            F pht = sf_rev_heat_psi(zol, za, z0t, za);
            F pht2 = sf_rev_heat_psi(zol, za, z0t, 2.0f);
            psit = sf_log((za + z0t) / z0t) - pht;
            psit2 = sf_log((2.0f + z0t) / z0t) - pht2;
            psih = sf_rev_heat_psi(zol, za, z0q, za);
            psih2 = sf_rev_heat_psi(zol, za, z0q, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0q, 10.0f);
            psiq = sf_log((za + z0q) / z0q) - psih;
            psiq2 = sf_log((2.0f + z0q) / z0q) - psih2;
            psiq10 = sf_log((10.0f + z0q) / z0q) - psih10;
        }
    }

    F ck = (karman / psix10) * (karman / psiq10);
    F cd = (karman / psix10) * (karman / psix10);
    F cka = (karman / psix) * (karman / psiq);
    F cda = (karman / psix) * (karman / psix);
    bool scalar_exp_overflow = false;
    if (iz0tlnd >= 1 && land) {
        F visc = (1.32f + 0.009f * (temp - 273.15f)) * 1.0e-5f;
        F restar = old_ust * z0 / visc;
        // CZIL=10.0**(-0.40*(ZL/0.07)): the quotient first
        // (sf_sfclayrev.F90:716; module_sf_sfclay.F:786).
        F czil = iz0tlnd == 1
                  ? sf_pow(10.0f, -(__fdiv_rn(z0, 0.07f) * 0.40f)) : 0.1f;
        if (option == 91) {
            F add = czil * karman * sf_sqrt(restar);
            psit = psiq = gz1 - psih + add;
            psit2 = psiq2 = gz2 - psih2 + add;
        } else {
            F scalar_exponent = czil * karman * sf_sqrt(restar);
            // Revised MM5 land scalar roughness overflow defect:
            // sf_sfclayrev.F90:723 overflows EXP, makes Z0T zero and then
            // FH=Inf and TH2/T2/Q2=NaN.  This is the last FP32 argument
            // whose exponential is finite.  Keep WRF's zero-exchange limit
            // beyond it, but diagnose the scalar resistance in log space.
            // The ordinary arm keeps its exact expression and FP32 word.
            scalar_exp_overflow = scalar_exponent > 88.72283172607421875f;
            F z0t = scalar_exp_overflow ? 0.0f : z0 / sf_exp(scalar_exponent);
            psih = sf_rev_heat_psi(zol, za, z0t, za);
            psih2 = sf_rev_heat_psi(zol, za, z0t, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0t, 10.0f);
            psit = psiq = scalar_exp_overflow
                ? sf_log_zratio(za, z0) + scalar_exponent - psih
                : sf_log((za + z0t) / z0t) - psih;
            psit2 = psiq2 = scalar_exp_overflow
                ? sf_log_zratio(2.0f, z0) + scalar_exponent - psih2
                : sf_log((2.0f + z0t) / z0t) - psih2;
        }
    }

    F new_ust = 0.5f * old_ust + 0.5f * karman * wspd / psix;
    // TKE coupling (module_sf_sfclay.F:800-804,
    // physics_mmm/sf_sfclayrev.F90:759-763): USTM repeats the UST
    // relaxation on the wind speed WITHOUT the Beljaars/Mahrt-Sun
    // vconv/vsgd correction and without WSPD's 0.1 floor, and takes
    // none of the land floor UST gets below.
    F wspdi = sf_sqrt(uu * uu + vv * vv);
    F new_ustm = 0.5f * old_ustm + 0.5f * karman * wspdi / psix;
    F u10 = uu * psix10 / psix, v10 = vv * psix10 / psix;
    F th2 = thgb + (thx - thgb) * psit2 / psit;
    F q2 = qs + (qvx - qs) * psiq2 / psiq;
    F t2 = th2 * sf_pow(__fdiv_rn(ps, P0), RCP);
    if (land) new_ust = sf_max(new_ust, option == 91 ? 0.1f : 0.001f);
    F mol_numerator = karman * (thx - thgb);
    F new_mol = scalar_exp_overflow ? sf_copysign(0.0f, mol_numerator)
                                       : mol_numerator / psit;

    F z0out = z0;
    if (isfflx && !land) {
        z0out = sf_min(__fdiv_rn(0.0185f * new_ust * new_ust, G)
                      + 0.11f * 1.5e-5f / new_ust, 2.85e-3f);
        if (isftcflx != 0) {
            F zw = sf_min(sf_pow(__fdiv_rn(new_ust, 1.06f), 0.3f), 1.0f);
            F zn1 = __fdiv_rn(0.011f * new_ust * new_ust, G) + 1.59e-5f;
            F zn2 = 10.0f * sf_exp(-9.5f * sf_pow(new_ust, -0.3333f))
                       + 0.11f * 1.5e-5f / sf_max(new_ust, 0.01f);
            z0out = sf_min(sf_max((1.0f - zw) * zn1 + zw * zn2, 1.27e-7f),
                           2.85e-3f);
        }
    }

    F flhc = 0.0f, flqc = 0.0f, new_hfx = 0.0f, new_qfx = 0.0f;
    // Flux-off carry defect: WRF skips the CHS/CHS2/CQS2 assignments
    // (sf_sfclayrev.F90:794,893,902,903), retaining the caller's buffers.
    // LH is unassigned in WRF on this path; WOOF defines it as zero.
    F lh = 0.0f, chs = isfflx ? 0.0f : F(r.chs[idx]);
    F chs2 = isfflx ? 0.0f : F(r.chs2[idx]);
    F cqs2 = isfflx ? 0.0f : F(r.cqs2[idx]);
    if (isfflx) {
        F flqc_numerator = rho * F(r.mavail[idx]) * new_ust * karman;
        flqc = scalar_exp_overflow ? sf_copysign(0.0f, flqc_numerator)
                                  : flqc_numerator / psiq;
        if (fabs(thx - thgb) > 1.0e-5f)
            flhc = cpm * rho * new_ust * new_mol / (thx - thgb);
        new_qfx = flqc * (qs - qvx); lh = XLV * new_qfx;
        new_hfx = flhc * (thgb - thx);
        F chs_numerator = new_ust * karman;
        chs = scalar_exp_overflow ? sf_copysign(0.0f, chs_numerator) : chs_numerator / psiq;
        cqs2 = scalar_exp_overflow ? sf_copysign(0.0f, chs_numerator) : chs_numerator / psiq2;
        chs2 = scalar_exp_overflow ? sf_copysign(0.0f, chs_numerator) : chs_numerator / psit2;
    }

    r.znt[idx] = float(z0out); r.ust[idx] = float(new_ust); r.ustm[idx] = float(new_ustm);
    r.mol[idx] = float(new_mol);
    r.hfx[idx] = float(new_hfx); r.qfx[idx] = float(new_qfx); r.qsfc[idx] = float(qs);
    r.zol[idx] = float(zol); r.regime[idx] = float(regime); r.psim[idx] = float(psim);
    r.psih[idx] = float(psih); r.fm[idx] = float(psix); r.fh[idx] = float(psit); r.lh[idx] = float(lh);
    r.u10[idx] = float(u10); r.v10[idx] = float(v10); r.th2[idx] = float(th2); r.t2[idx] = float(t2);
    r.q2[idx] = float(q2);
    // SFCLAYREV leaves the incoming exchange coefficients alone when
    // ISFFLX is zero. Clearing them erased the previous surface call.
    if (isfflx) {
        r.chs[idx] = float(chs); r.chs2[idx] = float(chs2); r.cqs2[idx] = float(cqs2);
    }
    r.flhc[idx] = float(flhc); r.flqc[idx] = float(flqc);
    r.qgh[idx] = float(qgh); r.rmol[idx] = float(rmol); r.wspd[idx] = float(wspd);
    r.br[idx] = float(br); r.gz1oz0[idx] = float(gz1); r.cpm[idx] = float(cpm);
    r.ck[idx] = float(ck); r.cka[idx] = float(cka); r.cd[idx] = float(cd); r.cda[idx] = float(cda);
}
