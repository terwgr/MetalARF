#include <metal_stdlib>
#include "sfclay_column_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_gfk_log.metalh"
#include "soft_libm64.metalh"
using namespace metal;
#pragma clang fp contract(off)
// FP32 logf follows the in-tree pinned glibc gfk_log bit implementation.
// The CUDA-defined zero/subnormal-z0 rescue widens exact FP32 bits and uses
// the shared software binary64 log. These helpers remain uncompiled and
// numerically unverified as a Metal composition.
constant float G=9.81f, RD=287.0f, RV=461.6f, CP=1004.5f;
constant float P0=100000.0f, RCP=0.2857142984867096f;
constant float SVP1=0.6112f, SVP2=17.67f, SVPT0=273.15f, SVP3=29.65f;
constant float EP2=0.6217157244682312f, XLV=2.5e6f;

// gpuwm/core/kernels/sfclay.cu
//
// WRF v4.6.1 MM5 surface-layer schemes, transcribed from
//   phys/module_sf_sfclay.F          SFCLAY1D (option 91), and
//   phys/physics_mmm/sf_sfclayrev.F90 sf_sfclayrev_run (option 1).
// module_sf_sfclayrev.F is only the WRF/CCPP array wrapper.  One CUDA thread
// handles one (j,i) surface column; inputs are the lowest mass-level values
// already interpolated to mass points exactly as the WRF wrappers do.
//
// The incoming znt, ust, ustm, mol, hfx, qfx, qsfc, and zol are previous-
// step/inout fields.  This matters: classic unstable z/L uses old MOL, its
// strong-stable branch leaves ZOL untouched, and both schemes average newly
// diagnosed u* with old UST and old USTM.  Outputs remain FP32 model state;
// gpuwm.verify.npref.np_sfclay is the float64 transcription mirror.

static inline float sf_div_rn(float a, float b) { return a / b; }
static inline float sf_logf(float x) { return as_type<float>(soft_gfk_log(as_type<uint>(x))); }
static inline float sf_hypotf(float x, float y) {
    // Metal Shading Language has no hypot overload. Scale before squaring so
    // finite magnitudes do not overflow; preserve hypot's Inf-over-NaN rule.
    if (isinf(x) || isinf(y)) return INFINITY;
    if (isnan(x) || isnan(y)) return NAN;
    float ax = fabs(x), ay = fabs(y);
    float hi = fmax(ax, ay), lo = fmin(ax, ay);
    if (hi == 0.0f) return 0.0f;
    float ratio = lo / hi;
    return hi * sqrt(1.0f + ratio * ratio);
}
static inline float sf_log_zratio(float num, float z0) {
    if (isnan(num) || isnan(z0)) return num + z0;
    const float r = sf_logf(num / z0);
    if (isfinite(r)) return r;
    ulong nd = soft64_from_f32_bits(as_type<uint>(num));
    ulong zd = soft64_from_f32_bits(as_type<uint>(z0));
    if (!soft64_greater(zd, 0ul)) zd = 0x3810000000000000ul;
    return as_type<float>(soft64_to_f32_bits(soft64_sub(
        soft64_plm_log(nd), soft64_plm_log(zd))));
}

static inline float sf_psim_classic_full(float z)
{
    float x = pow(1.0f - 16.0f * z, 0.25f);
    return 2.0f * sf_logf(0.5f * (1.0f + x))
         + sf_logf(0.5f * (1.0f + x * x)) - 2.0f * atan(x)
         + 2.0f * atan(1.0f);
}

static inline float sf_psih_classic_full(float z)
{
    float y = sqrt(1.0f - 16.0f * z);
    return 2.0f * sf_logf(0.5f * (1.0f + y));
}

static inline float sf_classic_table(float z, bool heat)
{
    float x = fmin(fmax(-z, 0.0f), 9.9999f) * 100.0f;
    int n = (int)x;
    float r = x - (float)n;
    float z0 = -0.01f * (float)n, z1 = -0.01f * (float)(n + 1);
    float f0 = heat ? sf_psih_classic_full(z0) : sf_psim_classic_full(z0);
    float f1 = heat ? sf_psih_classic_full(z1) : sf_psim_classic_full(z1);
    return f0 + r * (f1 - f0);
}

static inline float sf_psim_stable_full(float z)
{
    return -6.1f * sf_logf(z + pow(1.0f + pow(z, 2.5f), 1.0f / 2.5f));
}

static inline float sf_psih_stable_full(float z)
{
    return -5.3f * sf_logf(z + pow(1.0f + pow(z, 1.1f), 1.0f / 1.1f));
}

static inline float sf_psim_unstable_full(float z)
{
    float x = pow(1.0f - 16.0f * z, 0.25f);
    float psimk = 2.0f * sf_logf(0.5f * (1.0f + x))
               + sf_logf(0.5f * (1.0f + x * x)) - 2.0f * atan(x)
               + 2.0f * atan(1.0f);
    float ym = pow(1.0f - 10.0f * z, 0.33f); // file literal .33
    float rt3 = sqrt(3.0f);
    float psimc = 1.5f * sf_logf(sf_div_rn((ym * ym + ym + 1.0f), 3.0f))
                - rt3 * atan(sf_div_rn((2.0f * ym + 1.0f), rt3))
                + 4.0f * atan(1.0f) / rt3;
    return (psimk + z * z * psimc) / (1.0f + z * z);
}

static inline float sf_psih_unstable_full(float z)
{
    float y = sqrt(1.0f - 16.0f * z);
    float psihk = 2.0f * sf_logf((1.0f + y) / 2.0f);
    float yh = pow(1.0f - 34.0f * z, 0.33f);
    float rt3 = sqrt(3.0f);
    float psihc = 1.5f * sf_logf(sf_div_rn((yh * yh + yh + 1.0f), 3.0f))
                - rt3 * atan(sf_div_rn((2.0f * yh + 1.0f), rt3))
                + 4.0f * atan(1.0f) / rt3;
    return (psihk + z * z * psihc) / (1.0f + z * z);
}

static inline float sf_rev_table(float z, int which)
{
    // which: 0 psim stable, 1 psih stable, 2 psim unstable, 3 psih unstable
    bool unstable = which >= 2;
    z = unstable ? fmin(z, 0.0f) : fmax(z, 0.0f);
    float x = (unstable ? -z : z) * 100.0f;
    int n = (int)x;
    if (n + 1 >= 1000) {
        if (which == 0) return sf_psim_stable_full(z);
        if (which == 1) return sf_psih_stable_full(z);
        if (which == 2) return sf_psim_unstable_full(z);
        return sf_psih_unstable_full(z);
    }
    float r = x - (float)n;
    float sign = unstable ? -1.0f : 1.0f;
    float z0 = sign * 0.01f * (float)n;
    float z1 = sign * 0.01f * (float)(n + 1);
    float f0, f1;
    if (which == 0) { f0 = sf_psim_stable_full(z0); f1 = sf_psim_stable_full(z1); }
    else if (which == 1) { f0 = sf_psih_stable_full(z0); f1 = sf_psih_stable_full(z1); }
    else if (which == 2) { f0 = sf_psim_unstable_full(z0); f1 = sf_psim_unstable_full(z1); }
    else { f0 = sf_psih_unstable_full(z0); f1 = sf_psih_unstable_full(z1); }
    return f0 + r * (f1 - f0);
}

static inline float sf_psim_stable(float z) { return sf_rev_table(z, 0); }
static inline float sf_psih_stable(float z) { return sf_rev_table(z, 1); }
static inline float sf_psim_unstable(float z) { return sf_rev_table(z, 2); }
static inline float sf_psih_unstable(float z) { return sf_rev_table(z, 3); }

static inline float sf_zolri_residual(thread float &zeta, float ri,
                                                   float z, float z0)
{
    if (zeta * ri < 0.0f) zeta = 0.0f;
    float zeta0 = zeta * z0 / z;
    float zeta3 = zeta + zeta0;
    float fm, fh;
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

static inline float sf_zolri(float ri, float z, float z0)
{
    float x1 = ri < 0.0f ? -5.0f : 0.0f;
    float x2 = ri < 0.0f ? 0.0f : 5.0f;
    float fx1 = sf_zolri_residual(x1, ri, z, z0);
    float fx2 = sf_zolri_residual(x2, ri, z, z0);
    float result = fabs(fx1) < fabs(fx2) ? x1 : x2;
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

static inline float sf_rev_heat_psi(float zol, float za,
                                                 float rough, float height)
{
    float zh = zol * (height + rough) / za;
    float z0 = zol * rough / za;
    if (zol > 0.0f) return sf_psih_stable(zh) - sf_psih_stable(z0);
    if (zol < 0.0f) return sf_psih_unstable(zh) - sf_psih_unstable(z0);
    return 0.0f;
}

kernel void sfclay_column(constant SfclayColumnResources &r [[buffer(0)]],
                           constant SfclayColumnParams &q [[buffer(1)]],
                           uint col [[thread_position_in_grid]]) {
    if (uint(q.n) != uint(q.nx) * uint(q.ny) || col >= uint(q.n)) return;

    const uint idx = col;

    const float karman = 0.4f, ep1 = RV / RD - 1.0f, xka = 2.4e-5f;
    bool land = r.xland[idx] < 1.5f;
    float uu = r.u[idx], vv = r.v[idx], temp = r.t[idx], qvx = r.qv[idx];
    float press = r.p[idx], ps = r.psfc[idx], ground_t = r.tsk[idx];
    float z0 = r.znt[idx], old_ust = r.ust[idx], old_mol = r.mol[idx];
    float old_ustm = r.ustm[idx];
    float old_zol = r.zol[idx];
    float old_hfx = r.hfx[idx], old_qfx = r.qfx[idx], qs = r.qsfc[idx];

    float thgb = ground_t * pow(P0 / ps, RCP);
    float thx = temp * pow(P0 / press, RCP);
    float thvx = thx * (1.0f + ep1 * qvx);
    float tv = temp * (1.0f + ep1 * qvx);
    float cpm = CP * (1.0f + 0.8f * qvx);
    float es = SVP1 * exp(SVP2 * (ground_t - SVPT0) / (ground_t - SVP3));
    if (!land && r.lakemask[idx] == 0.0f) es *= 0.98f;
    if (!land || qs <= 0.0f) qs = EP2 * es / (sf_div_rn(ps, 1000.0f) - es);
    float es_air = SVP1 * exp(SVP2 * (temp - SVPT0) / (temp - SVP3));
    float qgh = EP2 * es_air / (sf_div_rn(press, 1000.0f) - es_air);
    float rho = ps / (RD * tv);
    float za = 0.5f * r.dz8w[idx];
    float gz1, gz2, gz10;
    if (q.option == 91) {
        gz1 = sf_log_zratio(za, z0);
        gz2 = sf_log_zratio(2.0f, z0);
        gz10 = sf_log_zratio(10.0f, z0);
    } else {
        gz1 = sf_log_zratio(za + z0, z0);
        gz2 = sf_log_zratio(2.0f + z0, z0);
        gz10 = sf_log_zratio(10.0f + z0, z0);
    }

    float tskv = thgb * (1.0f + ep1 * qs);
    float dthv = thvx - tskv;
    float wspd0 = sf_hypotf(uu, vv), vconv;
    if (land) {
        float fluxc = fmax(sf_div_rn(old_hfx / rho, CP) + ep1 * tskv * old_qfx / rho,
                           0.0f);
        vconv = pow(G / ground_t * r.pblh[idx] * fluxc, 0.33f);
    } else {
        vconv = sqrt(fmax(-dthv, 0.0f));
    }
    float vsgd = 0.32f * pow(fmax(sf_div_rn(q.dx, 5000.0f) - 1.0f, 0.0f), 0.33f);
    float wspd = fmax(sqrt(wspd0 * wspd0 + vconv * vconv + vsgd * vsgd),
                       0.1f);
    float br = G / thx * za * dthv / (wspd * wspd);
    if (old_mol < 0.0f) br = fmin(br, 0.0f);

    float psim = 0.0f, psih = 0.0f, psim10 = 0.0f, psih10 = 0.0f;
    float psim2 = 0.0f, psih2 = 0.0f, pq = 0.0f, pq2 = 0.0f, pq10 = 0.0f;
    float zol = old_zol, regime, rmol;
    if (q.option == 91) {
        if (br >= 0.2f) {
            regime = 1.0f;
            psim = fmax(-10.0f * gz1, -10.0f); psih = psim;
            psim10 = fmax(10.0f / za * psim, -10.0f); psih10 = psim10;
            psim2 = fmax(2.0f / za * psim, -10.0f); psih2 = psim2;
            float za_over_l = old_ust < 0.01f ? br * gz1
                  : karman * G / thx * za * old_mol / (old_ust * old_ust);
            rmol = fmin(za_over_l, 9.999f) / za;
        } else if (br > 0.0f) {
            regime = 2.0f;
            psim = fmax(-5.0f * br * gz1 / (1.1f - 5.0f * br), -10.0f);
            psih = psim;
            psim10 = fmax(10.0f / za * psim, -10.0f); psih10 = psim10;
            psim2 = fmax(2.0f / za * psim, -10.0f); psih2 = psim2;
            zol = br * gz1 / (1.00001f - 5.0f * br);
            if (zol > 0.5f) {
                zol = (1.89f * gz1 + 44.2f) * br * br
                    + (1.18f * gz1 - 1.37f) * br;
                zol = fmin(zol, 9.999f);
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
            float zol10 = fmax(fmin(10.0f / za * zol, 0.0f), -9.9999f);
            float zol2 = fmax(fmin(2.0f / za * zol, 0.0f), -9.9999f);
            zol = fmax(fmin(zol, 0.0f), -9.9999f);
            psim = sf_classic_table(zol, false); psih = sf_classic_table(zol, true);
            psim10 = sf_classic_table(zol10, false); psih10 = sf_classic_table(zol10, true);
            psim2 = sf_classic_table(zol2, false); psih2 = sf_classic_table(zol2, true);
            psih = fmin(psih, 0.9f * gz1); psim = fmin(psim, 0.9f * gz1);
            psih2 = fmin(psih2, 0.9f * gz2);
            psim10 = fmin(psim10, 0.9f * gz10);
            psih10 = fmin(psih10, 0.9f * gz10);
            rmol = zol / za;
        }
    } else {
        zol = 0.0f;
        if (br > 0.0f) zol = sf_zolri(fmin(br, 250.0f), za, z0);
        else if (br < 0.0f)
            zol = old_ust < 0.001f ? br * gz1
                  : sf_zolri(fmax(br, -250.0f), za, z0);
        float zz = zol * (za + z0) / za, z10 = zol * (10.0f + z0) / za;
        float z2 = zol * (2.0f + z0) / za, zz0 = zol * z0 / za;
        float scalar_z = land ? zol * 0.01f / za : zz0;
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
            psih = fmin(psih, 0.9f * gz1); psim = fmin(psim, 0.9f * gz1);
            psih2 = fmin(psih2, 0.9f * gz2);
            psim10 = fmin(psim10, 0.9f * gz10);
            psih10 = fmin(psih10, 0.9f * gz10);
        }
        rmol = zol / za;
    }

    float dtg = thx - thgb;
    float psix = gz1 - psim, psix10 = gz10 - psim10;
    float psit = q.option == 91 ? fmax(gz1 - psih, 2.0f) : gz1 - psih;
    float psit2 = gz2 - psih2;
    float zl = land ? 0.01f : z0;
    float psiq = sf_logf(sf_div_rn(karman * old_ust * za, xka) + sf_div_rn(za, zl))
              - (q.option == 91 ? psih : pq);
    float psiq2 = sf_logf(sf_div_rn(karman * old_ust * 2.0f, xka) + sf_div_rn(2.0f, zl))
               - (q.option == 91 ? psih2 : pq2);
    float psiq10 = sf_logf(sf_div_rn(karman * old_ust * 10.0f, xka) + sf_div_rn(10.0f, zl))
                - (q.option == 91 ? psih10 : pq10);

    if (!land) {
        float visc = (1.32f + 0.009f * (temp - 273.15f)) * 1.0e-5f;
        float restar = old_ust * z0 / visc;
        float z0t = fmin(fmax(5.5e-5f * pow(restar, -0.60f), 2.0e-9f),
                         1.0e-4f);
        if (q.option == 91) {
            psiq = fmax(sf_logf((za + z0t) / z0t) - psih, 2.0f);
            psit = fmax(sf_logf((za + z0t) / z0t) - psih, 2.0f);
            psiq2 = fmax(sf_logf((2.0f + z0t) / z0t) - psih2, 2.0f);
            psit2 = fmax(sf_logf((2.0f + z0t) / z0t) - psih2, 2.0f);
            psiq10 = fmax(sf_logf((10.0f + z0t) / z0t) - psih10, 2.0f);
        } else {
            psih = sf_rev_heat_psi(zol, za, z0t, za);
            psih2 = sf_rev_heat_psi(zol, za, z0t, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0t, 10.0f);
            psit = sf_logf((za + z0t) / z0t) - psih;
            psit2 = sf_logf((2.0f + z0t) / z0t) - psih2;
            psiq = psit; psiq2 = psit2;
            psiq10 = sf_logf((10.0f + z0t) / z0t) - psih10;
        }
    }

    if (q.isftcflx == 1 && !land) {
        float z0q = 1.0e-4f;
        if (q.option == 91) {
            psiq = sf_logf(sf_div_rn(za, z0q)) - psih;
            psiq2 = sf_logf(2.0f / z0q) - psih2;
            psiq10 = sf_logf(10.0f / z0q) - psih10;
        } else {
            psih = sf_rev_heat_psi(zol, za, z0q, za);
            psih2 = sf_rev_heat_psi(zol, za, z0q, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0q, 10.0f);
            psiq = sf_logf(sf_div_rn((za + z0q), z0q)) - psih;
            psiq2 = sf_logf((2.0f + z0q) / z0q) - psih2;
            psiq10 = sf_logf((10.0f + z0q) / z0q) - psih10;
        }
        psit = psiq; psit2 = psiq2;
    } else if (q.isftcflx == 2 && !land) {
        float visc = (1.32f + 0.009f * (temp - 273.15f)) * 1.0e-5f;
        float restar = old_ust * z0 / visc;
        float gz0t = 0.4f * (7.3f * pow(restar, 0.25f) * sqrt(0.71f) - 5.0f);
        float gz0q = 0.4f * (7.3f * pow(restar, 0.25f) * sqrt(0.60f) - 5.0f);
        if (q.option == 91) {
            psit = gz1 - psih + gz0t; psiq = gz1 - psih + gz0q;
            psit2 = gz2 - psih2 + gz0t; psiq2 = gz2 - psih2 + gz0q;
            psiq10 = gz10 - psih + gz0q;
        } else {
            float z0t = z0 / exp(gz0t), z0q = z0 / exp(gz0q);
            float pht = sf_rev_heat_psi(zol, za, z0t, za);
            float pht2 = sf_rev_heat_psi(zol, za, z0t, 2.0f);
            psit = sf_logf((za + z0t) / z0t) - pht;
            psit2 = sf_logf((2.0f + z0t) / z0t) - pht2;
            psih = sf_rev_heat_psi(zol, za, z0q, za);
            psih2 = sf_rev_heat_psi(zol, za, z0q, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0q, 10.0f);
            psiq = sf_logf((za + z0q) / z0q) - psih;
            psiq2 = sf_logf((2.0f + z0q) / z0q) - psih2;
            psiq10 = sf_logf((10.0f + z0q) / z0q) - psih10;
        }
    }

    float ck = (karman / psix10) * (karman / psiq10);
    float cd = (karman / psix10) * (karman / psix10);
    float cka = (karman / psix) * (karman / psiq);
    float cda = (karman / psix) * (karman / psix);
    if (q.iz0tlnd >= 1 && land) {
        float visc = (1.32f + 0.009f * (temp - 273.15f)) * 1.0e-5f;
        float restar = old_ust * z0 / visc;
        float czil = q.iz0tlnd == 1 ? pow(10.0f, sf_div_rn(-0.40f * z0, 0.07f)) : 0.1f;
        if (q.option == 91) {
            float add = czil * karman * sqrt(restar);
            psit = psiq = gz1 - psih + add;
            psit2 = psiq2 = gz2 - psih2 + add;
        } else {
            float z0t = z0 / exp(czil * karman * sqrt(restar));
            psih = sf_rev_heat_psi(zol, za, z0t, za);
            psih2 = sf_rev_heat_psi(zol, za, z0t, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0t, 10.0f);
            psit = psiq = sf_logf((za + z0t) / z0t) - psih;
            psit2 = psiq2 = sf_logf((2.0f + z0t) / z0t) - psih2;
        }
    }

    float new_ust = 0.5f * old_ust + 0.5f * karman * wspd / psix;
    // TKE coupling (module_sf_sfclay.F:800-804,
    // physics_mmm/sf_sfclayrev.F90:759-763): USTM repeats the UST
    // relaxation on the wind speed WITHOUT the Beljaars/Mahrt-Sun
    // vconv/vsgd correction and without WSPD's 0.1 floor, and takes
    // none of the land floor UST gets below.
    float wspdi = sqrt(uu * uu + vv * vv);
    float new_ustm = 0.5f * old_ustm + 0.5f * karman * wspdi / psix;
    float u10 = uu * psix10 / psix, v10 = vv * psix10 / psix;
    float th2 = thgb + (thx - thgb) * psit2 / psit;
    float q2 = qs + (qvx - qs) * psiq2 / psiq;
    float t2 = th2 * pow(sf_div_rn(ps, P0), RCP);
    if (land) new_ust = fmax(new_ust, q.option == 91 ? 0.1f : 0.001f);
    float new_mol = karman * (thx - thgb) / psit;

    float z0out = z0;
    if (q.isfflx && !land) {
        z0out = fmin(sf_div_rn(0.0185f * new_ust * new_ust, G)
                      + 0.11f * 1.5e-5f / new_ust, 2.85e-3f);
        if (q.isftcflx != 0) {
            float zw = fmin(pow(sf_div_rn(new_ust, 1.06f), 0.3f), 1.0f);
            float zn1 = sf_div_rn(0.011f * new_ust * new_ust, G) + 1.59e-5f;
            float zn2 = 10.0f * exp(-9.5f * pow(new_ust, -0.3333f))
                       + 0.11f * 1.5e-5f / fmax(new_ust, 0.01f);
            z0out = fmin(fmax((1.0f - zw) * zn1 + zw * zn2, 1.27e-7f),
                           2.85e-3f);
        }
    }

    float flhc = 0.0f, flqc = 0.0f, new_hfx = 0.0f, new_qfx = 0.0f;
    float lh = 0.0f, chs = 0.0f, chs2 = 0.0f, cqs2 = 0.0f;
    if (q.isfflx) {
        flqc = rho * r.mavail[idx] * new_ust * karman / psiq;
        if (fabs(thx - thgb) > 1.0e-5f)
            flhc = cpm * rho * new_ust * new_mol / (thx - thgb);
        new_qfx = flqc * (qs - qvx); lh = XLV * new_qfx;
        new_hfx = flhc * (thgb - thx);
        chs = new_ust * karman / psiq;
        cqs2 = new_ust * karman / psiq2;
        chs2 = new_ust * karman / psit2;
    }

    r.znt[idx] = z0out; r.ust[idx] = new_ust; r.ustm[idx] = new_ustm;
    r.mol[idx] = new_mol;
    r.hfx[idx] = new_hfx; r.qfx[idx] = new_qfx; r.qsfc[idx] = qs;
    r.zol[idx] = zol; r.regime[idx] = regime; r.psim[idx] = psim;
    r.psih[idx] = psih; r.fm[idx] = psix; r.fh[idx] = psit; r.lh[idx] = lh;
    r.u10[idx] = u10; r.v10[idx] = v10; r.th2[idx] = th2; r.t2[idx] = t2;
    r.q2[idx] = q2; r.chs[idx] = chs; r.chs2[idx] = chs2;
    r.cqs2[idx] = cqs2; r.flhc[idx] = flhc; r.flqc[idx] = flqc;
    r.qgh[idx] = qgh; r.rmol[idx] = rmol; r.wspd[idx] = wspd;
    r.br[idx] = br; r.gz1oz0[idx] = gz1; r.cpm[idx] = cpm;
    r.ck[idx] = ck; r.cka[idx] = cka; r.cd[idx] = cd; r.cda[idx] = cda;

}
