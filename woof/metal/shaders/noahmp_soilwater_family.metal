#include <metal_stdlib>
#include "noahmp_soilwater_family_bindings.metalh"

using namespace metal;

// SSTEP and its fixed four-layer ROSR12 solve use only float32 operations.
// Disallow contraction so each source multiply and add remains a separate op.
#pragma clang fp contract(off)

constant uint NSOIL = 4;

inline float f_max_first(float a, float b) { return b > a ? b : a; }
inline float f_min_first(float a, float b) { return b < a ? b : a; }
inline float f32_1em4() { return as_type<float>(0x38d1b717u); }

inline void rosr12(thread const float *a, thread const float *b,
                   thread float *c, thread const float *d,
                   thread float *p, thread float *delta) {
    c[NSOIL - 1] = 0.0f;
    p[0] = -c[0] / b[0];
    delta[0] = d[0] / b[0];
    for (uint k = 1; k < NSOIL; ++k) {
        float recip = 1.0f / (b[k] + a[k] * p[k - 1]);
        p[k] = -c[k] * recip;
        delta[k] = (d[k] - a[k] * delta[k - 1]) * recip;
    }
    p[NSOIL - 1] = delta[NSOIL - 1];
    for (int k = int(NSOIL) - 2; k >= 0; --k)
        p[k] = p[k] * p[k + 1] + delta[k];
}

kernel void noahmp_soilwater_sstep(
    constant NoahmpSoilwaterSstepResources &r [[buffer(0)]],
    constant NoahmpSoilwaterSstepParams &q [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(q.n)) return;
    ulong pb = ulong(idx) * 24;
    ulong xb = ulong(idx) * 40;
    ulong ob = ulong(idx) * 32;
    float dt = r.fin[xb];
    float dz[NSOIL], sice[NSOIL], sh2o[NSOIL];
    float ai[NSOIL], bi[NSOIL], ci[NSOIL], rhstt[NSOIL];
    for (uint k = 0; k < NSOIL; ++k) {
        dz[k] = r.fin[xb + 1 + k];
        sh2o[k] = r.fin[xb + 5 + k];
        ai[k] = r.fin[xb + 9 + k];
        bi[k] = r.fin[xb + 13 + k];
        ci[k] = r.fin[xb + 17 + k];
        rhstt[k] = r.fin[xb + 21 + k];
        sice[k] = r.fin[xb + 25 + k];
    }

    float wplus = 0.0f;
    for (uint k = 0; k < NSOIL; ++k) {
        rhstt[k] = rhstt[k] * dt;
        ai[k] = ai[k] * dt;
        bi[k] = 1.0f + bi[k] * dt;
        ci[k] = ci[k] * dt;
    }

    float rhsttin[NSOIL], ciin[NSOIL];
    for (uint k = 0; k < NSOIL; ++k) {
        rhsttin[k] = rhstt[k];
        ciin[k] = ci[k];
    }
    rosr12(ai, bi, ciin, rhsttin, ci, rhstt);

    for (uint k = 0; k < NSOIL; ++k) sh2o[k] = sh2o[k] + ci[k];
    for (int k = int(NSOIL) - 1; k >= 1; --k) {
        float epore = f_max_first(f32_1em4(), r.par[pb + k] - sice[k]);
        wplus = f_max_first(sh2o[k] - epore, 0.0f) * dz[k];
        sh2o[k] = f_min_first(epore, sh2o[k]);
        sh2o[k - 1] = sh2o[k - 1] + wplus / dz[k - 1];
    }
    float epore = f_max_first(f32_1em4(), r.par[pb] - sice[0]);
    wplus = f_max_first(sh2o[0] - epore, 0.0f) * dz[0];
    sh2o[0] = f_min_first(epore, sh2o[0]);

    if (wplus > 0.0f) {
        sh2o[1] = sh2o[1] + wplus / dz[1];
        for (uint k = 1; k < NSOIL - 1; ++k) {
            float e = f_max_first(f32_1em4(), r.par[pb + k] - sice[k]);
            wplus = f_max_first(sh2o[k] - e, 0.0f) * dz[k];
            sh2o[k] = f_min_first(e, sh2o[k]);
            sh2o[k + 1] = sh2o[k + 1] + wplus / dz[k + 1];
        }
        float e = f_max_first(f32_1em4(), r.par[pb + NSOIL - 1] - sice[NSOIL - 1]);
        wplus = f_max_first(sh2o[NSOIL - 1] - e, 0.0f) * dz[NSOIL - 1];
        sh2o[NSOIL - 1] = f_min_first(e, sh2o[NSOIL - 1]);
    }

    for (uint k = 0; k < NSOIL; ++k) {
        r.fout[ob + k] = sh2o[k];
        r.fout[ob + 4 + k] = sh2o[k] + sice[k];
        r.fout[ob + 8 + k] = ai[k];
        r.fout[ob + 12 + k] = bi[k];
        r.fout[ob + 16 + k] = ci[k];
        r.fout[ob + 20 + k] = rhstt[k];
    }
    r.fout[ob + 24] = wplus;
}
