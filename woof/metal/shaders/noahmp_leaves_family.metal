#include <metal_stdlib>
#include "noahmp_leaves_family_bindings.metalh"

using namespace metal;

// The CUDA ESAT leaf uses explicit RN add/multiply macros and glibc's
// compiler path is built without contraction. Preserve each Horner stage.
#pragma clang fp contract(off)

inline float nmp_esat_add(float a, float b) { return a + b; }
inline float nmp_esat_mul(float a, float b) { return a * b; }

// ATM's pinned r_pow/r_exp sites use native MSL math for the ordinary Metal
// policy; no glibc-exact parity claim is made for this leaf.
kernel void noahmp_leaf_atm(
    constant NoahmpLeafAtmResources &r [[buffer(0)]],
    constant NoahmpLeafAtmParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(q.ncase)) return;
    const ulong xb = ulong(tid) * 11ul;
    const ulong yb = ulong(tid) * 17ul;
    device const float *x = r.xs + xb;
    device float *y = r.ys + yb;
    const float sfcprs=x[0], sfctmp=x[1], q2=x[2];
    const float prcpconv=x[3], prcpnonc=x[4], prcpshcv=x[5];
    const float soldn=x[9], cosz=x[10];
    const float thair = sfctmp * pow(sfcprs / sfcprs, 287.04f / 1004.64f);
    const float qair = q2;
    const float eair = (qair * sfcprs) / (0.622f + 0.378f * qair);
    const float rhoair = (sfcprs - 0.378f * eair) / (287.04f * sfctmp);
    const float swdown = cosz <= 0.0f ? 0.0f : soldn;
    const float solad1 = (swdown * 0.7f) * 0.5f;
    const float solai1 = (swdown * 0.3f) * 0.5f;
    const float prcp = (prcpconv + prcpnonc) + prcpshcv;
    const float qprecc = 0.10f * prcp;
    const float qprecl = 0.90f * prcp;
    float fp = 0.0f;
    if (qprecc + qprecl > 0.0f)
        fp = (qprecc + qprecl) / (10.0f * qprecc + qprecl);
    float fpice;
    if (sfctmp > 273.16f + 2.5f) fpice = 0.0f;
    else if (sfctmp <= 273.16f + 0.5f) fpice = 1.0f;
    else if (sfctmp <= 273.16f + 2.0f)
        fpice = 1.0f - (-54.632f + 0.2f * sfctmp);
    else fpice = 0.6f;
    const float bdfall = min(120.0f,
        67.92f + 51.25f * exp((sfctmp - 273.16f) / 2.59f));
    y[0]=thair; y[1]=qair; y[2]=eair; y[3]=rhoair;
    y[4]=qprecc; y[5]=qprecl; y[6]=solad1; y[7]=solad1;
    y[8]=solai1; y[9]=solai1; y[10]=swdown; y[11]=bdfall;
    y[12]=prcp * (1.0f - fpice); y[13]=prcp * fpice;
    y[14]=fp; y[15]=fpice; y[16]=prcp;
}

inline float nmp_esat_poly(float t, float k0, float k1, float k2,
                           float k3, float k4, float k5, float k6) {
    float p = nmp_esat_add(k5, nmp_esat_mul(t, k6));
    p = nmp_esat_add(k4, nmp_esat_mul(t, p));
    p = nmp_esat_add(k3, nmp_esat_mul(t, p));
    p = nmp_esat_add(k2, nmp_esat_mul(t, p));
    p = nmp_esat_add(k1, nmp_esat_mul(t, p));
    p = nmp_esat_add(k0, nmp_esat_mul(t, p));
    return nmp_esat_mul(100.0f, p);
}

kernel void noahmp_leaf_esat(
    constant NoahmpLeafEsatResources &r [[buffer(0)]],
    constant NoahmpLeafEsatParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(q.ncase)) return;
    const float t = r.xs[tid];
    const uint out = tid * 4;
    r.ys[out + 0] = nmp_esat_poly(
        t, 6.107799961f, 4.436518521e-01f, 1.428945805e-02f,
        2.650648471e-04f, 3.031240396e-06f, 2.034080948e-08f,
        6.136820929e-11f);
    r.ys[out + 1] = nmp_esat_poly(
        t, 6.109177956f, 5.034698970e-01f, 1.886013408e-02f,
        4.176223716e-04f, 5.824720280e-06f, 4.838803174e-08f,
        1.838826904e-10f);
    r.ys[out + 2] = nmp_esat_poly(
        t, 4.438099984e-01f, 2.857002636e-02f, 7.938054040e-04f,
        1.215215065e-05f, 1.036561403e-07f, 3.532421810e-10f,
        -7.090244804e-13f);
    r.ys[out + 3] = nmp_esat_poly(
        t, 5.030305237e-01f, 3.773255020e-02f, 1.267995369e-03f,
        2.477563108e-05f, 3.005693132e-07f, 2.158542548e-09f,
        7.131097725e-12f);
}

kernel void noahmp_leaf_rosr12(
    constant NoahmpLeafRosr12Resources &r [[buffer(0)]],
    constant NoahmpLeafRosr12Params &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    constexpr int nsoil = 4;
    constexpr int nlay = 7;
    constexpr int off = 2;
    const int ntop = q.ntop0 + 1;
    float a[nlay], b[nlay], c[nlay], d[nlay];
    float p[nlay], del[nlay];
    for (int i = 0; i < nlay; ++i) {
        a[i] = r.xs[i];
        b[i] = r.xs[nlay + i];
        c[i] = r.xs[2 * nlay + i];
        d[i] = r.xs[3 * nlay + i];
        p[i] = 0.0f;
        del[i] = 0.0f;
    }

    c[nsoil + off] = 0.0f;
    const int t = ntop + off;
    p[t] = -c[t] / b[t];
    del[t] = d[t] / b[t];
    for (int k = ntop + 1; k <= nsoil; ++k) {
        const int i = k + off;
        const float recip = 1.0f / (b[i] + a[i] * p[i - 1]);
        p[i] = -c[i] * recip;
        del[i] = (d[i] - a[i] * del[i - 1]) * recip;
    }
    p[nsoil + off] = del[nsoil + off];
    for (int k = ntop + 1; k <= nsoil; ++k) {
        const int kk = nsoil - k + (ntop - 1) + 1;
        const int i = kk + off;
        p[i] = p[i] * p[i + 1] + del[i];
    }

    for (int i = 0; i < nlay; ++i) {
        r.ys[i] = p[i];
        r.ys[nlay + i] = del[i];
        r.ys[2 * nlay + i] = c[i];
    }
}
