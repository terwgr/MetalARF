#include "hexv842_nt_prefix_bindings.metalh"
#include <metal_stdlib>
#include "soft_binary32.metalh"
using namespace metal;

// MPAS forcing and the arithmetic portion of native Tiedtke preparation.
// Field-major arrays: level inputs are t, pcps, dz, rho, qv, exner,
// rthdyn, rthbl, lw, sw, rqvdyn, rqvbl; interfaces are p8w, w.
// Outputs: thften, qvften, prsl, tf, ghtl, omg, zqp1, pgeo, ptte,
// pqte; interfaces prsi, ghti, pgeoh. No saturation or dx calculations.
static inline float add32(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float mul32(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}
static inline float div32(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}

kernel void hexv842_nt_prefix(constant Hexv842NtPrefixResources &r [[buffer(0)]], constant Hexv842NtPrefixParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    device const float *levels=r.levels;
    device const float *interfaces=r.interfaces;
    device float *out=r.out;
    device float *outif=r.outif;

    if (i >= uint(p.ncol)) return;
    int s = p.nz * p.ncol, si = (p.nz+1) * p.ncol;
    float zi = 0.0f;
    outif[si + p.nz*p.ncol+i] = 0.0f;
    for (int k=0; k<p.nz; ++k) {
        int j=k*p.ncol+i, o=(p.nz-1-k)*p.ncol+i;
        float th = add32(levels[6*s+j], levels[7*s+j]);
        th = add32(th, levels[8*s+j]);
        th = add32(th, levels[9*s+j]);
        th = mul32(th, levels[5*s+j]);
        float qf = add32(levels[10*s+j], levels[11*s+j]);
        out[j] = th; out[s+j] = qf;
        float next = add32(zi, levels[2*s+j]);
        float zl = mul32(0.5f, add32(zi, next));
        float ws = add32(interfaces[si+j], interfaces[si+j+p.ncol]);
        float omega = mul32(mul32(mul32(0.5f, p.grav), levels[3*s+j]), ws);
        float dot = as_type<float>(as_type<uint>(omega) ^ 0x80000000u);
        out[2*s+o] = levels[s+j]; out[3*s+o] = levels[j];
        out[4*s+o] = zl; out[5*s+o] = dot;
        float q = levels[4*s+j]; out[6*s+o] = div32(q, add32(1.0f, q));
        out[7*s+o] = mul32(p.grav, zl);
        out[8*s+o] = p.initflag == 1 ? 0.0f : th;
        out[9*s+o] = p.initflag == 1 ? 0.0f : qf;
        outif[si+o] = next;
        zi = next;
    }
    for (int k=0; k<=p.nz; ++k) {
        int j=k*p.ncol+i, o=(p.nz-k)*p.ncol+i;
        outif[o] = interfaces[j];
        outif[2*si+j] = mul32(p.grav, outif[si+j]);
    }
}
