#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "glue_accumulate_scaled_bindings.metalh"
#include "glue_direct_normal_flux_bindings.metalh"
#include "glue_add_product_rn_bindings.metalh"
#include "glue_add_bindings.metalh"
#include "glue_add_positive_bindings.metalh"
#include "glue_capture_theta_forcing_bindings.metalh"

using namespace metal;

// Match the explicit CUDA FP32 operation boundaries in glue_theta_rate.
#pragma clang fp contract(off)

inline float glue_add_rn(float a, float b) { return a + b; }
inline float glue_mul_rn(float a, float b) { return a * b; }
inline float glue_div_rn(float a, float b) { return a / b; }

kernel void glue_add(constant GlueAddResources &r [[buffer(0)]],
                     constant GlueAddParams &q [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {
    if (ulong(tid) >= q.n) return;
    r.dst[tid] = glue_add_rn(r.dst[tid], r.src[tid]);
}

// dst += max(src, 0): the surface precipitation carriers' accumulation.
kernel void glue_add_positive(constant GlueAddPositiveResources &r [[buffer(0)]],
                              constant GlueAddPositiveParams &q [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {
    if (ulong(tid) >= q.n) return;
    const float value = r.src[tid];
    r.dst[tid] = glue_add_rn(r.dst[tid], value > 0.0f ? value : 0.0f);
}

kernel void glue_capture_theta_forcing(
    constant GlueCaptureThetaForcingResources &r [[buffer(0)]],
    constant GlueCaptureThetaForcingParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (ulong(tid) >= q.n) return;
    const uint col = tid % q.ncol;
    const uint k = tid / q.ncol;
    const float mu = glue_add_rn(r.mub[col], r.mup[col]);
    volatile float scaled = glue_mul_rn(r.c1[k], mu);
    const float coupling = glue_add_rn(scaled, r.c2[k]);
    float rate = glue_div_rn(r.rth[tid], coupling);
    if (q.mapped) rate = glue_mul_rn(rate, r.msft[col]);
    r.dst[tid] = rate;
}

// The shared driver uses separate CuPy multiply/add ufuncs. Each CUDA
// ufunc rounds to binary32 and flushes subnormal operands and results.
static inline uint glue_ufunc_ftz(uint bits) {
    return (bits & 0x7f800000u) == 0u ? bits & 0x80000000u : bits;
}

kernel void glue_accumulate_scaled(
    constant GlueAccumulateScaledResources &r [[buffer(0)]],
    constant GlueAccumulateScaledParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    const uint product = glue_ufunc_ftz(soft32_mul(
        glue_ufunc_ftz(as_type<uint>(r.src[gid])),
        glue_ufunc_ftz(as_type<uint>(p.scale))));
    r.dst[gid] = as_type<float>(glue_ufunc_ftz(soft32_add(
        glue_ufunc_ftz(as_type<uint>(r.dst[gid])), product)));
}

kernel void glue_direct_normal_flux(
    constant GlueDirectNormalFluxResources &r [[buffer(0)]],
    constant GlueDirectNormalFluxParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    const uint cosine = glue_ufunc_ftz(as_type<uint>(r.cosine[gid]));
    const uint magnitude = cosine & 0x7fffffffu;
    const bool nan = magnitude > 0x7f800000u;
    const bool dark = !nan && ((cosine & 0x80000000u) || magnitude == 0u);
    const uint floor = as_type<uint>(1.0e-6f);
    const uint divisor = soft32_less(cosine, floor) ? floor : cosine;
    const uint quotient = glue_ufunc_ftz(soft32_div(
        glue_ufunc_ftz(as_type<uint>(r.direct[gid])), divisor));
    r.normal[gid] = as_type<float>(dark ? 0u : quotient);
}

// Noah-MP RHSUR retains the subnormal exponential between its two REAL(4)
// operations. Integer arithmetic prevents native FTZ or FMA contraction.
kernel void glue_add_product_rn(
    constant GlueAddProductRnResources &r [[buffer(0)]],
    constant GlueAddProductRnParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    const uint product = soft32_mul(as_type<uint>(r.b[gid]), as_type<uint>(r.c[gid]));
    r.out[gid] = as_type<float>(soft32_add(as_type<uint>(r.a[gid]), product));
}
