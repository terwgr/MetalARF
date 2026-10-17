#include <metal_stdlib>
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
