#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "wrf_exact_capture_theta_forcing_bindings.metalh"
#include "wrf_exact_map_copy_bindings.metalh"
#include "wrf_exact_map_add_bindings.metalh"
#include "wrf_exact_accumulate_scaled_bindings.metalh"
using namespace metal;

// bandwidth_glue.cu glue_theta_rate: preserve its explicit RN operations.
kernel void wrf_exact_capture_theta_forcing(
    constant WrfExactCaptureThetaForcingResources &r [[buffer(0)]],
    constant WrfExactCaptureThetaForcingParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    uint col = tid % uint(p.ncol), k = tid / uint(p.ncol);
    uint mu = soft32_add(as_type<uint>(r.mub[col]), as_type<uint>(r.mup[col]));
    uint coupling = soft32_add(soft32_mul(as_type<uint>(r.c1[k]), mu), as_type<uint>(r.c2[k]));
    uint rate = soft32_div(as_type<uint>(r.rth[tid]), coupling);
    if (p.mapped) rate = soft32_mul(rate, as_type<uint>(r.msft[col]));
    reinterpret_cast<device uint *>(r.dst)[tid] = rate;
}

// The dry tendency callers use separate FP32 multiply and add operations.
kernel void wrf_exact_map_copy(constant WrfExactMapCopyResources &r [[buffer(0)]],
    constant WrfExactMapCopyParams &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    uint value = soft32_mul(as_type<uint>(r.src[tid]), as_type<uint>(r.map[tid % uint(p.ncol)]));
    reinterpret_cast<device uint *>(r.dst)[tid] = value;
}
kernel void wrf_exact_map_add(constant WrfExactMapAddResources &r [[buffer(0)]],
    constant WrfExactMapAddParams &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    uint value = soft32_mul(as_type<uint>(r.src[tid]), as_type<uint>(r.map[tid % uint(p.ncol)]));
    device uint *dst = reinterpret_cast<device uint *>(r.dst);
    dst[tid] = soft32_add(dst[tid], value);
}
kernel void wrf_exact_accumulate_scaled(
    constant WrfExactAccumulateScaledResources &r [[buffer(0)]],
    constant WrfExactAccumulateScaledParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    uint value = soft32_mul(as_type<uint>(r.src[tid]), as_type<uint>(p.scale));
    device uint *dst = reinterpret_cast<device uint *>(r.dst);
    dst[tid] = soft32_add(dst[tid], value);
}
