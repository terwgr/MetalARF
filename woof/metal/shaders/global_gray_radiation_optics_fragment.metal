// Compute gray longwave layer emissivity and source function on Metal.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "global_gray_radiation_optics_fragment_bindings.metalh"
using namespace metal;

static inline ulong global_gray_optics_min(ulong a, ulong b) {
    return soft64_less(a, b) ? a : b;
}

static inline ulong global_gray_optics_max(ulong a, ulong b) {
    return soft64_greater(a, b) ? a : b;
}

kernel void global_gray_radiation_optics_fragment(
    constant GlobalGrayRadiationOpticsFragmentResources &r [[buffer(0)]],
    constant GlobalGrayRadiationOpticsFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const uint column = tid % uint(p.ncolumn);
    ulong column_dp = 0ul;
    for (uint k = 0u; k < uint(p.nlevel); ++k)
        column_dp = soft64_add(column_dp, r.dp[k * uint(p.ncolumn) + column]);
    const ulong mass = global_gray_optics_max(column_dp, 0x3ff0000000000000ul);
    const ulong layer_dp = r.dp[tid];
    const ulong weight = soft64_div(layer_dp, mass);
    r.shortwave_heating[tid] = soft64_mul(r.sw_atmosphere[column], weight);
    ulong cloud = soft64_add(soft64_add(r.qc[tid], r.qi[tid]), r.qs[tid]);
    cloud = global_gray_optics_min(
        global_gray_optics_max(cloud, 0ul), 0x3f847ae147ae147bul);
    const ulong optical_depth = soft64_add(
        soft64_mul(r.tau_column[0], weight),
        soft64_mul(r.cloud_optical_depth[0], cloud));
    const ulong epsilon = soft64_sub(
        0x3ff0000000000000ul, soft64_plm_exp(soft64_negate_bits(optical_depth)));
    const ulong temperature = soft64_mul(r.theta[tid], r.exner[tid]);
    const ulong temperature2 = soft64_mul(temperature, temperature);
    const ulong emission = soft64_mul(
        r.stefan_boltzmann[0], soft64_mul(temperature2, temperature2));
    r.epsilon[tid] = epsilon;
    r.emission[tid] = emission;
}
