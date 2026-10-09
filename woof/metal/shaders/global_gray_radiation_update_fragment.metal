// Apply the gray-radiation layer heating to potential temperature on Metal.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_gray_radiation_update_fragment_bindings.metalh"
using namespace metal;

kernel void global_gray_radiation_update_fragment(
    constant GlobalGrayRadiationUpdateFragmentResources &r [[buffer(0)]],
    constant GlobalGrayRadiationUpdateFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong dp = r.dp[tid];
    const ulong min_dp = 0x3ff0000000000000ul;
    const ulong mass = soft64_greater(dp, min_dp) ? dp : min_dp;
    const ulong radiative = soft64_add(r.shortwave_heating[tid],
                                       r.longwave_heating[tid]);
    const ulong heating = soft64_div(
        soft64_mul(soft64_mul(r.dt[0], radiative), r.gravity[0]),
        soft64_mul(r.dry_air_cp[0], mass));
    r.theta_out[tid] = soft64_add(
        r.theta[tid], soft64_div(heating, r.exner[tid]));
}
