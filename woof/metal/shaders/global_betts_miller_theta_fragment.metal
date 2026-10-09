// Restore potential temperature from the resident Betts-Miller temperature.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_betts_miller_theta_fragment_bindings.metalh"
using namespace metal;

kernel void global_betts_miller_theta_fragment(
    constant GlobalBettsMillerThetaFragmentResources &r [[buffer(0)]],
    constant GlobalBettsMillerThetaFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    r.theta[tid] = soft64_div(r.temperature[tid], r.exner[tid]);
}
