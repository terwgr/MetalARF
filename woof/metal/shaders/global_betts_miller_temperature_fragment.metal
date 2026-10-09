// Convert resident potential temperature to the temperature field consumed
// by the Global Betts-Miller column kernel.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_betts_miller_temperature_fragment_bindings.metalh"
using namespace metal;

kernel void global_betts_miller_temperature_fragment(
    constant GlobalBettsMillerTemperatureFragmentResources &r [[buffer(0)]],
    constant GlobalBettsMillerTemperatureFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    r.temperature[tid] = soft64_mul(r.theta[tid], r.exner[tid]);
}
