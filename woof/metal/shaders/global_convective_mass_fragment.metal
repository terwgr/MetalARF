// Convert resident pressure thickness to the convective kernel's dp/g mass.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_convective_mass_fragment_bindings.metalh"
using namespace metal;

kernel void global_convective_mass_fragment(
    constant GlobalConvectiveMassFragmentResources &r [[buffer(0)]],
    constant GlobalConvectiveMassFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    r.mass[tid] = soft64_div(r.dp[tid], r.gravity[0]);
}
