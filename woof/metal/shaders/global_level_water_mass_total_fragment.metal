// Finish the compact resident Global water-mass reduction on Metal.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_level_water_mass_total_fragment_bindings.metalh"

using namespace metal;

kernel void global_level_water_mass_total_fragment(
    constant GlobalLevelWaterMassTotalFragmentResources &r [[buffer(0)]],
    constant GlobalLevelWaterMassTotalFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    ulong sum = 0ul;
    for (uint index = 0u; index < uint(p.count); ++index) {
        sum = soft64_add(sum, r.mass[index]);
    }
    r.total[0] = sum;
}
