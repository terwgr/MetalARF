// FP64 surface-water correction and deficit guard for the Global fixer.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_fix_total_water_fragment_bindings.metalh"
using namespace metal;

kernel void global_fix_total_water_fragment(
    constant GlobalFixTotalWaterFragmentResources &r [[buffer(0)]],
    constant GlobalFixTotalWaterFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong corrected = soft64_add(r.water[tid], r.correction[0]);
    r.deficit[tid] = soft64_less(corrected, 0xbe7ad7f29abcaf48ul) ? 1u : 0u;
    r.fixed_water[tid] = soft64_greater(corrected, 0ul) ? corrected : 0ul;
}
