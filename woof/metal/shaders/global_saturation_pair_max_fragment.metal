// Sum two per-column condensation planes and reduce their maximum in chunks.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_saturation_pair_max_fragment_bindings.metalh"

using namespace metal;

kernel void global_saturation_pair_max_fragment(
    constant GlobalSaturationPairMaxFragmentResources &r [[buffer(0)]],
    constant GlobalSaturationPairMaxFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.partial_count)) return;
    const uint start = tid * uint(p.chunk_size);
    const uint end = min(uint(p.count), start + uint(p.chunk_size));
    ulong maximum = 0ul;
    for (uint index = start; index < end; ++index) {
        const ulong total = soft64_add(r.first[index], r.second[index]);
        if (soft64_greater(total, maximum)) maximum = total;
    }
    r.partial_max[tid] = maximum;
}
