// Finish compact Global FP64 partial sums and write one scalar result.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_sum_finalize_fragment_bindings.metalh"

using namespace metal;

kernel void global_sum_finalize_fragment(
    constant GlobalSumFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalSumFinalizeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    ulong sum = 0ul;
    for (uint index = 0; index < uint(p.count); ++index)
        sum = soft64_add(sum, r.partial_sums[index]);
    r.total[0] = sum;
}
