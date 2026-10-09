#include <metal_stdlib>
#include "global_reductions_fragment_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;

#pragma clang fp contract(off)

// Each thread reduces one contiguous vertical column. count is the number of
// columns; chunk_size is the number of vertical levels in each column.
kernel void global_reductions_fragment(
    constant GlobalReductionsFragmentResources &r [[buffer(0)]],
    constant GlobalReductionsFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    ulong sum = 0ul;
    const uint start = tid * p.chunk_size;
    for (uint level = 0; level < p.chunk_size; ++level) {
        sum = soft64_add(sum, r.values[start + level]);
    }
    r.partial_sums[tid] = sum;
}
