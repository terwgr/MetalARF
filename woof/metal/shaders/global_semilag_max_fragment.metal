// Chunked FP64 maximum reduction for SL convergence and Lipschitz gates.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_max_fragment_bindings.metalh"

using namespace metal;

kernel void global_semilag_max_fragment(
    constant GlobalSemilagMaxFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagMaxFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.partial_count))return;
    const uint start=tid*uint(p.chunk_size);
    const uint end=min(uint(p.count),start+uint(p.chunk_size));
    ulong maximum=0ul;
    for(uint i=start;i<end;++i)
        if(soft64_greater(r.values[i],maximum))maximum=r.values[i];
    r.partial_max[tid]=maximum;
}
