// Reduce per-column deep-convection flags to the scalar diagnostic count.
#include <metal_stdlib>
#include "global_betts_miller_deep_finalize_fragment_bindings.metalh"
using namespace metal;

kernel void global_betts_miller_deep_finalize_fragment(
    constant GlobalBettsMillerDeepFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalBettsMillerDeepFinalizeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    ulong total = 0ul;
    for (uint i = 0u; i < uint(p.count); ++i)
        total += ulong(r.deep[i]);
    r.deep_count[0] = uint(total);
}
