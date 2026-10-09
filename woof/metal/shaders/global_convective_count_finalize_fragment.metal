// Fold the per-column convective pair counts to one host-readable scalar.
#include <metal_stdlib>
#include "global_convective_count_finalize_fragment_bindings.metalh"
using namespace metal;

kernel void global_convective_count_finalize_fragment(
    constant GlobalConvectiveCountFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalConvectiveCountFinalizeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    ulong total = 0ul;
    for (uint i = 0u; i < uint(p.count); ++i)
        total += ulong(r.mixed_pairs[i]);
    r.mixed_total[0] = uint(total);
}
