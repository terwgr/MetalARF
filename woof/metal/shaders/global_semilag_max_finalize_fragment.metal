// Finish the resident partial maximum used by Global SL gates.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_max_finalize_fragment_bindings.metalh"

using namespace metal;

kernel void global_semilag_max_finalize_fragment(
    constant GlobalSemilagMaxFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagMaxFinalizeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    ulong maximum = 0ul;
    for (uint index = 0; index < uint(p.partial_count); ++index) {
        if (soft64_greater(r.partial_max[index], maximum))
            maximum = r.partial_max[index];
    }
    r.maximum[0] = maximum;
}
