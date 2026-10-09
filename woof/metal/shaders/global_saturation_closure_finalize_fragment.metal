// Finish resident saturation validation and repair reductions to scalars.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_saturation_closure_finalize_fragment_bindings.metalh"

using namespace metal;

kernel void global_saturation_closure_finalize_fragment(
    constant GlobalSaturationClosureFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalSaturationClosureFinalizeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    uint status = 0u;
    ulong maximum = 0ul;
    for (uint index = 0; index < uint(p.count); ++index) {
        status |= r.partial_status[index];
        if (soft64_greater(r.partial_repair_max[index], maximum))
            maximum = r.partial_repair_max[index];
    }
    r.status[0] = status;
    r.maximum_repair[0] = maximum;
}
