// Finish the resident Global tracer min/max reduction.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_tracer_minmax_finalize_fragment_bindings.metalh"

using namespace metal;

kernel void global_tracer_minmax_finalize_fragment(
    constant GlobalTracerMinmaxFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalTracerMinmaxFinalizeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u || p.partial_count <= 0) return;
    ulong low = r.partial_min[0];
    ulong high = r.partial_max[0];
    for (int index = 1; index < p.partial_count; ++index) {
        const ulong part_low = r.partial_min[index];
        const ulong part_high = r.partial_max[index];
        if (soft64_less(part_low, low)) low = part_low;
        if (soft64_greater(part_high, high)) high = part_high;
    }
    r.minimum[0] = low;
    r.maximum[0] = high;
}
