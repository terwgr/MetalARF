// Fold chunked min/max pairs to one twenty-scalar diagnostic vector.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_tracer_extrema_finalize_fragment_bindings.metalh"
using namespace metal;

static inline ulong tracer_min(ulong a, ulong b) {
    return soft64_less(b, a) ? b : a;
}
static inline ulong tracer_max(ulong a, ulong b) {
    return soft64_greater(b, a) ? b : a;
}

kernel void global_tracer_extrema_finalize_fragment(
    constant GlobalTracerExtremaFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalTracerExtremaFinalizeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    const ulong inf = 0x7ff0000000000000ul;
    const ulong ninf = 0xfff0000000000000ul;
    for (uint tracer = 0; tracer < 10u; ++tracer) {
        ulong lo = inf;
        ulong hi = ninf;
        for (uint chunk = 0; chunk < uint(p.partial_count); ++chunk) {
            lo = tracer_min(lo, r.partial_extrema[tracer * uint(p.partial_count) + chunk]);
            hi = tracer_max(hi, r.partial_extrema[(10u + tracer) * uint(p.partial_count) + chunk]);
        }
        r.extrema[tracer] = lo;
        r.extrema[10u + tracer] = hi;
    }
}
