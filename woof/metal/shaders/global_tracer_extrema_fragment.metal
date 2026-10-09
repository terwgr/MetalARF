// Chunked min/max reductions for the resident Global grid tracers.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_tracer_extrema_fragment_bindings.metalh"
using namespace metal;

static inline ulong tracer_min(ulong a, ulong b) {
    return soft64_less(b, a) ? b : a;
}
static inline ulong tracer_max(ulong a, ulong b) {
    return soft64_greater(b, a) ? b : a;
}

kernel void global_tracer_extrema_fragment(
    constant GlobalTracerExtremaFragmentResources &r [[buffer(0)]],
    constant GlobalTracerExtremaFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.partial_count)) return;
    const ulong inf = 0x7ff0000000000000ul;
    const ulong ninf = 0xfff0000000000000ul;
    const uint start = tid * uint(p.chunk_size);
    const uint end = min(uint(p.count), start + uint(p.chunk_size));
    ulong lo[10] = {inf, inf, inf, inf, inf, inf, inf, inf, inf, inf};
    ulong hi[10] = {ninf, ninf, ninf, ninf, ninf, ninf, ninf, ninf, ninf, ninf};
    for (uint i = start; i < end; ++i) {
        lo[0] = tracer_min(lo[0], r.qc[i]); hi[0] = tracer_max(hi[0], r.qc[i]);
        lo[1] = tracer_min(lo[1], r.qr[i]); hi[1] = tracer_max(hi[1], r.qr[i]);
        lo[2] = tracer_min(lo[2], r.qi[i]); hi[2] = tracer_max(hi[2], r.qi[i]);
        lo[3] = tracer_min(lo[3], r.qs[i]); hi[3] = tracer_max(hi[3], r.qs[i]);
        lo[4] = tracer_min(lo[4], r.qg[i]); hi[4] = tracer_max(hi[4], r.qg[i]);
        lo[5] = tracer_min(lo[5], r.nc[i]); hi[5] = tracer_max(hi[5], r.nc[i]);
        lo[6] = tracer_min(lo[6], r.nr[i]); hi[6] = tracer_max(hi[6], r.nr[i]);
        lo[7] = tracer_min(lo[7], r.ni[i]); hi[7] = tracer_max(hi[7], r.ni[i]);
        lo[8] = tracer_min(lo[8], r.ns[i]); hi[8] = tracer_max(hi[8], r.ns[i]);
        lo[9] = tracer_min(lo[9], r.ng[i]); hi[9] = tracer_max(hi[9], r.ng[i]);
    }
    for (uint tracer = 0; tracer < 10u; ++tracer) {
        r.partial_extrema[tracer * uint(p.partial_count) + tid] = lo[tracer];
        r.partial_extrema[(10u + tracer) * uint(p.partial_count) + tid] = hi[tracer];
    }
}
