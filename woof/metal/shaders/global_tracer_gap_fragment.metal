// Pointwise pseudo-density mismatch and latitude-weighted global diagnostic.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_tracer_gap_fragment_bindings.metalh"

using namespace metal;

kernel void global_tracer_gap_fragment(
    constant GlobalTracerGapFragmentResources &r [[buffer(0)]],
    constant GlobalTracerGapFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;

    const uint latitude = (tid / uint(p.nlon)) % uint(p.nlat);
    const ulong difference = soft64_abs_bits(
        soft64_sub(r.pseudo[tid], r.dp_end[tid]));
    const ulong gap = soft64_div(difference, r.dp_end[tid]);
    r.gap[tid] = gap;
    r.weighted_gap[tid] = soft64_mul(gap, r.lat_weights[latitude]);
}
