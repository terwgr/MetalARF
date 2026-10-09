#include <metal_stdlib>
#include "global_insitu_capture_fragment_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void global_insitu_capture_partial_fragment(
    constant GlobalInsituCapturePartialFragmentResources &r [[buffer(0)]],
    constant GlobalInsituCapturePartialFragmentParams &p [[buffer(1)]],
    uint task [[thread_position_in_grid]]) {
    const uint chunks = uint(p.partial_count);
    const uint total_tasks = 4u * chunks;
    if (task >= total_tasks) return;
    const uint field = task / chunks;
    const uint chunk = task % chunks;
    const uint count = uint(p.nlevel) * uint(p.nlat) * uint(p.nlon);
    const uint start = chunk * uint(p.chunk_size);
    const uint end = min(count, start + uint(p.chunk_size));
    const ulong zero = 0ul;
    ulong sum = zero;
    ulong maximum = zero;
    for (uint i = start; i < end; ++i) {
        const ulong delta = soft64_sub(
            r.current[field * count + i], r.previous[field * count + i]);
        const uint local_row = (i / uint(p.nlon)) % uint(p.nlat);
        const ulong weighted = soft64_mul(
            delta, r.quadrature_weights[uint(p.row_start) + local_row]);
        sum = soft64_add(sum, weighted);
        const ulong magnitude = soft64_abs_bits(delta);
        if (soft64_greater(magnitude, maximum)) maximum = magnitude;
    }
    r.partial_stats[(field * 2u) * chunks + chunk] = sum;
    r.partial_stats[(field * 2u + 1u) * chunks + chunk] = maximum;
}

kernel void global_insitu_capture_finalize_fragment(
    constant GlobalInsituCaptureFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalInsituCaptureFinalizeFragmentParams &p [[buffer(1)]],
    uint field [[thread_position_in_grid]]) {
    if (field >= 4u) return;
    const uint chunks = uint(p.partial_count);
    ulong sum = r.partial_stats[(field * 2u) * chunks];
    ulong maximum = r.partial_stats[(field * 2u + 1u) * chunks];
    for (uint chunk = 1u; chunk < chunks; ++chunk) {
        sum = soft64_add(sum, r.partial_stats[(field * 2u) * chunks + chunk]);
        const ulong candidate = r.partial_stats[(field * 2u + 1u) * chunks + chunk];
        if (soft64_greater(candidate, maximum)) maximum = candidate;
    }
    const ulong denominator = soft64_from_i32(2 * p.nlevel * p.nlon);
    r.statistics[field * 2u] = soft64_div(sum, denominator);
    r.statistics[field * 2u + 1u] = maximum;
}

kernel void global_insitu_capture_merge_fragment(
    constant GlobalInsituCaptureMergeFragmentResources &r [[buffer(0)]],
    constant GlobalInsituCaptureMergeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u || uint(p.nfield) != 4u) return;
    for (uint field = 0u; field < uint(p.nfield); ++field) {
        const uint mean_index = field * 2u;
        const uint max_index = mean_index + 1u;
        r.accumulator[mean_index] = soft64_add(
            r.accumulator[mean_index], r.band_statistics[mean_index]);
        const ulong candidate = r.band_statistics[max_index];
        if (soft64_greater(candidate, r.accumulator[max_index]))
            r.accumulator[max_index] = candidate;
    }
}
