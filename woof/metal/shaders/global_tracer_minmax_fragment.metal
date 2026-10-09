// Chunked FP64 extrema for resident Global tracer guards.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_tracer_minmax_fragment_bindings.metalh"

using namespace metal;

kernel void global_tracer_minmax_fragment(
    constant GlobalTracerMinmaxFragmentResources &r [[buffer(0)]],
    constant GlobalTracerMinmaxFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.partial_count)) return;
    const uint start = tid * uint(p.chunk_size);
    const uint end = min(uint(p.count), start + uint(p.chunk_size));
    if (start >= end) return;
    ulong low = r.values[start];
    ulong high = low;
    for (uint index = start + 1u; index < end; ++index) {
        const ulong value = r.values[index];
        if (soft64_less(value, low)) low = value;
        if (soft64_greater(value, high)) high = value;
    }
    r.partial_min[tid] = low;
    r.partial_max[tid] = high;
}
