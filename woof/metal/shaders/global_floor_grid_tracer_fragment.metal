// FP64 pointwise floor for nonnegative Global grid tracers.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_floor_grid_tracer_fragment_bindings.metalh"
using namespace metal;

kernel void global_floor_grid_tracer_fragment(
    constant GlobalFloorGridTracerFragmentResources &r [[buffer(0)]],
    constant GlobalFloorGridTracerFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong value = r.values[tid];
    r.floored[tid] = soft64_less(value, 0ul) ? 0ul : value;
}
