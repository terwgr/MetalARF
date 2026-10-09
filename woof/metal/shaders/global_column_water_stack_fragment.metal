// Pack Global vapor and total condensate for the level-water-mass reduction.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_column_water_stack_fragment_bindings.metalh"
using namespace metal;

kernel void global_column_water_stack_fragment(
    constant GlobalColumnWaterStackFragmentResources &r [[buffer(0)]],
    constant GlobalColumnWaterStackFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    ulong condensate = soft64_add(r.qc[tid], r.qr[tid]);
    condensate = soft64_add(condensate, r.qi[tid]);
    condensate = soft64_add(condensate, r.qs[tid]);
    condensate = soft64_add(condensate, r.qg[tid]);
    r.stacked[tid] = r.qv[tid];
    r.stacked[uint(p.count) + tid] = condensate;
}
