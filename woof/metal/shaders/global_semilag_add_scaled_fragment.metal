// Add one scalar multiple of a resident FP64 semi-Lagrangian field.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_add_scaled_fragment_bindings.metalh"
using namespace metal;

kernel void global_semilag_add_scaled_fragment(
    constant GlobalSemilagAddScaledFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagAddScaledFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong advanced = soft64_add(
        r.base[tid], soft64_mul(r.current_scale[0], r.increment[tid]));
    r.destination[tid] = soft64_sub(
        advanced, soft64_mul(r.previous_scale[0], r.previous[tid]));
}
