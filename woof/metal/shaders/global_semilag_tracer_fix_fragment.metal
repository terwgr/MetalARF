// Column mass readings and conservative multiplicative SL tracer repair.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_tracer_fix_fragment_bindings.metalh"
using namespace metal;

kernel void global_semilag_tracer_mass_fragment(
    constant GlobalSemilagTracerMassFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagTracerMassFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.columns)) return;
    const uint latitude = tid / uint(p.nlon);
    ulong target = 0ul;
    ulong after = 0ul;
    ulong clamp_mass = 0ul;
    for (uint level = 0; level < uint(p.nlevel); ++level) {
        const uint index = level * uint(p.columns) + tid;
        const ulong before_weight = soft64_mul(
            r.dp_before[index], r.cell_weight[latitude]);
        const ulong after_weight = soft64_mul(
            r.dp_after[index], r.cell_weight[latitude]);
        const ulong q_before = soft64_mul(r.before[index], before_weight);
        const ulong q_advected = soft64_mul(r.advected[index], after_weight);
        const ulong q_floor = soft64_greater(r.advected[index], 0ul)
            ? r.advected[index] : 0ul;
        const ulong q_positive = soft64_mul(q_floor, after_weight);
        target = soft64_add(target, q_before);
        after = soft64_add(after, q_positive);
        clamp_mass = soft64_add(clamp_mass, soft64_sub(q_positive, q_advected));
    }
    r.target_partial[tid] = target;
    r.after_partial[tid] = after;
    r.clamp_partial[tid] = clamp_mass;
}

kernel void global_semilag_tracer_scale_fragment(
    constant GlobalSemilagTracerScaleFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagTracerScaleFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong value = r.advected[tid];
    const ulong positive = soft64_greater(value, 0ul) ? value : 0ul;
    r.fixed[tid] = soft64_mul(positive, r.scale[0]);
}
