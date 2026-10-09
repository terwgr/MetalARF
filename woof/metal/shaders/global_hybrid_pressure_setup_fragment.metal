// FP64 hybrid pressure setup for the Global forecast.
#include <metal_stdlib>
#include "global_hybrid_pressure_setup_fragment_bindings.metalh"
#include "soft_binary64.metalh"

using namespace metal;
#pragma clang fp contract(off)

kernel void global_hybrid_pressure_setup_fragment(
    constant GlobalHybridPressureSetupFragmentResources &r [[buffer(0)]],
    constant GlobalHybridPressureSetupFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.ncolumn) * uint(p.nlevel + 1);
    if (tid >= count) return;
    const uint level = tid / uint(p.ncolumn);
    const uint column = tid - level * uint(p.ncolumn);
    const ulong upper = soft64_add(
        r.a_half[level], soft64_mul(r.b_half[level], r.surface_pressure[column]));
    r.pressure_half[tid] = upper;
    if (level < uint(p.nlevel)) {
        const ulong lower = soft64_add(
            r.a_half[level + 1],
            soft64_mul(r.b_half[level + 1], r.surface_pressure[column]));
        const uint layer = level * uint(p.ncolumn) + column;
        r.layer_thickness[layer] = soft64_sub(lower, upper);
        r.pressure_full[layer] = soft64_sqrt(soft64_mul(upper, lower));
    }
}
