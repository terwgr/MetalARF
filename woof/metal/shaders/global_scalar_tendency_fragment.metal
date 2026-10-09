// needs integration: FP64 Global scalar tendency closure.
//
// The horizontal transform/divergence and limited vertical flux divergence
// are supplied by preceding Global stages. This kernel closes the mixing
// ratio tendency with the pressure-thickness tendency and divides by the
// positive layer thickness, preserving the source expression's operation
// order. The ABI owner must register this fragment before dispatch.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_scalar_tendency_fragment_bindings.metalh"

using namespace metal;

kernel void global_scalar_tendency_fragment(
    constant GlobalScalarTendencyFragmentResources &r [[buffer(0)]],
    constant GlobalScalarTendencyFragmentParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= q.count) return;
    ulong numerator = soft64_sub(0ul, r.horizontal_divergence[tid]);
    numerator = soft64_sub(numerator, r.vertical_divergence[tid]);
    numerator = soft64_sub(numerator,
        soft64_mul(r.scalar[tid], r.dp_tendency[tid]));
    r.tendency[tid] = soft64_div(numerator, r.layer_thickness[tid]);
}
