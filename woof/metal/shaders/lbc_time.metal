#include <metal_stdlib>
#include "evaluate_linear_boundary_bindings.metalh"

using namespace metal;

// Match the CUDA source's explicit multiply then add ordering on this default
// FP32 path. Keep contraction disabled so the source operation sequence stays
// visible and deterministic within the selected Metal arithmetic policy.
#pragma clang fp contract(off)

kernel void evaluate_linear_boundary(
    constant EvaluateLinearBoundaryResources &r [[buffer(0)]],
    constant EvaluateLinearBoundaryParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.count)) return;
    r.out_value[i] = r.value[i] + p.seconds * r.tendency[i];
    r.out_tendency[i] = r.tendency[i];
}
