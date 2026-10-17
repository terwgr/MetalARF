#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "physics_params_family_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void scale_physics_param_values(
    constant ScalePhysicsParamValuesResources &r [[buffer(0)]],
    constant ScalePhysicsParamValuesParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= uint(p.count)) return;
    const ulong value = r.values[index], factor = r.factors[index];
    const float result = as_type<float>(soft64_to_f32_bits(soft64_mul(value, factor)));
    r.outputs[index] = result;
    int code = 0;
    if (soft64_is_nan(value) || soft64_is_infinite(value) ||
        soft64_is_nan(factor) || soft64_is_infinite(factor) || !isfinite(result)) code = 1;
    else if (result <= 0.0f) code = 2;
    else if (r.crop_flags[index] && result <= 0.125f) code = 3;
    r.status[index] = code;
}
