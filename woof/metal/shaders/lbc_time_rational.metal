#include <metal_stdlib>
#include "evaluate_rational_boundary_bindings.metalh"
#include "soft_binary64.metalh"

using namespace metal;

// Binary64 intermediates by soft arithmetic, in the order of lbc_time.cu.
kernel void evaluate_rational_boundary(
    constant EvaluateRationalBoundaryResources &r [[buffer(0)]],
    constant EvaluateRationalBoundaryParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.count)) return;
    const ulong one = 0x3FF0000000000000ul, two = 0x4000000000000000ul;
    const ulong t = soft64_from_f32_bits(as_type<uint>(p.seconds));
    const ulong pr = soft64_from_f32_bits(as_type<uint>(r.tendency[i]));
    const ulong q = soft64_from_f32_bits(as_type<uint>(r.quadratic[i]));
    const ulong d = soft64_from_f32_bits(as_type<uint>(r.rate[i]));
    const ulong v = soft64_from_f32_bits(as_type<uint>(r.value[i]));
    const ulong denominator = soft64_add(one, soft64_mul(t, d));
    const ulong shift = soft64_div(
        soft64_mul(t, soft64_add(pr, soft64_mul(t, q))), denominator);
    r.out_value[i] = as_type<float>(soft64_to_f32_bits(soft64_add(v, shift)));
    const ulong tq = soft64_mul(soft64_mul(two, t), q);
    const ulong tt = soft64_mul(soft64_mul(soft64_mul(t, t), q), d);
    const ulong numerator = soft64_add(soft64_add(pr, tq), tt);
    r.out_tendency[i] = as_type<float>(soft64_to_f32_bits(
        soft64_div(numerator, soft64_mul(denominator, denominator))));
}
