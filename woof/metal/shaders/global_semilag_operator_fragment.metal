// Resident FP64 vertical semi-implicit operator contractions for Global SL.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_operator_fragment_bindings.metalh"

using namespace metal;

kernel void global_semilag_operator_fragment(
    constant GlobalSemilagOperatorFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagOperatorFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint nlev = uint(p.nlevel);
    const uint plane = uint(p.ndegree) * uint(p.mcount);
    const uint phi_count = nlev * plane;
    const uint x_count = (nlev + 1u) * plane;
    const ulong weight = r.weight[0];
    if (tid < phi_count) {
        const uint level = tid / plane;
        const uint mode = tid % plane;
        const uint degree = mode / uint(p.mcount);
        const uint order = mode - degree * uint(p.mcount);
        if (order > degree) {
            r.phi_l[2u * tid] = 0ul;
            r.phi_l[2u * tid + 1u] = 0ul;
            return;
        }
        ulong real_sum = 0ul;
        ulong imag_sum = 0ul;
        for (uint source = 0; source < nlev; ++source) {
            const ulong factor = r.b_matrix[level * (nlev + 1u) + source];
            const uint input = source * plane + mode;
            real_sum = soft64_add(real_sum,
                soft64_mul(factor, r.theta[2u * input]));
            imag_sum = soft64_add(imag_sum,
                soft64_mul(factor, r.theta[2u * input + 1u]));
        }
        const ulong surface_factor = r.b_matrix[level * (nlev + 1u) + nlev];
        real_sum = soft64_add(real_sum,
            soft64_mul(surface_factor, r.logps[2u * mode]));
        imag_sum = soft64_add(imag_sum,
            soft64_mul(surface_factor, r.logps[2u * mode + 1u]));
        r.phi_l[2u * tid] = soft64_mul(weight, real_sum);
        r.phi_l[2u * tid + 1u] = order == 0u
            ? 0ul : soft64_mul(weight, imag_sum);
        return;
    }
    const uint output = tid - phi_count;
    if (output >= x_count) return;
    const uint level = output / plane;
    const uint mode = output % plane;
    const uint degree = mode / uint(p.mcount);
    const uint order = mode - degree * uint(p.mcount);
    if (order > degree) {
        r.x_t[2u * output] = 0ul;
        r.x_t[2u * output + 1u] = 0ul;
        return;
    }
    ulong real_sum = 0ul;
    ulong imag_sum = 0ul;
    for (uint source = 0; source < nlev; ++source) {
        const ulong factor = r.c_matrix[level * nlev + source];
        const uint input = source * plane + mode;
        real_sum = soft64_add(real_sum,
            soft64_mul(factor, r.divergence[2u * input]));
        imag_sum = soft64_add(imag_sum,
            soft64_mul(factor, r.divergence[2u * input + 1u]));
    }
    r.x_t[2u * output] = soft64_mul(weight, real_sum);
    r.x_t[2u * output + 1u] = order == 0u
        ? 0ul : soft64_mul(weight, imag_sum);
}
