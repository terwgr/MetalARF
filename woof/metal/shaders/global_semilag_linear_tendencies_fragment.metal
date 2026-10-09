// Resident linear tendencies for the Global vertical-mode gravity-wave operator.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_linear_tendencies_fragment_bindings.metalh"

using namespace metal;

struct GlobalLinearTendenciesComplex64 {
    ulong real;
    ulong imag;
};

static inline GlobalLinearTendenciesComplex64 global_linear_complex_add(
    GlobalLinearTendenciesComplex64 a,
    GlobalLinearTendenciesComplex64 b) {
    return {soft64_add(a.real, b.real), soft64_add(a.imag, b.imag)};
}

static inline GlobalLinearTendenciesComplex64 global_linear_complex_scale(
    ulong scale, GlobalLinearTendenciesComplex64 value) {
    return {soft64_mul(scale, value.real), soft64_mul(scale, value.imag)};
}

kernel void global_semilag_linear_tendencies_fragment(
    constant GlobalSemilagLinearTendenciesFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagLinearTendenciesFragmentParams &p [[buffer(1)]],
    uint mode [[thread_position_in_grid]]) {
    const uint nlevel = uint(p.nlevel);
    const uint ndegree = uint(p.ndegree);
    const uint mcount = uint(p.mcount);
    const uint nmodes = ndegree * mcount;
    if (mode >= nmodes) return;

    const uint degree = mode / mcount;
    const uint nx = nlevel + 1u;
    const ulong weight = r.weight[0];
    const ulong divergence_scale = soft64_mul(weight, r.k2[degree]);

    // D_t = w k^2 B x, where x stacks theta and log(ps).
    for (uint row = 0u; row < nlevel; ++row) {
        GlobalLinearTendenciesComplex64 value = {0ul, 0ul};
        for (uint col = 0u; col < nx; ++col) {
            GlobalLinearTendenciesComplex64 x;
            if (col < nlevel) {
                const uint index = 2u * (col * nmodes + mode);
                x = {r.theta[index], r.theta[index + 1u]};
            } else {
                const uint index = 2u * mode;
                x = {r.logps[index], r.logps[index + 1u]};
            }
            const ulong coefficient = r.b_matrix[row * nx + col];
            value = global_linear_complex_add(
                value, global_linear_complex_scale(coefficient, x));
        }
        value = global_linear_complex_scale(divergence_scale, value);
        const uint output = 2u * (row * nmodes + mode);
        r.divergence_tendency[output] = value.real;
        r.divergence_tendency[output + 1u] = value.imag;
    }

    // x_t = w C D, including the surface log-pressure row.
    for (uint row = 0u; row < nx; ++row) {
        GlobalLinearTendenciesComplex64 value = {0ul, 0ul};
        for (uint col = 0u; col < nlevel; ++col) {
            const uint input = 2u * (col * nmodes + mode);
            const GlobalLinearTendenciesComplex64 d = {
                r.divergence[input], r.divergence[input + 1u]};
            const ulong coefficient = soft64_mul(
                weight, r.c_matrix[row * nlevel + col]);
            value = global_linear_complex_add(
                value, global_linear_complex_scale(coefficient, d));
        }
        const uint output = 2u * (row * nmodes + mode);
        r.state_tendency[output] = value.real;
        r.state_tendency[output + 1u] = value.imag;
    }
}
