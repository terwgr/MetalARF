// Resident square-root semi-implicit map for Global spectral columns.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_half_apply_fragment_bindings.metalh"

using namespace metal;

struct GlobalComplex64 {
    ulong real;
    ulong imag;
};

static inline GlobalComplex64 global_complex_add(
    GlobalComplex64 a, GlobalComplex64 b) {
    return {soft64_add(a.real, b.real), soft64_add(a.imag, b.imag)};
}

static inline GlobalComplex64 global_complex_sub(
    GlobalComplex64 a, GlobalComplex64 b) {
    return {soft64_sub(a.real, b.real), soft64_sub(a.imag, b.imag)};
}

static inline GlobalComplex64 global_complex_scale(ulong scale, GlobalComplex64 a) {
    return {soft64_mul(scale, a.real), soft64_mul(scale, a.imag)};
}

static inline ulong global_complex_abs(GlobalComplex64 a) {
    return soft64_sqrt(soft64_add(
        soft64_mul(a.real, a.real), soft64_mul(a.imag, a.imag)));
}

kernel void global_semilag_half_apply_fragment(
    constant GlobalSemilagHalfApplyFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagHalfApplyFragmentParams &p [[buffer(1)]],
    uint mode [[thread_position_in_grid]]) {
    const uint nlevel = uint(p.nlevel);
    const uint ndegree = uint(p.ndegree);
    const uint mcount = uint(p.mcount);
    const uint nmodes = ndegree * mcount;
    if (mode >= nmodes) return;
    const uint degree = mode / mcount;
    const uint nx = nlevel + 1u;
    thread GlobalComplex64 x[129];
    thread GlobalComplex64 d[128];
    thread GlobalComplex64 ld[128];
    thread GlobalComplex64 lx[129];
    thread GlobalComplex64 dnew[128];
    thread GlobalComplex64 xnew[129];
    ulong maximum = 0ul;
    for (uint level = 0u; level < nlevel; ++level) {
        const uint index = 2u * (level * nmodes + mode);
        d[level] = {r.divergence[index], r.divergence[index + 1u]};
        x[level] = {r.theta[index], r.theta[index + 1u]};
    }
    const uint surface = 2u * mode;
    x[nlevel] = {r.logps[surface], r.logps[surface + 1u]};

    if (degree == 0u) {
        for (uint level = 0u; level < nlevel; ++level) {
            dnew[level] = d[level];
            xnew[level] = x[level];
        }
        xnew[nlevel] = x[nlevel];
    } else {
        for (uint row = 0u; row < nlevel; ++row) {
            GlobalComplex64 value = {0ul, 0ul};
            for (uint col = 0u; col < nx; ++col) {
                const ulong coefficient = r.b_matrix[row * nx + col];
                value = global_complex_add(
                    value, global_complex_scale(coefficient, x[col]));
            }
            ld[row] = global_complex_scale(r.k2[degree], value);
        }
        for (uint row = 0u; row < nx; ++row) {
            GlobalComplex64 value = {0ul, 0ul};
            for (uint col = 0u; col < nlevel; ++col) {
                const ulong coefficient = r.c_matrix[row * nlevel + col];
                value = global_complex_add(
                    value, global_complex_scale(coefficient, d[col]));
            }
            lx[row] = value;
        }
        for (uint row = 0u; row < nlevel; ++row) {
            GlobalComplex64 value_d = {0ul, 0ul};
            for (uint col = 0u; col < nlevel; ++col) {
                const ulong offset = (degree * nlevel + row) * nlevel + col;
                value_d = global_complex_add(value_d,
                    global_complex_scale(r.p_d[offset], d[col]));
                value_d = global_complex_add(value_d,
                    global_complex_scale(r.q_d[offset], ld[col]));
            }
            dnew[row] = value_d;
        }
        for (uint row = 0u; row < nx; ++row) {
            GlobalComplex64 value_x = {0ul, 0ul};
            for (uint col = 0u; col < nx; ++col) {
                const ulong offset = (degree * nx + row) * nx + col;
                value_x = global_complex_add(value_x,
                    global_complex_scale(r.p_x[offset], x[col]));
                value_x = global_complex_add(value_x,
                    global_complex_scale(r.q_x[offset], lx[col]));
            }
            xnew[row] = value_x;
        }
        for (uint level = 0u; level < nlevel; ++level) {
            const GlobalComplex64 delta = global_complex_sub(dnew[level], d[level]);
            const ulong magnitude = global_complex_abs(delta);
            if (soft64_greater(magnitude, maximum)) maximum = magnitude;
        }
    }
    for (uint level = 0u; level < nlevel; ++level) {
        const uint index = 2u * (level * nmodes + mode);
        r.divergence_new[index] = dnew[level].real;
        r.divergence_new[index + 1u] = dnew[level].imag;
    }
    for (uint level = 0u; level < nx; ++level) {
        const uint index = 2u * (level * nmodes + mode);
        r.x_new[index] = xnew[level].real;
        r.x_new[index + 1u] = xnew[level].imag;
    }
    r.partial_max[mode] = maximum;
}
