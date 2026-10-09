// needs integration: Hex vertical rho-theta flux and divergence.
#include <metal_stdlib>
#include "hex_vertical_theta_tendency_fragment_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float hex_theta_flux3(float qim2, float qim1, float qi, float qip1,
                                    float velocity, float coefficient) {
    const float flux4 = velocity *
        (7.0f * (qi + qim1) - (qip1 + qim2)) / 12.0f;
    return flux4 + coefficient * abs(velocity) *
        ((qip1 - qim2) - 3.0f * (qi - qim1)) / 12.0f;
}

static inline float hex_theta_vertical_flux(
    constant HexVerticalThetaTendencyFragmentResources &r,
    constant HexVerticalThetaTendencyFragmentParams &p,
    uint level, uint cell) {
    const uint nlev = uint(p.nlev), ncells = uint(p.ncells);
    if (level == 0u || level >= nlev) return 0.0f;
    const uint current = level * ncells + cell;
    const uint previous = (level - 1u) * ncells + cell;
    if (level == 1u) {
        return r.rho_w[current] *
            (r.fzm[level] * r.theta[current] + r.fzp[level] * r.theta[previous])
            + (r.rho_w_saved[current] - r.rho_w[current]) *
            (r.fzm[level] * r.theta_saved[current]
             + r.fzp[level] * r.theta_saved[previous]);
    }
    if (level + 1u == nlev) {
        return r.rho_w_saved[current] *
            (r.fzm[level] * r.theta[current] + r.fzp[level] * r.theta[previous]);
    }
    const uint before = (level - 2u) * ncells + cell;
    const uint after = (level + 1u) * ncells + cell;
    return hex_theta_flux3(r.theta[before], r.theta[previous],
        r.theta[current], r.theta[after], r.rho_w[current], p.coefficient)
        + (r.rho_w_saved[current] - r.rho_w[current]) *
        (r.fzm[level] * r.theta_saved[current]
         + r.fzp[level] * r.theta_saved[previous]);
}

kernel void hex_vertical_theta_tendency_fragment(
    constant HexVerticalThetaTendencyFragmentResources &r [[buffer(0)]],
    constant HexVerticalThetaTendencyFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.ncells);
    if (tid >= count) return;
    const uint level = tid / uint(p.ncells);
    const uint cell = tid % uint(p.ncells);
    r.tendency[tid] = r.horizontal[tid] - r.rdzw[level] *
        (hex_theta_vertical_flux(r, p, level + 1u, cell)
         - hex_theta_vertical_flux(r, p, level, cell));
}
