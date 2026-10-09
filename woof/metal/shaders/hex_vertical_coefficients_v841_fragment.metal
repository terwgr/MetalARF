#include <metal_stdlib>
#include "hex_vertical_coefficients_v841_fragment_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

static inline float hex_vcoef_cofwt(
    constant HexVerticalCoefficientsV841FragmentResources &r,
    constant HexVerticalCoefficientsV841FragmentParams &p,
    uint level, uint cell) {
    const uint index = level * uint(p.ncells) + cell;
    const float rcv = p.rgas / (p.cp - p.rgas);
    return 0.5f * rcv * r.zz[index] * p.gravity * r.rho_base[index]
        / (1.0f + r.qtot[index]) * r.exner[index]
        / ((r.rho_theta_base[index] + r.rho_theta_perturbation[index])
            * r.exner_base[index]);
}

static inline float hex_vcoef_cofwr(
    constant HexVerticalCoefficientsV841FragmentResources &r,
    constant HexVerticalCoefficientsV841FragmentParams &p,
    uint level, uint cell) {
    const uint index = level * uint(p.ncells) + cell;
    const uint below = (level - 1u) * uint(p.ncells) + cell;
    const float interp_zz = r.fzm[level] * r.zz[index]
        + r.fzp[level] * r.zz[below];
    return 0.5f * p.gravity * interp_zz;
}

static inline float hex_vcoef_cofwz(
    constant HexVerticalCoefficientsV841FragmentResources &r,
    constant HexVerticalCoefficientsV841FragmentParams &p,
    uint level, uint cell) {
    const uint index = level * uint(p.ncells) + cell;
    const uint below = (level - 1u) * uint(p.ncells) + cell;
    const float interp_zz = r.fzm[level] * r.zz[index]
        + r.fzp[level] * r.zz[below];
    const float interp_exner = r.fzm[level] * r.exner[index]
        + r.fzp[level] * r.exner[below];
    const float rcv = p.rgas / (p.cp - p.rgas);
    const float c2 = p.cp * rcv;
    return c2 * interp_zz * r.rdzu[level] * r.cqw[index] * interp_exner;
}

static inline float hex_vcoef_coftz(
    constant HexVerticalCoefficientsV841FragmentResources &r,
    constant HexVerticalCoefficientsV841FragmentParams &p,
    uint level, uint cell) {
    if (level == 0u || level >= uint(p.nlev)) return 0.0f;
    const uint index = level * uint(p.ncells) + cell;
    const uint below = (level - 1u) * uint(p.ncells) + cell;
    return r.fzm[level] * r.theta[index] + r.fzp[level] * r.theta[below];
}

kernel void hex_vertical_coefficients_v841_fragment(
    constant HexVerticalCoefficientsV841FragmentResources &r [[buffer(0)]],
    constant HexVerticalCoefficientsV841FragmentParams &p [[buffer(1)]],
    uint cell [[thread_position_in_grid]]) {
    const uint nlev = uint(p.nlev), ncells = uint(p.ncells);
    if (cell >= ncells) return;
    const float rcv = p.rgas / (p.cp - p.rgas);
    const float dts2 = p.dts * p.dts;
    if (cell == 0u) {
        for (uint level = 0u; level < nlev; ++level)
            r.cofrz[level] = r.rdzw[level];
    }
    r.errors[cell] = 0;
    for (uint level = 0u; level <= nlev; ++level) {
        const uint index = level * ncells + cell;
        r.coftz[index] = hex_vcoef_coftz(r, p, level, cell);
        if (level < nlev) {
            const float cofwt = hex_vcoef_cofwt(r, p, level, cell);
            r.cofwt[index] = cofwt;
            if (level == 0u) {
                r.cofwr[index] = 0.0f;
                r.cofwz[index] = 0.0f;
            } else {
                r.cofwr[index] = hex_vcoef_cofwr(r, p, level, cell);
                r.cofwz[index] = hex_vcoef_cofwz(r, p, level, cell);
            }
            r.a_tri[index] = 0.0f;
            r.b_tri[index] = level == 0u ? 1.0f : 0.0f;
            r.c_tri[index] = 0.0f;
            r.alpha_tri[index] = 0.0f;
            r.gamma_tri[index] = 0.0f;
        }
    }

    float gamma_previous = 0.0f;
    for (uint level = 1u; level < nlev; ++level) {
        const uint index = level * ncells + cell;
        const uint below = (level - 1u) * ncells + cell;
        const uint above = (level + 1u) * ncells + cell;
        const float cofwr = hex_vcoef_cofwr(r, p, level, cell);
        const float cofwz = hex_vcoef_cofwz(r, p, level, cell);
        const float coftz = hex_vcoef_coftz(r, p, level, cell);
        const float cofwt = hex_vcoef_cofwt(r, p, level, cell);
        const float cofwt_below = hex_vcoef_cofwt(r, p, level - 1u, cell);
        const float coftz_below = hex_vcoef_coftz(r, p, level - 1u, cell);
        const float coftz_above = hex_vcoef_coftz(r, p, level + 1u, cell);
        const float a_unscaled = -cofwz * coftz_below * r.rdzw[level - 1u]
            * r.zz[below] + cofwr * r.rdzw[level - 1u]
            - cofwt_below * coftz_below * r.rdzw[level - 1u];
        const float a_value = a_unscaled * r.etp[level - 1u]
            * r.ewp[level - 1u];
        float b_value = cofwz * coftz * (
            r.etp[level] * r.rdzw[level] * r.zz[index]
            + r.etp[level - 1u] * r.rdzw[level - 1u] * r.zz[below]);
        b_value = b_value - coftz * (
            r.etp[level] * cofwt * r.rdzw[level]
            - r.etp[level - 1u] * cofwt_below * r.rdzw[level - 1u]);
        b_value = b_value + cofwr * (
            r.etp[level] * r.rdzw[level]
            - r.etp[level - 1u] * r.rdzw[level - 1u]);
        b_value = b_value * r.ewp[level];
        float c_value = -cofwz * coftz_above * r.rdzw[level] * r.zz[index]
            - cofwr * r.rdzw[level]
            + cofwt * coftz_above * r.rdzw[level];
        c_value = c_value * r.etp[level] * r.ewp[level + 1u];
        if (level == nlev - 1u) c_value = 0.0f;
        r.a_tri[index] = a_value;
        r.b_tri[index] = b_value;
        r.c_tri[index] = c_value;
        const float denominator = 1.0f + dts2 * (b_value - a_value * gamma_previous);
        if (denominator == 0.0f) {
            r.errors[cell] = int(level);
            r.alpha_tri[index] = INFINITY;
            r.gamma_tri[index] = INFINITY;
            gamma_previous = INFINITY;
        } else {
            const float alpha = 1.0f / denominator;
            const float gamma = dts2 * c_value * alpha;
            r.alpha_tri[index] = alpha;
            r.gamma_tri[index] = gamma;
            gamma_previous = gamma;
        }
    }
}
