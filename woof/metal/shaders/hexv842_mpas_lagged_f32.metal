#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "soft_gfk_pow.metalh"
#include "hexv842_mpas_add_lagged_f32_bindings.metalh"
#include "hexv842_mpas_zero_lagged_f32_bindings.metalh"
#include "hexv842_mpas_recover_cells_f32_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv842_mpas_add_lagged_f32(constant Hexv842MpasAddLaggedF32Resources &r [[buffer(0)]], constant Hexv842MpasAddLaggedF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.nlev * p.ncells)) return;
    r.tend_rt[i] = mpas_add(r.tend_rt[i], mpas_mul(r.rho[i], r.diabatic_theta[i]));
}

kernel void hexv842_mpas_zero_lagged_f32(constant Hexv842MpasZeroLaggedF32Resources &r [[buffer(0)]], constant Hexv842MpasZeroLaggedF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.nlev * p.nspec)) return;
    const int k = int(i) / p.nspec, slot = int(i) % p.nspec;
    r.diabatic_theta[k * p.ncells + r.spec_cells[slot]] = 0.0f;
}

kernel void hexv842_mpas_recover_cells_f32(constant Hexv842MpasRecoverCellsF32Resources &r [[buffer(0)]], constant Hexv842MpasRecoverCellsF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.nlev * p.ncells)) return;
    const float density_p = mpas_add(r.rho_p_saved[i], r.rho_pp[i]);
    float theta_p = mpas_add(r.rtheta_p_saved[i], r.rtheta_pp[i]);
    const float full_rho = mpas_add(density_p, r.rho_base[i]);
    if (p.stage == 3 && p.has_diabatic)
        theta_p = mpas_sub(theta_p, mpas_mul(mpas_mul(p.dt, full_rho), r.diabatic_theta[i]));
    const float full_rtheta = mpas_add(theta_p, r.rtheta_base[i]);
    r.rho_p[i] = density_p; r.rtheta_p[i] = theta_p;
    r.rho[i] = full_rho; r.rtheta[i] = full_rtheta;
    r.theta[i] = mpas_div(full_rtheta, full_rho);
    if (p.stage == 3) {
        const float argument = mpas_mul(mpas_mul(r.zz[i], mpas_div(p.rgas, p.p0)), full_rtheta);
        const uint arg_bits = as_type<uint>(argument);
        if (arg_bits < 0x00800000u || arg_bits >= 0x7f800000u) {
            atomic_fetch_or_explicit((device atomic_int *)r.invalid, 1, memory_order_relaxed);
            return;
        }
        const float exponent = mpas_div(p.rgas, mpas_sub(p.cp, p.rgas));
        const float value = as_type<float>(soft_gfk_pow_bits(as_type<uint>(argument), as_type<uint>(exponent)));
        r.exner[i] = value;
        r.pressure_p[i] = mpas_mul(mpas_mul(r.zz[i], p.rgas), mpas_add(mpas_mul(value, theta_p), mpas_mul(r.rtheta_base[i], mpas_sub(value, r.exner_base[i]))));
    } else {
        r.exner[i] = r.exner_saved[i]; r.pressure_p[i] = r.pressure_p_saved[i];
    }
}
