#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_rk_cells_f32_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv842_mpas_rk_cells_f32(
    constant Hexv842MpasRkCellsF32Resources &r [[buffer(0)]],
    constant Hexv842MpasRkCellsF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int i = int(tid), plane = p.nlev * p.ncells;
    if (i >= plane || i % p.ncells >= p.nsolve) return;
    float th = 0.0f;
    if (p.pbl) {
        th = mpas_add(th, mpas_mul(r.rates[i], r.mass[i]));
        for (int s = 0; s < 8; ++s) {
            const int species = r.pbl_indices[s];
            if (species >= 0) {
                const int si = species * plane + i;
                r.out_q[si] = mpas_add(r.out_q[si], mpas_mul(r.rates[(1+s)*plane+i], r.mass[i]));
            }
        }
    }
    if (p.cu) {
        th = mpas_add(th, mpas_mul(r.rates[9*plane+i], r.mass[i]));
        for (int s = 0; s < 5; ++s) {
            const int species = r.cu_indices[s];
            if (species >= 0) {
                const int si = species * plane + i;
                r.out_q[si] = mpas_add(r.out_q[si], mpas_mul(r.rates[(10+s)*plane+i], r.mass[i]));
            }
        }
    }
    if (p.lw) th = mpas_add(th, mpas_mul(r.rates[15*plane+i], r.mass[i]));
    if (p.sw) th = mpas_add(th, mpas_mul(r.rates[16*plane+i], r.mass[i]));
    const float coeff = mpas_add(1.0f, mpas_mul(p.rvord, r.qv[i]));
    const float vapor = r.out_q[r.pbl_indices[0] * plane + i];
    r.out_theta[i] = mpas_add(0.0f, mpas_add(mpas_mul(coeff, th),
        mpas_div(mpas_mul(mpas_mul(p.rvord, r.theta_m[i]), vapor), coeff)));
    r.out_rho[i] = 0.0f;
}
