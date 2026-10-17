#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_divergence_damping_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx(const int k, const int i, const int n) { return k * n + i; }
inline float cell_sign(const int cell, const int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_divergence_damping_f32(
    constant Hexv841DivergenceDampingF32Resources &r [[buffer(0)]],
    constant Hexv841DivergenceDampingF32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *theta_m = r.theta_m;
    device const float *rtheta_pp = r.rtheta_pp;
    device const float *rtheta_pp_old = r.rtheta_pp_old;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *spec_zone_mask_edge = r.spec_zone_mask_edge;
    device float *rho_u_perturbation = r.rho_u_perturbation;
    const float coefficient = p.coefficient;
    const int n_cells_solve = p.n_cells_solve;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;

    const int edge = int(gid);
    if (edge >= nedges) return;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    if (cell0 >= n_cells_solve && cell1 >= n_cells_solve) return;
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, edge, nedges);
        const float delta0 = mpas_sub(
            rtheta_pp[lidx(k, cell0, ncells)],
            rtheta_pp_old[lidx(k, cell0, ncells)]);
        const float delta1 = mpas_sub(
            rtheta_pp[lidx(k, cell1, ncells)],
            rtheta_pp_old[lidx(k, cell1, ncells)]);
        const float denominator = mpas_add(
            theta_m[lidx(k, cell0, ncells)],
            theta_m[lidx(k, cell1, ncells)]);
        rho_u_perturbation[tid] = mpas_add(rho_u_perturbation[tid],
            mpas_div(mpas_mul(mpas_mul(coefficient,
                mpas_sub(delta0, delta1)), mpas_sub(
                    1.0f, spec_zone_mask_edge[edge])), denominator));
    }
}
