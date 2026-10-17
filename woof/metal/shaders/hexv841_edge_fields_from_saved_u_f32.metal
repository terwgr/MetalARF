#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_edge_fields_from_saved_u_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx(const int k, const int i, const int n) { return k * n + i; }
inline float cell_sign(const int cell, const int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_edge_fields_from_saved_u_f32(
    constant Hexv841EdgeFieldsFromSavedUF32Resources &r [[buffer(0)]],
    constant Hexv841EdgeFieldsFromSavedUF32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *rho = r.rho;
    device const float *normal_velocity = r.normal_velocity;
    device const float *dc_edge = r.dc_edge;
    device const float *dv_edge = r.dv_edge;
    device const int *cells_on_edge = r.cells_on_edge;
    device float *rho_edge = r.rho_edge;
    device float *ke_edge = r.ke_edge;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;

    const int edge = int(gid);
    if (edge >= nedges) return;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, edge, nedges);
        const float edge_rho = mpas_mul(0.5f, mpas_add(
            rho[lidx(k, cell0, ncells)], rho[lidx(k, cell1, ncells)]
        ));
        const float u = normal_velocity[tid];
        rho_edge[tid] = edge_rho;
        ke_edge[tid] = mpas_mul(mpas_mul(mpas_mul(
            dc_edge[edge], dv_edge[edge]), u), u);
    }
}
