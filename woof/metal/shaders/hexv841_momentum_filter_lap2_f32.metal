#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_momentum_filter_lap2_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx(const int k, const int i, const int n) { return k * n + i; }
inline float cell_sign(const int cell, const int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_momentum_filter_lap2_f32(
    constant Hexv841MomentumFilterLap2F32Resources &r [[buffer(0)]],
    constant Hexv841MomentumFilterLap2F32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *rho_edge = r.rho_edge;
    device const float *divergence = r.divergence;
    device const float *vorticity = r.vorticity;
    device const float *kdiff = r.kdiff;
    device const float *del2 = r.del2;
    device const int *cells_on_edge = r.cells_on_edge;
    device const int *vertices_on_edge = r.vertices_on_edge;
    device const float *dc_edge = r.dc_edge;
    device const float *dv_edge = r.dv_edge;
    device float *delsq_u = r.delsq_u;
    device float *tendency = r.tendency;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int nvertices = p.nvertices;

    const int edge = int(gid);
    if (edge >= nedges) return;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    const int vertex0 = vertices_on_edge[2 * edge];
    const int vertex1 = vertices_on_edge[2 * edge + 1];
    const float inv_dc = 1.0f / dc_edge[edge];
    const float inv_dv = fmin(1.0f / dv_edge[edge], 4.0f * inv_dc);
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, edge, nedges);
        const float lap = mpas_sub(mpas_mul(mpas_sub(
            divergence[lidx(k, cell1, ncells)],
            divergence[lidx(k, cell0, ncells)]), inv_dc),
            mpas_mul(mpas_sub(vorticity[lidx(k, vertex1, nvertices)],
                vorticity[lidx(k, vertex0, nvertices)]), inv_dv));
        const float kedge = mpas_mul(0.5f, mpas_add(
            kdiff[lidx(k, cell0, ncells)],
            kdiff[lidx(k, cell1, ncells)]));
        delsq_u[tid] = lap;
        tendency[tid] = mpas_mul(mpas_mul(mpas_mul(
            rho_edge[tid], kedge), lap), del2[edge]);
    }
}
