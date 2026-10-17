#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_momentum_filter_lap4_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx(const int k, const int i, const int n) { return k * n + i; }
inline float cell_sign(const int cell, const int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_momentum_filter_lap4_f32(
    constant Hexv841MomentumFilterLap4F32Resources &r [[buffer(0)]],
    constant Hexv841MomentumFilterLap4F32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *rho_edge = r.rho_edge;
    device const float *delsq_divergence = r.delsq_divergence;
    device const float *delsq_vorticity = r.delsq_vorticity;
    device const float *del4 = r.del4;
    device const int *cells_on_edge = r.cells_on_edge;
    device const int *vertices_on_edge = r.vertices_on_edge;
    device const float *dc_edge = r.dc_edge;
    device const float *dv_edge = r.dv_edge;
    device float *tendency = r.tendency;
    const float h4 = p.h4;
    const float div_factor = p.div_factor;
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
    const float scale = mpas_mul(del4[edge], h4);
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, edge, nedges);
        const float diffusion = mpas_mul(rho_edge[tid], mpas_sub(
            mpas_mul(mpas_mul(mpas_mul(mpas_sub(
                delsq_divergence[lidx(k, cell1, ncells)],
                delsq_divergence[lidx(k, cell0, ncells)]), scale),
                div_factor), inv_dc),
            mpas_mul(mpas_mul(mpas_sub(
                delsq_vorticity[lidx(k, vertex1, nvertices)],
                delsq_vorticity[lidx(k, vertex0, nvertices)]), scale),
                inv_dv)));
        tendency[tid] = mpas_sub(tendency[tid], diffusion);
    }
}
