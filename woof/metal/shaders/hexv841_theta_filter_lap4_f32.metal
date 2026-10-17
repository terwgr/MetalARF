#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_theta_filter_lap4_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx(const int k, const int i, const int n) { return k * n + i; }
inline float cell_sign(const int cell, const int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_theta_filter_lap4_f32(
    constant Hexv841ThetaFilterLap4F32Resources &r [[buffer(0)]],
    constant Hexv841ThetaFilterLap4F32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *delsq_theta = r.delsq_theta;
    device const float *del4 = r.del4;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *dc_edge = r.dc_edge;
    device const float *dv_edge = r.dv_edge;
    device const float *area_cell = r.area_cell;
    device float *tendency = r.tendency;
    const float h4 = p.h4;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int max_edges = p.max_edges;

    if (gid >= uint(nlev * ncells)) return;
    {
        const int element = int(gid);
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        const int tid = lidx(k, cell, ncells);
        float tend = tendency[tid];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_div(mpas_mul(mpas_mul(mpas_div(
                mpas_mul(del4[edge], h4), area_cell[cell]), dv_edge[edge]),
                cell_sign(cell, edge, cells_on_edge)), dc_edge[edge]);
            tend = mpas_sub(tend, mpas_mul(factor, mpas_sub(
                delsq_theta[lidx(k, cell1, ncells)],
                delsq_theta[lidx(k, cell0, ncells)])));
        }
        tendency[tid] = tend;
    }
}
