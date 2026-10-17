#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_w_filter_lap2_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx(const int k, const int i, const int n) { return k * n + i; }
inline float cell_sign(const int cell, const int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_w_filter_lap2_f32(
    constant Hexv841WFilterLap2F32Resources &r [[buffer(0)]],
    constant Hexv841WFilterLap2F32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *vertical_velocity = r.vertical_velocity;
    device const float *rho_edge = r.rho_edge;
    device const float *kdiff = r.kdiff;
    device const float *del2 = r.del2;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *dc_edge = r.dc_edge;
    device const float *dv_edge = r.dv_edge;
    device const float *area_cell = r.area_cell;
    device float *delsq_w = r.delsq_w;
    device float *tendency = r.tendency;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges = p.max_edges;

    if (gid >= uint(nlev * ncells)) return;
    {
        const int element = int(gid);
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        const int tid = lidx(k, cell, ncells);
        if (k == 0) {
            delsq_w[tid] = 0.0f;
            tendency[tid] = 0.0f;
            return;
        }
        float lap = 0.0f;
        float tend = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_div(mpas_mul(mpas_mul(
                mpas_div(0.5f, area_cell[cell]),
                cell_sign(cell, edge, cells_on_edge)), dv_edge[edge]),
                dc_edge[edge]);
            const float flux = mpas_mul(mpas_mul(factor, mpas_add(
                rho_edge[lidx(k, edge, nedges)],
                rho_edge[lidx(k - 1, edge, nedges)])), mpas_sub(
                    vertical_velocity[lidx(k, cell1, ncells)],
                    vertical_velocity[lidx(k, cell0, ncells)]));
            lap = mpas_add(lap, flux);
            const float kedge = mpas_mul(0.25f, mpas_add(mpas_add(
                kdiff[lidx(k, cell0, ncells)],
                kdiff[lidx(k, cell1, ncells)]), mpas_add(
                    kdiff[lidx(k - 1, cell0, ncells)],
                    kdiff[lidx(k - 1, cell1, ncells)])));
            tend = mpas_add(tend, mpas_mul(mpas_mul(
                flux, del2[edge]), kedge));
        }
        delsq_w[tid] = lap;
        tendency[tid] = tend;
    }
}
