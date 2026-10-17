#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_theta_filter_lap2_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx(const int k, const int i, const int n) { return k * n + i; }
inline float cell_sign(const int cell, const int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_theta_filter_lap2_f32(
    constant Hexv841ThetaFilterLap2F32Resources &r [[buffer(0)]],
    constant Hexv841ThetaFilterLap2F32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *theta_m = r.theta_m;
    device const float *rho_edge = r.rho_edge;
    device const float *kdiff = r.kdiff;
    device const float *del2 = r.del2;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *dc_edge = r.dc_edge;
    device const float *dv_edge = r.dv_edge;
    device const float *area_cell = r.area_cell;
    device float *delsq_theta = r.delsq_theta;
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
        float lap = 0.0f;
        float tend = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            const int cell0 = cells_on_edge[2 * edge];
            const int cell1 = cells_on_edge[2 * edge + 1];
            const float factor = mpas_div(mpas_mul(
                cell_sign(cell, edge, cells_on_edge), dv_edge[edge]),
                mpas_mul(area_cell[cell], dc_edge[edge]));
            const float flux = mpas_mul(mpas_mul(factor, mpas_sub(
                theta_m[lidx(k, cell1, ncells)],
                theta_m[lidx(k, cell0, ncells)])),
                rho_edge[lidx(k, edge, nedges)]);
            lap = mpas_add(lap, flux);
            tend = mpas_add(tend, mpas_mul(mpas_mul(mpas_mul(
                flux, 0.5f), mpas_add(
                    kdiff[lidx(k, cell0, ncells)],
                    kdiff[lidx(k, cell1, ncells)])), del2[edge]));
        }
        const int tid = lidx(k, cell, ncells);
        delsq_theta[tid] = lap;
        tendency[tid] = tend;
    }
}
