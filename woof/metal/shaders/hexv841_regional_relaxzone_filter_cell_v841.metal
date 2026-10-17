// Hex v8.4.1 Metal transcription of CUDA kernel regional_relaxzone_filter_cell_v841 (woof/hex/cuda_regional_v841.py:274).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_regional_relaxzone_filter_cell_v841_bindings.metalh"
#pragma clang fp contract(off)

#define RC2(k,c,nc) ((k)*(nc) + (c))
#define RE2(k,e,ne) ((k)*(ne) + (e))
#define RQ3(t,k,c,nlev,nc) (((t)*(nlev) + (k))*(nc) + (c))
#define RQE3(t,k,e,nlev,ne) (((t)*(nlev) + (k))*(ne) + (e))
#define RQI3(t,k,c,nlev,nc) (((t)*((nlev)+1) + (k))*(nc) + (c))
#define RCES(c,s,me) ((c)*(me) + (s))
#define RADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
#define REGIONAL_N_RELAX_ZONE 5
#define REGIONAL_N_SPEC_ZONE 2
#define MPAS_FTZ_FALLBACK_ENABLED 1


kernel void hexv841_regional_relaxzone_filter_cell_v841(
    constant Hexv841RegionalRelaxzoneFilterCellV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841RegionalRelaxzoneFilterCellV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm__.nlev;
    const int ncells = prm__.ncells;
    const int nedges = prm__.nedges;
    const int max_edges = prm__.max_edges;
    const int nrelax = prm__.nrelax;
    const float ten_dt = prm__.ten_dt;
    const float relax_zone = prm__.relax_zone;
    device const int *relax_cells = rsc__.relax_cells;
    device const int *bdy_mask_cell = rsc__.bdy_mask_cell;
    device const float *mesh_scaling_cell = rsc__.mesh_scaling_cell;
    device const int *n_edges_on_cell = rsc__.n_edges_on_cell;
    device const int *edges_on_cell = rsc__.edges_on_cell;
    device const int *cells_on_edge = rsc__.cells_on_edge;
    device const float *edge_sign_on_cell = rsc__.edge_sign_on_cell;
    device const float *dv_edge = rsc__.dv_edge;
    device const float *inv_dc_edge = rsc__.inv_dc_edge;
    device const float *rho_zz = rsc__.rho_zz;
    device const float *theta_m = rsc__.theta_m;
    device const float *rho_driving = rsc__.rho_driving;
    device const float *rt_driving = rsc__.rt_driving;
    device float *tend_rho = rsc__.tend_rho;
    device float *tend_rt = rsc__.tend_rt;

    const int element = int(tid);
    if (element >= (nlev * nrelax)) return;
    {
        const int slot = element % nrelax;
        const int k = element / nrelax;
        const int cell = relax_cells[slot];
        const float mask = (float)bdy_mask_cell[cell];
        const float filter_coef = mpas_div(
            mpas_div(mpas_sub(mask, 1.0f), relax_zone),
            mpas_mul(ten_dt, mesh_scaling_cell[cell]));
        const int count = n_edges_on_cell[cell];
        const int i = RC2(k, cell, ncells);
        float rt = tend_rt[i];
        float rho = tend_rho[i];
        for (int s = 0; s < count; ++s) {
            const int edge = edges_on_cell[RCES(cell, s, max_edges)];
            float edge_sign = mpas_mul(
                edge_sign_on_cell[RCES(cell, s, max_edges)], dv_edge[edge]);
            edge_sign = mpas_mul(edge_sign, inv_dc_edge[edge]);
            edge_sign = mpas_mul(edge_sign, filter_coef);
            const int cell1 = cells_on_edge[2 * edge];
            const int cell2 = cells_on_edge[2 * edge + 1];
            /* A relaxation cell (mask 2..5) has every neighbour present; a
               sentinel here would mean the mask derivation disagrees with
               the connectivity, so the lane refuses through the caller's
               admission rather than reading out of bounds. */
            const int i1 = RC2(k, cell1, ncells);
            const int i2 = RC2(k, cell2, ncells);
            rt = mpas_add(rt, mpas_mul(edge_sign, mpas_sub(
                mpas_sub(mpas_mul(rho_zz[i2], theta_m[i2]), rt_driving[i2]),
                mpas_sub(mpas_mul(rho_zz[i1], theta_m[i1]), rt_driving[i1]))));
            rho = mpas_add(rho, mpas_mul(edge_sign, mpas_sub(
                mpas_sub(rho_zz[i2], rho_driving[i2]),
                mpas_sub(rho_zz[i1], rho_driving[i1]))));
        }
        tend_rt[i] = rt;
        tend_rho[i] = rho;
    }
}
