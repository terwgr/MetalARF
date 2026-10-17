// Hex v8.4.1 Metal transcription of CUDA kernel regional_bdy_adjust_scalars_compute_v841 (woof/hex/cuda_regional_v841.py:474).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_regional_bdy_adjust_scalars_compute_v841_bindings.metalh"
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


kernel void hexv841_regional_bdy_adjust_scalars_compute_v841(
    constant Hexv841RegionalBdyAdjustScalarsComputeV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841RegionalBdyAdjustScalarsComputeV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int ntracers = prm__.ntracers;
    const int nlev = prm__.nlev;
    const int ncells = prm__.ncells;
    const int nedges = prm__.nedges;
    const int max_edges = prm__.max_edges;
    const int nrelax = prm__.nrelax;
    const int nspec = prm__.nspec;
    const float ten_dt = prm__.ten_dt;
    const float dt_rk = prm__.dt_rk;
    const float relax_zone = prm__.relax_zone;
    device const int *relax_cells = rsc__.relax_cells;
    device const int *spec_cells = rsc__.spec_cells;
    device const int *bdy_mask_cell = rsc__.bdy_mask_cell;
    device const float *mesh_scaling_cell = rsc__.mesh_scaling_cell;
    device const int *n_edges_on_cell = rsc__.n_edges_on_cell;
    device const int *edges_on_cell = rsc__.edges_on_cell;
    device const int *cells_on_edge = rsc__.cells_on_edge;
    device const float *edge_sign_on_cell = rsc__.edge_sign_on_cell;
    device const float *dv_edge = rsc__.dv_edge;
    device const float *inv_dc_edge = rsc__.inv_dc_edge;
    device const float *scalars_new = rsc__.scalars_new;
    device const float *scalars_driving = rsc__.scalars_driving;
    device float *updates = rsc__.updates;

    const int nowners = nrelax + nspec;
    const int element = int(tid);
    if (element >= (ntracers * nlev * nowners)) return;
    {
        const int slot = element % nowners;
        const int rest = element / nowners;
        const int k = rest % nlev;
        const int t = rest / nlev;
        if (slot < nrelax) {
            const int cell = relax_cells[slot];
            const float mask = (float)bdy_mask_cell[cell];
            const float filter_coef = mpas_div(
                mpas_div(mpas_mul(dt_rk, mpas_sub(mask, 1.0f)),
                         relax_zone),
                mpas_mul(ten_dt, mesh_scaling_cell[cell]));
            const float rayleigh_coef = mpas_div(filter_coef, relax_zone);
            const int count = n_edges_on_cell[cell];
            const int i = RQ3(t, k, cell, nlev, ncells);
            float column = scalars_new[i];
            for (int s = 0; s < count; ++s) {
                const int edge = edges_on_cell[RCES(cell, s, max_edges)];
                float edge_sign = mpas_mul(
                    edge_sign_on_cell[RCES(cell, s, max_edges)],
                    dv_edge[edge]);
                edge_sign = mpas_mul(edge_sign, inv_dc_edge[edge]);
                edge_sign = mpas_mul(edge_sign, filter_coef);
                const int cell1 = cells_on_edge[2 * edge];
                const int cell2 = cells_on_edge[2 * edge + 1];
                const int i1 = RQ3(t, k, cell1, nlev, ncells);
                const int i2 = RQ3(t, k, cell2, nlev, ncells);
                column = mpas_add(column, mpas_mul(edge_sign, mpas_sub(
                    mpas_sub(scalars_new[i2], scalars_driving[i2]),
                    mpas_sub(scalars_new[i1], scalars_driving[i1]))));
            }
            column = mpas_sub(column, mpas_mul(rayleigh_coef,
                mpas_sub(scalars_new[i], scalars_driving[i])));
            updates[i] = column;
        } else {
            const int cell = spec_cells[slot - nrelax];
            const int i = RQ3(t, k, cell, nlev, ncells);
            updates[i] = scalars_driving[i];
        }
    }
}
