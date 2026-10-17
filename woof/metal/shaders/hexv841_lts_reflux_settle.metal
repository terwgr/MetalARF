// Hex v8.4.1 kernel lts_reflux_settle, transcribed statement for statement from
// woof/hex/cuda_acoustic_lts.py.
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_lts_reflux_settle_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))

kernel void hexv841_lts_reflux_settle(
    constant Hexv841LtsRefluxSettleResources &rsc [[buffer(0)]],
    constant Hexv841LtsRefluxSettleParams &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm.nlev;
    const int nedges = prm.nedges;
    const int ncells = prm.ncells;
    const int n_coarse = prm.n_coarse;
    const int slot = int(tid);
    if (slot >= n_coarse) return;
    const int cell = rsc.coarse_cells[slot];
    const int begin = rsc.slot_offsets[slot];
    const int end = rsc.slot_offsets[slot + 1];
    const float inv_area = rsc.inv_area_cell[cell];
    for (int k = 0; k < nlev; ++k) {
        const int cindex = C2(k, cell, ncells);
        float drho = 0.0f;
        float drtheta = 0.0f;
        for (int s = begin; s < end; ++s) {
            const int edge = rsc.slot_edges[s];
            const int eindex = E2(k, edge, nedges);
            float flux = mpas_mul(rsc.slot_signs[s], rsc.resid[eindex]);
            flux = mpas_mul(flux, inv_area);
            const int c0 = rsc.cells_on_edge[2 * edge];
            const int c1 = rsc.cells_on_edge[2 * edge + 1];
            drho = mpas_add(drho, mpas_sub(0.0f, flux));
            drtheta = mpas_add(drtheta, mpas_sub(0.0f,
                mpas_mul(mpas_mul(flux, 0.5f), mpas_add(
                    rsc.theta_m[C2(k, c1, ncells)],
                    rsc.theta_m[C2(k, c0, ncells)]))));
            rsc.resid[eindex] = 0.0f;
        }
        rsc.rho_pp[cindex] = mpas_add(rsc.rho_pp[cindex], drho);
        rsc.rtheta_pp[cindex] = mpas_add(rsc.rtheta_pp[cindex], drtheta);
    }
}
