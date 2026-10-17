// Hex v8.4.1 kernel acoustic_rs_ts_lts, transcribed statement for statement from
// woof/hex/cuda_acoustic_lts.py (derived from acoustic_rs_ts_v841 with the index-list gather).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_acoustic_rs_ts_lts_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ZBS(k,c,s,nc,me) ((((k)*(nc) + (c))*(me)) + (s))

kernel void hexv841_acoustic_rs_ts_lts(
    constant Hexv841AcousticRsTsLtsResources &rsc [[buffer(0)]],
    constant Hexv841AcousticRsTsLtsParams &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm.nlev;
    const int ncells = prm.ncells;
    const int nedges = prm.nedges;
    const int max_edges = prm.max_edges;
    const float dts = prm.dts;

    const int lts_slot = int(tid);
    if (lts_slot >= prm.lts_n_active) return;
    const int cell = rsc.lts_active[lts_slot];
    const int count = rsc.n_edges_on_cell[cell];
    for (int k = 0; k < nlev; ++k) {
        const int index = C2(k, cell, ncells);
        float r = 0.0f;
        float t = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = rsc.edges_on_cell[CES(cell, slot, max_edges)];
            const int c0 = rsc.cells_on_edge[2 * edge];
            const int c1 = rsc.cells_on_edge[2 * edge + 1];
            float flux = mpas_mul(
                rsc.edge_sign_on_cell[CES(cell, slot, max_edges)], dts);
            flux = mpas_mul(flux, rsc.dv_edge[edge]);
            flux = mpas_mul(flux, rsc.ru_p[E2(k, edge, nedges)]);
            flux = mpas_mul(flux, rsc.inv_area_cell[cell]);
            r = mpas_sub(r, flux);
            t = mpas_sub(t, mpas_mul(mpas_mul(flux, 0.5f), mpas_add(
                rsc.theta_m[C2(k, c1, ncells)],
                rsc.theta_m[C2(k, c0, ncells)])));
        }
        r = mpas_add(mpas_add(rsc.rho_pp[index], mpas_mul(dts, rsc.tend_rho[index])),
            r);
        r = mpas_sub(r, mpas_mul(mpas_mul(dts, rsc.cofrz[k]), mpas_sub(
            mpas_mul(rsc.ewm[k + 1], rsc.rw_p[C2(k + 1, cell, ncells)]),
            mpas_mul(rsc.ewm[k], rsc.rw_p[index]))));
        t = mpas_add(mpas_add(rsc.rtheta_pp[index], mpas_mul(dts, rsc.tend_rt[index])),
            t);
        t = mpas_sub(t, mpas_mul(mpas_mul(dts, rsc.rdzw[k]), mpas_sub(
            mpas_mul(mpas_mul(rsc.ewm[k + 1], rsc.coftz[C2(k + 1, cell, ncells)]),
                rsc.rw_p[C2(k + 1, cell, ncells)]),
            mpas_mul(mpas_mul(rsc.ewm[k], rsc.coftz[index]), rsc.rw_p[index]))));
        rsc.rs[index] = r;
        rsc.ts[index] = t;
    }
}
