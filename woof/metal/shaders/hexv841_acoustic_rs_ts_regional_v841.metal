// Hex v8.4.1 Metal transcription of CUDA kernel acoustic_rs_ts_regional_v841 (woof/hex/cuda_regional_v841.py:641).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_acoustic_rs_ts_regional_v841_bindings.metalh"
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


kernel void hexv841_acoustic_rs_ts_regional_v841(
    constant Hexv841AcousticRsTsRegionalV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841AcousticRsTsRegionalV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm__.nlev;
    const int ncells = prm__.ncells;
    const int nedges = prm__.nedges;
    const int max_edges = prm__.max_edges;
    const float dts = prm__.dts;
    device const int *n_edges_on_cell = rsc__.n_edges_on_cell;
    device const int *edges_on_cell = rsc__.edges_on_cell;
    device const int *cells_on_edge = rsc__.cells_on_edge;
    device const float *edge_sign_on_cell = rsc__.edge_sign_on_cell;
    device const float *dv_edge = rsc__.dv_edge;
    device const float *inv_area_cell = rsc__.inv_area_cell;
    device const float *spec_zone_mask_cell = rsc__.spec_zone_mask_cell;
    device const float *theta_m = rsc__.theta_m;
    device const float *rdzw = rsc__.rdzw;
    device const float *cofrz = rsc__.cofrz;
    device const float *coftz = rsc__.coftz;
    device const float *ewm = rsc__.ewm;
    device const float *ru_p = rsc__.ru_p;
    device const float *rw_p = rsc__.rw_p;
    device const float *rho_pp = rsc__.rho_pp;
    device const float *rtheta_pp = rsc__.rtheta_pp;
    device const float *tend_rho = rsc__.tend_rho;
    device const float *tend_rt = rsc__.tend_rt;
    device float *rs = rsc__.rs;
    device float *ts = rsc__.ts;

    const int element = int(tid);
    if (element >= (nlev * ncells)) return;
    {
        const int cell = element % ncells;
        const int k = element / ncells;
        const int index = RC2(k, cell, ncells);
        if (spec_zone_mask_cell[cell] != 0.0f) {
            rs[index] = 0.0f;
            ts[index] = 0.0f;
            return;
        }
        const int count = n_edges_on_cell[cell];
        float r = 0.0f;
        float t = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[RCES(cell, slot, max_edges)];
            const int c0 = cells_on_edge[2 * edge];
            const int c1 = cells_on_edge[2 * edge + 1];
            float flux = mpas_mul(
                edge_sign_on_cell[RCES(cell, slot, max_edges)], dts);
            flux = mpas_mul(flux, dv_edge[edge]);
            flux = mpas_mul(flux, ru_p[RE2(k, edge, nedges)]);
            flux = mpas_mul(flux, inv_area_cell[cell]);
            r = mpas_sub(r, flux);
            t = mpas_sub(t, mpas_mul(mpas_mul(flux, 0.5f), mpas_add(
                theta_m[RC2(k, c1, ncells)],
                theta_m[RC2(k, c0, ncells)])));
        }
        r = mpas_add(mpas_add(rho_pp[index], mpas_mul(dts, tend_rho[index])),
            r);
        r = mpas_sub(r, mpas_mul(mpas_mul(dts, cofrz[k]), mpas_sub(
            mpas_mul(ewm[k + 1], rw_p[RC2(k + 1, cell, ncells)]),
            mpas_mul(ewm[k], rw_p[index]))));
        t = mpas_add(mpas_add(rtheta_pp[index], mpas_mul(dts, tend_rt[index])),
            t);
        t = mpas_sub(t, mpas_mul(mpas_mul(dts, rdzw[k]), mpas_sub(
            mpas_mul(mpas_mul(ewm[k + 1], coftz[RC2(k + 1, cell, ncells)]),
                rw_p[RC2(k + 1, cell, ncells)]),
            mpas_mul(mpas_mul(ewm[k], coftz[index]), rw_p[index]))));
        rs[index] = r;
        ts[index] = t;
    }
}
