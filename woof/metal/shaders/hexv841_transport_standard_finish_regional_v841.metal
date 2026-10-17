// Hex v8.4.1 Metal transcription of CUDA kernel transport_standard_finish_regional_v841 (woof/hex/cuda_regional_v841.py:919).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_transport_standard_finish_regional_v841_bindings.metalh"
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


kernel void hexv841_transport_standard_finish_regional_v841(
    constant Hexv841TransportStandardFinishRegionalV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841TransportStandardFinishRegionalV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int ntracers = prm__.ntracers;
    const int nlev = prm__.nlev;
    const int ncells = prm__.ncells;
    const int nedges = prm__.nedges;
    const int max_edges = prm__.max_edges;
    const float dt = prm__.dt;
    device const int *bdy_mask_cell = rsc__.bdy_mask_cell;
    device const int *n_edges_on_cell = rsc__.n_edges_on_cell;
    device const int *edges_on_cell = rsc__.edges_on_cell;
    device const float *acoustic_sign = rsc__.acoustic_sign;
    device const float *inv_area_cell = rsc__.inv_area_cell;
    device const float *velocity = rsc__.velocity;
    device const float *edge_values = rsc__.edge_values;
    device const float *vertical_flux = rsc__.vertical_flux;
    device const float *rdzw = rsc__.rdzw;
    device const float *old = rsc__.old;
    device const float *rho_old = rsc__.rho_old;
    device const float *target_density = rsc__.target_density;
    device const float *source = rsc__.source;
    device const float *stage = rsc__.stage;
    device float *output = rsc__.output;

    const int element = int(tid);
    if (element >= (ntracers * nlev * ncells)) return;
    {
        const int cell = element % ncells;
        const int rest = element / ncells;
        const int k = rest % nlev;
        const int tracer = rest / nlev;
        const int keep = bdy_mask_cell[cell] > REGIONAL_N_RELAX_ZONE;
        const int index = RQ3(tracer, k, cell, nlev, ncells);
        if (keep) {
            output[index] = stage[index];
            return;
        }
        float horizontal = 0.0f;
        for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
            const int edge = edges_on_cell[RCES(cell, slot, max_edges)];
            const float sign = -acoustic_sign[RCES(cell, slot, max_edges)];
            horizontal = mpas_add(horizontal, mpas_mul(mpas_mul(
                sign, velocity[RE2(k, edge, nedges)]),
                edge_values[RQE3(tracer, k, edge, nlev, nedges)]));
        }
        const float tendency = mpas_add(
            mpas_mul(horizontal, inv_area_cell[cell]), source[index]);
        const float vertical = mpas_mul(rdzw[k], mpas_sub(
            vertical_flux[RQI3(tracer, k + 1, cell, nlev, ncells)],
            vertical_flux[RQI3(tracer, k, cell, nlev, ncells)]));
        const float numerator = mpas_add(
            mpas_mul(old[index], rho_old[RC2(k, cell, ncells)]),
            mpas_mul(dt, mpas_sub(tendency, vertical)));
        const float rho_new_inv = mpas_div(
            1.0f, target_density[RC2(k, cell, ncells)]);
        output[index] = mpas_mul(numerator, rho_new_inv);
    }
}
