// Hex v8.4.1 Metal transcription of CUDA kernel regional_lbc_rho_edge_v841 (woof/hex/cuda_regional_v841.py:135).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_regional_lbc_rho_edge_v841_bindings.metalh"
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


kernel void hexv841_regional_lbc_rho_edge_v841(
    constant Hexv841RegionalLbcRhoEdgeV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841RegionalLbcRhoEdgeV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm__.nlev;
    const int ncells = prm__.ncells;
    const int nedges = prm__.nedges;
    device const int *cells_on_edge = rsc__.cells_on_edge;
    device const float *rho_zz = rsc__.rho_zz;
    device const float *lbc_u = rsc__.lbc_u;
    device float *rho_edge = rsc__.rho_edge;
    device float *ru = rsc__.ru;

    const int element = int(tid);
    if (element >= (nlev * nedges)) return;
    {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        const int present = (c0 >= 0 && c1 >= 0);
        const int i = RE2(k, edge, nedges);
        float value = rho_edge[i];
        if (present) {
            value = mpas_mul(0.5f, mpas_add(
                rho_zz[RC2(k, c0, ncells)], rho_zz[RC2(k, c1, ncells)]));
            rho_edge[i] = value;
        }
        ru[i] = mpas_mul(lbc_u[i], value);
    }
}
