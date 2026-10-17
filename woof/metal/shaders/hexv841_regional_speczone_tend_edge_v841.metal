// Hex v8.4.1 Metal transcription of CUDA kernel regional_speczone_tend_edge_v841 (woof/hex/cuda_regional_v841.py:206).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_regional_speczone_tend_edge_v841_bindings.metalh"
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


kernel void hexv841_regional_speczone_tend_edge_v841(
    constant Hexv841RegionalSpeczoneTendEdgeV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841RegionalSpeczoneTendEdgeV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm__.nlev;
    const int nedges = prm__.nedges;
    const int nspec = prm__.nspec;
    device const int *spec_edges = rsc__.spec_edges;
    device const float *ru_driving_tend = rsc__.ru_driving_tend;
    device float *tend_ru = rsc__.tend_ru;

    const int element = int(tid);
    if (element >= (nlev * nspec)) return;
    {
        const int slot = element % nspec;
        const int k = element / nspec;
        const int edge = spec_edges[slot];
        const int i = RE2(k, edge, nedges);
        tend_ru[i] = ru_driving_tend[i];
    }
}
