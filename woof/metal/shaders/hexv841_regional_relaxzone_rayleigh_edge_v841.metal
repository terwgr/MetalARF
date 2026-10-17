// Hex v8.4.1 Metal transcription of CUDA kernel regional_relaxzone_rayleigh_edge_v841 (woof/hex/cuda_regional_v841.py:250).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_regional_relaxzone_rayleigh_edge_v841_bindings.metalh"
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


kernel void hexv841_regional_relaxzone_rayleigh_edge_v841(
    constant Hexv841RegionalRelaxzoneRayleighEdgeV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841RegionalRelaxzoneRayleighEdgeV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm__.nlev;
    const int nedges = prm__.nedges;
    const int nrelax = prm__.nrelax;
    const float fifty_dt = prm__.fifty_dt;
    const float relax_zone = prm__.relax_zone;
    device const int *relax_edges = rsc__.relax_edges;
    device const int *bdy_mask_edge = rsc__.bdy_mask_edge;
    device const float *mesh_scaling_edge = rsc__.mesh_scaling_edge;
    device const float *ru = rsc__.ru;
    device const float *ru_driving = rsc__.ru_driving;
    device float *tend_ru = rsc__.tend_ru;

    const int element = int(tid);
    if (element >= (nlev * nrelax)) return;
    {
        const int slot = element % nrelax;
        const int k = element / nrelax;
        const int edge = relax_edges[slot];
        const float mask = (float)bdy_mask_edge[edge];
        const float coef = mpas_div(
            mpas_div(mpas_sub(mask, 1.0f), relax_zone),
            mpas_mul(fifty_dt, mesh_scaling_edge[edge]));
        const int i = RE2(k, edge, nedges);
        tend_ru[i] = mpas_sub(tend_ru[i], mpas_mul(
            coef, mpas_sub(ru[i], ru_driving[i])));
    }
}
