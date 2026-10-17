// Hex v8.4.1 Metal transcription of CUDA kernel regional_moist_edge_coefficients (woof/hex/cuda_regional_forecast_v841.py:1824).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_regional_moist_edge_coefficients_bindings.metalh"
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



kernel void hexv841_regional_moist_edge_coefficients(
    constant Hexv841RegionalMoistEdgeCoefficientsResources &rsc__ [[buffer(0)]],
    constant Hexv841RegionalMoistEdgeCoefficientsParams &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm__.nlev;
    const int ncells = prm__.ncells;
    const int nedges = prm__.nedges;
    device const float *q = rsc__.q;
    device const int *cells_on_edge = rsc__.cells_on_edge;
    device float *cqu = rsc__.cqu;

    const int element = int(tid);
    if (element >= (nlev * nedges)) return;
    {
        /* CuPy: edge_total = half * (q[:, coe0] + q[:, coe1]);
           cqu = one / (one + edge_total). */
        const float half_ = 0.5f;
        const float one = 1.0f;
        const int edge = element % nedges;
        const int k = element / nedges;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        const float edge_total = half_ * (q[RC2(k, c0, ncells)] + q[RC2(k, c1, ncells)]);
        cqu[element] = one / (one + edge_total);
    }
}
