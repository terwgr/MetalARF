// Hex v8.4.1 Metal transcription of CUDA kernel regional_bdy_adjust_scalars_copyback_v841 (woof/hex/cuda_regional_v841.py:530).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_regional_bdy_adjust_scalars_copyback_v841_bindings.metalh"
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


kernel void hexv841_regional_bdy_adjust_scalars_copyback_v841(
    constant Hexv841RegionalBdyAdjustScalarsCopybackV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841RegionalBdyAdjustScalarsCopybackV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int ntracers = prm__.ntracers;
    const int nlev = prm__.nlev;
    const int ncells = prm__.ncells;
    const int nnudged = prm__.nnudged;
    device const int *nudged_cells = rsc__.nudged_cells;
    device const float *updates = rsc__.updates;
    device float *scalars_new = rsc__.scalars_new;

    const int element = int(tid);
    if (element >= (ntracers * nlev * nnudged)) return;
    {
        const int slot = element % nnudged;
        const int rest = element / nnudged;
        const int k = rest % nlev;
        const int t = rest / nlev;
        const int cell = nudged_cells[slot];
        const int i = RQ3(t, k, cell, nlev, ncells);
        scalars_new[i] = updates[i];
    }
}
