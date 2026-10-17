// Hex v8.4.1 Metal transcription of CUDA kernel regional_lbc_tendency_v841 (woof/hex/cuda_regional_v841.py:160).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_regional_lbc_tendency_v841_bindings.metalh"
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


kernel void hexv841_regional_lbc_tendency_v841(
    constant Hexv841RegionalLbcTendencyV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841RegionalLbcTendencyV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nvalues = prm__.nvalues;
    const float inv_dt = prm__.inv_dt;
    device const float *new_state = rsc__.new_state;
    device const float *old_state = rsc__.old_state;
    device float *tend = rsc__.tend;

    const int index = int(tid);
    if (index >= nvalues) return;
    tend[index] = mpas_mul(
        mpas_sub(new_state[index], old_state[index]), inv_dt);
}
