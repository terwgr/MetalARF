// Hex v8.4.1 Metal transcription of CUDA kernel solve_region_scrub_columns_v841 (woof/hex/cuda_solve_region_v841.py:274).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_solve_region_scrub_columns_v841_bindings.metalh"
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



kernel void hexv841_solve_region_scrub_columns_v841(
    constant Hexv841SolveRegionScrubColumnsV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841SolveRegionScrubColumnsV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int rows = prm__.rows;
    const int stride = prm__.stride;
    const int solve = prm__.solve;
    const int pool_bits = prm__.pool_bits;
    device float *field = rsc__.field;

    /* One array per launch (CUDA: blockIdx.y). */
    const long r = (long)tid;
    if (r >= (long)rows) return;
    field[r * (long)stride + (long)solve] = as_type<float>(pool_bits);
}
