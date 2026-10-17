// Hex v8.4.1 Metal transcription of CUDA kernel solve_region_validate_v841 (woof/hex/cuda_solve_region_v841.py:248).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_solve_region_validate_v841_bindings.metalh"
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



kernel void hexv841_solve_region_validate_v841(
    constant Hexv841SolveRegionValidateV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841SolveRegionValidateV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint t [[thread_position_in_threadgroup]],
    uint bx [[threadgroup_position_in_grid]],
    uint tgs [[threads_per_threadgroup]]) {
    const int rows = prm__.rows;
    const int stride = prm__.stride;
    const int n = prm__.n;
    const int positive = prm__.positive;
    const int blocks = prm__.blocks;
    device const float *field = rsc__.field;
    device int *flag = rsc__.flag;

    /* One field per launch (CUDA: blockIdx.y). */
    const long rows_l = rows;
    const long stride_l = stride;
    const long n_l = n;
    const long step = (long)blocks * (long)tgs;
    for (long c = (long)bx * tgs + t; c < n_l; c += step) {
        for (long r = 0; r < rows_l; ++r) {
            const float v = field[r * stride_l + c];
            const bool ok = positive ? (isfinite(v) && v > 0.0f) : isfinite(v);
            if (!ok) {
                flag[0] = 1;
                return;
            }
        }
    }
}
