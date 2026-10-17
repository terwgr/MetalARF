// Hex v8.4.1 Metal transcription of CUDA kernel regional_moist_cell_coefficients (woof/hex/cuda_regional_forecast_v841.py:1824).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_regional_moist_cell_coefficients_bindings.metalh"
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



kernel void hexv841_regional_moist_cell_coefficients(
    constant Hexv841RegionalMoistCellCoefficientsResources &rsc__ [[buffer(0)]],
    constant Hexv841RegionalMoistCellCoefficientsParams &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm__.nlev;
    const int ncells = prm__.ncells;
    device const float *q = rsc__.q;
    device float *qtot = rsc__.qtot;
    device float *cqw = rsc__.cqw;

    const int element = int(tid);
    if (element >= (nlev * ncells)) return;
    {
        /* CuPy: qtot = zeros_like(q); qtot = qtot + q; cqw = ones; cqw[1:] =
           one / (one + half * (qtot[1:] + qtot[:-1])), each a separate
           float32 operation. */
        const float half_ = 0.5f;
        const float one = 1.0f;
        const int cell = element % ncells;
        const int k = element / ncells;
        const float qt = 0.0f + q[element];
        qtot[element] = qt;
        if (k == 0) {
            cqw[element] = one;
        } else {
            const float qtm = 0.0f + q[RC2(k - 1, cell, ncells)];
            cqw[element] = one / (one + half_ * (qt + qtm));
        }
    }
}
