// Hex v8.4.1 Metal transcription of CUDA kernel regional_lbc_derive_v841 (woof/hex/cuda_regional_v841.py:120).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_regional_lbc_derive_v841_bindings.metalh"
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


kernel void hexv841_regional_lbc_derive_v841(
    constant Hexv841RegionalLbcDeriveV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841RegionalLbcDeriveV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nvalues = prm__.nvalues;
    const float rvord = prm__.rvord;
    device const float *zz = rsc__.zz;
    device const float *lbc_rho = rsc__.lbc_rho;
    device const float *lbc_theta = rsc__.lbc_theta;
    device const float *lbc_qv = rsc__.lbc_qv;
    device float *rho_zz = rsc__.rho_zz;
    device float *rtheta_m = rsc__.rtheta_m;

    const int i = int(tid);
    if (i >= nvalues) return;
    const float r = mpas_div(lbc_rho[i], zz[i]);
    rho_zz[i] = r;
    rtheta_m[i] = mpas_mul(mpas_mul(lbc_theta[i], r),
        mpas_add(1.0f, mpas_mul(rvord, lbc_qv[i])));
}
