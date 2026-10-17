// Hex v8.4.1 Metal transcription of CUDA kernel regional_reset_speczone_values_v841 (woof/hex/cuda_regional_v841.py:449).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_regional_reset_speczone_values_v841_bindings.metalh"
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


kernel void hexv841_regional_reset_speczone_values_v841(
    constant Hexv841RegionalResetSpeczoneValuesV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841RegionalResetSpeczoneValuesV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm__.nlev;
    const int ncells = prm__.ncells;
    const int nspec = prm__.nspec;
    device const int *spec_cells = rsc__.spec_cells;
    device const float *rt_driving = rsc__.rt_driving;
    device const float *rho_driving = rsc__.rho_driving;
    device const float *rho_theta_base = rsc__.rho_theta_base;
    device float *theta_m = rsc__.theta_m;
    device float *rho_theta = rsc__.rho_theta;
    device float *rho_theta_perturbation = rsc__.rho_theta_perturbation;

    const int element = int(tid);
    if (element >= (nlev * nspec)) return;
    {
        const int slot = element % nspec;
        const int k = element / nspec;
        const int cell = spec_cells[slot];
        const int i = RC2(k, cell, ncells);
        theta_m[i] = mpas_div(rt_driving[i], rho_driving[i]);
        rho_theta[i] = rt_driving[i];
        rho_theta_perturbation[i] = mpas_sub(
            rt_driving[i], rho_theta_base[i]);
    }
}
