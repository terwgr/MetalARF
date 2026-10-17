// Hex v8.4.1 Metal transcription of CUDA kernel modified_theta_dynamics_rate_v841_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_modified_theta_dynamics_rate_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)


kernel void hexv841_modified_theta_dynamics_rate_v841_f32(
    constant Hexv841ModifiedThetaDynamicsRateV841F32Resources &r [[buffer(0)]],
    constant Hexv841ModifiedThetaDynamicsRateV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    device const float *tend_rtheta = r.tend_rtheta;
    device const float *tend_rho = r.tend_rho;
    device const float *rho = r.rho;
    device const float *theta_m = r.theta_m;
    device float *result = r.result;
    device int *invalid = r.invalid;

    const int cell = int(tid);
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const float value = mpas_div(mpas_sub(tend_rtheta[i],
            mpas_mul(tend_rho[i], theta_m[i])), rho[i]);
        if (!isfinite(rho[i]) || rho[i] <= 0.0f || !isfinite(value))
            atomicExch(invalid, 1);
        result[i] = value;
    }
}
