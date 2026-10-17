// Hex v8.4.1 Metal transcription of CUDA kernel gf_dry_theta_dynamics_rate_v841_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_gf_dry_theta_dynamics_rate_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)


kernel void hexv841_gf_dry_theta_dynamics_rate_v841_f32(
    constant Hexv841GfDryThetaDynamicsRateV841F32Resources &r [[buffer(0)]],
    constant Hexv841GfDryThetaDynamicsRateV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float rvord = p.rvord;
    device const float *modified_rate = r.modified_rate;
    device const float *theta_m = r.theta_m;
    device const float *qv = r.qv;
    device float *rthdynten = r.rthdynten;
    device float *rqvdynten = r.rqvdynten;
    device int *invalid = r.invalid;

    const int cell = int(tid);
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const float rqv = 0.0f;
        const float fac_m = mpas_div(1.0f,
            mpas_add(1.0f, mpas_mul(rvord, qv[i])));
        const float theta_local = mpas_mul(theta_m[i], fac_m);
        const float value = mpas_mul(fac_m, mpas_sub(modified_rate[i],
            mpas_mul(mpas_mul(theta_local, rvord), rqv)));
        if (!isfinite(qv[i]) || !isfinite(fac_m) || fac_m <= 0.0f
                || !isfinite(value)) atomicExch(invalid, 1);
        rqvdynten[i] = rqv;
        rthdynten[i] = value;
    }
}
