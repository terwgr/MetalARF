// Hex v8.4.1 Metal transcription of CUDA kernel moist_dpdz_v841_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_moist_dpdz_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)


kernel void hexv841_moist_dpdz_v841_f32(
    constant Hexv841MoistDpdzV841F32Resources &r [[buffer(0)]],
    constant Hexv841MoistDpdzV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float gravity = p.gravity;
    device const float *rho_base = r.rho_base;
    device const float *density_perturbation = r.density_perturbation;
    device const float *qtot = r.qtot;
    device float *dpdz = r.dpdz;
    device int *invalid = r.invalid;

    const int cell = int(tid);
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const float moist_base = mpas_mul(rho_base[i], qtot[i]);
        const float moist_perturbation = mpas_mul(density_perturbation[i],
            mpas_add(1.0f, qtot[i]));
        const float value = mpas_mul(mpas_sub(0.0f, gravity),
            mpas_add(moist_base, moist_perturbation));
        if (!isfinite(rho_base[i]) || !isfinite(density_perturbation[i])
                || !isfinite(qtot[i]) || !isfinite(value))
            atomicExch(invalid, 1);
        dpdz[i] = value;
    }
}
