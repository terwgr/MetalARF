// Hex v8.4.1 Metal transcription of CUDA kernel euler_w_moist_v841_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_euler_w_moist_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)


kernel void hexv841_euler_w_moist_v841_f32(
    constant Hexv841EulerWMoistV841F32Resources &r [[buffer(0)]],
    constant Hexv841EulerWMoistV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    device const float *pressure_p = r.pressure_p;
    device const float *dpdz = r.dpdz;
    device const float *cqw = r.cqw;
    device const float *rdzu = r.rdzu;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device float *result = r.result;
    device int *invalid = r.invalid;

    const int cell = int(tid);
    if (cell >= ncells) return;
    for (int k = 0; k <= nlev; ++k) {
        const int index = C2(k, cell, ncells);
        if (k == 0 || k == nlev) {
            result[index] = 0.0f;
        } else {
            const float pressure_term = mpas_mul(rdzu[k], mpas_sub(
                pressure_p[C2(k, cell, ncells)],
                pressure_p[C2(k - 1, cell, ncells)]));
            const float buoyancy = mpas_add(
                mpas_mul(fzm[k], dpdz[C2(k, cell, ncells)]),
                mpas_mul(fzp[k], dpdz[C2(k - 1, cell, ncells)]));
            const float value = mpas_sub(0.0f, mpas_mul(cqw[index],
                mpas_sub(pressure_term, buoyancy)));
            if (!isfinite(cqw[index]) || !isfinite(value))
                atomicExch(invalid, 1);
            result[index] = value;
        }
    }
}
