// Hex v8.4.1 Metal transcription of CUDA kernel euler_w_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_euler_w_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_euler_w_f32(
    constant Hexv841EulerWF32Resources &r [[buffer(0)]],
    constant Hexv841EulerWF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    device const float *pressure_p = r.pressure_p;
    device const float *dpdz = r.dpdz;
    device const float *rdzu = r.rdzu;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device float *result = r.result;
    const int element = int(tid);
    if (element >= ((nlev + 1) * ncells)) return;
    {

        const int cell = element % ncells;
        const int k = element / ncells;
        const int index = C2(k, cell, ncells);
        if (k == 0 || k == nlev) {
            result[index] = 0.0f;
        } else {
            result[index] = mpas_sub(0.0f, mpas_sub(mpas_mul(rdzu[k], mpas_sub(
                pressure_p[C2(k, cell, ncells)],
                pressure_p[C2(k - 1, cell, ncells)])), mpas_add(
                    mpas_mul(fzm[k], dpdz[C2(k, cell, ncells)]),
                    mpas_mul(fzp[k],
                        dpdz[C2(k - 1, cell, ncells)]))));
        }
    }
}
