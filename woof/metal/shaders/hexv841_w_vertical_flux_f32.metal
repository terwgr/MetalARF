// Hex v8.4.1 Metal transcription of CUDA kernel w_vertical_flux_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_w_vertical_flux_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_w_vertical_flux_f32(
    constant Hexv841WVerticalFluxF32Resources &r [[buffer(0)]],
    constant Hexv841WVerticalFluxF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    device const float *w = r.w;
    device const float *rw = r.rw;
    device float *flux = r.flux;
    const int element = int(tid);
    if (element >= ((nlev + 1) * ncells)) return;
    {

        const int cell = element % ncells;
        const int k = element / ncells;
        const int index = C2(k, cell, ncells);
        float value = 0.0f;
        if (k > 0 && k < nlev) {
            const float velocity = mpas_mul(0.5f, mpas_add(
                rw[index], rw[C2(k - 1, cell, ncells)]));
            if (k == 1 || k == nlev - 1) {
                value = mpas_mul(mpas_mul(0.5f, velocity), mpas_add(
                    w[index], w[C2(k - 1, cell, ncells)]));
            } else {
                const float qm2 = w[C2(k - 2, cell, ncells)];
                const float qm1 = w[C2(k - 1, cell, ncells)];
                const float qi = w[index];
                const float qp1 = w[C2(k + 1, cell, ncells)];
                value = mpas_add(mpas_div(mpas_mul(velocity, mpas_sub(
                    mpas_mul(7.0f, mpas_add(qi, qm1)),
                    mpas_add(qp1, qm2))),
                    mpas_third_order_denominator), mpas_div(mpas_mul(
                        mpas_abs(velocity), mpas_sub(mpas_sub(qp1, qm2),
                            mpas_mul(3.0f, mpas_sub(qi, qm1)))),
                        mpas_third_order_denominator));
            }
        }
        flux[index] = value;
    }
}
