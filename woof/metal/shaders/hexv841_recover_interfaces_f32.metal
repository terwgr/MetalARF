// Hex v8.4.1 Metal transcription of CUDA kernel recover_interfaces_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_recover_interfaces_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_recover_interfaces_f32(
    constant Hexv841RecoverInterfacesF32Resources &r [[buffer(0)]],
    constant Hexv841RecoverInterfacesF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int acoustic_steps = p.acoustic_steps;
    device const float *rw_saved = r.rw_saved;
    device const float *rw_p = r.rw_p;
    device const float *ww_avg_p = r.ww_avg_p;
    device float *rw = r.rw;
    device float *flux_w = r.flux_w;
    const int element = int(tid);
    if (element >= ((nlev + 1) * ncells)) return;
    {

        const int k = element / ncells;
        const int index = element;
        if (k == 0 || k == nlev) {
            rw[index] = 0.0f;
        } else {
            rw[index] = mpas_add(rw_saved[index], rw_p[index]);
        }
        flux_w[index] = mpas_add(rw_saved[index],
            mpas_div(ww_avg_p[index], (float)acoustic_steps));
    }
}
