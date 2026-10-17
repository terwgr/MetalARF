// Hex v8.4.1 Metal transcription of CUDA kernel recover_edges_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_recover_edges_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_recover_edges_f32(
    constant Hexv841RecoverEdgesF32Resources &r [[buffer(0)]],
    constant Hexv841RecoverEdgesF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int nedges = p.nedges;
    const int acoustic_steps = p.acoustic_steps;
    device const float *ru_saved = r.ru_saved;
    device const float *ru_p = r.ru_p;
    device const float *ru_avg_p = r.ru_avg_p;
    device float *ru = r.ru;
    device float *flux_u = r.flux_u;
    const int element = int(tid);
    if (element >= (nlev * nedges)) return;
    {

        const int index = element;
        ru[index] = mpas_add(ru_saved[index], ru_p[index]);
        flux_u[index] = mpas_add(ru_saved[index],
            mpas_div(ru_avg_p[index], (float)acoustic_steps));
    }
}
