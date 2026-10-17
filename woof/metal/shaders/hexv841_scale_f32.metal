// Hex v8.4.1 Metal transcription of CUDA kernel scale_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_scale_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_scale_f32(
    constant Hexv841ScaleF32Resources &r [[buffer(0)]],
    constant Hexv841ScaleF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlevels = p.nlevels;
    const int nowners = p.nowners;
    const float scale = p.scale;
    device const float *source = r.source;
    device float *target = r.target;
    const int element = int(tid);
    if (element >= (nlevels * nowners)) return;
    {

        target[element] = mpas_mul(scale, source[element]);
    }
}
