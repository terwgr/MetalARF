// Hex v8.4.1 Metal transcription of CUDA kernel add_inplace_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_add_inplace_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_add_inplace_f32(
    constant Hexv841AddInplaceF32Resources &r [[buffer(0)]],
    constant Hexv841AddInplaceF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlevels = p.nlevels;
    const int nowners = p.nowners;
    device const float *increment = r.increment;
    device float *target = r.target;
    const int element = int(tid);
    if (element >= (nlevels * nowners)) return;
    {

        target[element] = mpas_add(target[element], increment[element]);
    }
}
