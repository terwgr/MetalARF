// Hex v8.4.1 Metal transcription of CUDA kernel vertical_u_finish_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_vertical_u_finish_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_vertical_u_finish_f32(
    constant Hexv841VerticalUFinishF32Resources &r [[buffer(0)]],
    constant Hexv841VerticalUFinishF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int nedges = p.nedges;
    device const float *rdzw = r.rdzw;
    device const float *flux = r.flux;
    device float *result = r.result;
    const int element = int(tid);
    if (element >= (nlev * nedges)) return;
    {

        const int edge = element % nedges;
        const int k = element / nedges;
        const int index = E2(k, edge, nedges);
        result[index] = mpas_mul(mpas_sub(0.0f, rdzw[k]), mpas_sub(
            flux[E2(k + 1, edge, nedges)], flux[index]));
    }
}
