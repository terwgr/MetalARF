// Hex v8.4.1 Metal transcription of CUDA kernel clamp_wsm6_v841_f32 (woof/hex/cuda_physics_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_clamp_wsm6_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)


kernel void hexv841_clamp_wsm6_v841_f32(
    constant Hexv841ClampWsm6V841F32Resources &r [[buffer(0)]],
    constant Hexv841ClampWsm6V841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int count = p.count;
    const int apply_clamp = p.apply_clamp;
    device float *values = r.values;
    device int *invalid = r.invalid;

    const int i = int(tid);
    if (i >= count) return;
    const uint bits = as_type<uint>(values[i]);
    const uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {
        atomicExch(invalid, 1);
        return;
    }
    // Floating comparisons are DAZ-sensitive on the certified terminal-FTZ
    // route. Inspect the stored sign bit so -minsub and -0 both become the
    // native +0 bit pattern while every positive subnormal survives exactly.
    if (apply_clamp != 0 && (bits & 0x80000000u) != 0u)
        values[i] = as_type<float>(0u);
}
