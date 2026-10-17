// Hex v8.4.1 Metal transcription of CUDA kernel validate_finite_array_v841_f32 (woof/hex/cuda_dynamics_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_validate_finite_array_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_validate_finite_array_v841_f32(
    constant Hexv841ValidateFiniteArrayV841F32Resources &r [[buffer(0)]],
    constant Hexv841ValidateFiniteArrayV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nvalues = p.nvalues;
    device const float *values = r.values;
    device int *invalid = r.invalid;
    const int index = int(tid);
    if (index >= nvalues) return;


    if (index < nvalues && !isfinite(values[index])) atomicExch(invalid, 1);

}
