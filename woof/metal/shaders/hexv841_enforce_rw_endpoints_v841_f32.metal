// Hex v8.4.1 Metal transcription of CUDA kernel enforce_rw_endpoints_v841_f32 (woof/hex/cuda_dynamics_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_enforce_rw_endpoints_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_enforce_rw_endpoints_v841_f32(
    constant Hexv841EnforceRwEndpointsV841F32Resources &r [[buffer(0)]],
    constant Hexv841EnforceRwEndpointsV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    device float *rw = r.rw;
    const int cell = int(tid);
    if (cell >= ncells) return;


    rw[C2(0, cell, ncells)] = 0.0f;
    rw[C2(nlev, cell, ncells)] = 0.0f;

}
