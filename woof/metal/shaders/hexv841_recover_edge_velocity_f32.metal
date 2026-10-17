// Hex v8.4.1 Metal transcription of CUDA kernel recover_edge_velocity_f32 (woof/hex/cuda_backend/recovery.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_recover_edge_velocity_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_recover_edge_velocity_f32(
    constant Hexv841RecoverEdgeVelocityF32Resources &r [[buffer(0)]],
    constant Hexv841RecoverEdgeVelocityF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    device const float *rho = r.rho;
    device const float *ru = r.ru;
    device const int *cells_on_edge = r.cells_on_edge;
    device float *u = r.u;
    const int element = int(tid);
    if (element >= (nlev * nedges)) return;
    {

    const int edge = element % nedges;
    const int level = element / nedges;
    const int index = level * nedges + edge;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    const float denominator = mpas_add(
        rho[level * ncells + cell0], rho[level * ncells + cell1]);
    u[index] = mpas_div(mpas_mul(2.0f, ru[index]), denominator);
    }
}
