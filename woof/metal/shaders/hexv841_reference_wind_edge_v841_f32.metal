// Hex v8.4.1 Metal transcription of CUDA kernel reference_wind_edge_v841_f32 (woof/hex/cuda_dynamics_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_reference_wind_edge_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_reference_wind_edge_v841_f32(
    constant Hexv841ReferenceWindEdgeV841F32Resources &r [[buffer(0)]],
    constant Hexv841ReferenceWindEdgeV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int nedges = p.nedges;
    device const float *angle_edge = r.angle_edge;
    device const float *u_init = r.u_init;
    device const float *v_init = r.v_init;
    device float *reference_u = r.reference_u;
    const int element = int(tid);
    if (element >= (nlev * nedges)) return;
    {

        const int edge = element % nedges;
        const int k = element / nedges;
        reference_u[E2(k, edge, nedges)] = mpas_sub(
            mpas_mul(u_init[k], cos(angle_edge[edge])),
            mpas_mul(v_init[k], sin(angle_edge[edge])));
    }
}
