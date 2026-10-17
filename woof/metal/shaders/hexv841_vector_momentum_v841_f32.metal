// Hex v8.4.1 Metal transcription of CUDA kernel vector_momentum_v841_f32 (woof/hex/cuda_dynamics_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_vector_momentum_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_vector_momentum_v841_f32(
    constant Hexv841VectorMomentumV841F32Resources &r [[buffer(0)]],
    constant Hexv841VectorMomentumV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges2 = p.max_edges2;
    device const float *u = r.u;
    device const float *rho_edge = r.rho_edge;
    device const float *pv_edge = r.pv_edge;
    device const float *kinetic = r.kinetic;
    device const float *mass_divergence = r.mass_divergence;
    device const int *cells_on_edge = r.cells_on_edge;
    device const int *edges_on_edge = r.edges_on_edge;
    device const int *n_edges_on_edge = r.n_edges_on_edge;
    device const float *weights_on_edge = r.weights_on_edge;
    device const float *inv_dc_edge = r.inv_dc_edge;
    device const float *reference_u = r.reference_u;
    device const float *f_edge = r.f_edge;
    device float *result = r.result;
    const int element = int(tid);
    if (element >= (nlev * nedges)) return;
    {

        const int edge = element % nedges;
        const int k = element / nedges;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        const int index = E2(k, edge, nedges);
        float q = 0.0f;
        for (int slot = 0; slot < n_edges_on_edge[edge]; ++slot) {
            const int offset = edge * max_edges2 + slot;
            const int neighbor = edges_on_edge[offset];
            const float work_pv = mpas_mul(0.5f, mpas_add(
                pv_edge[index], pv_edge[E2(k, neighbor, nedges)]));
            q = mpas_add(q, mpas_mul(mpas_mul(
                weights_on_edge[offset], u[E2(k, neighbor, nedges)]), work_pv));
        }
        for (int slot = 0; slot < n_edges_on_edge[edge]; ++slot) {
            const int offset = edge * max_edges2 + slot;
            const int neighbor = edges_on_edge[offset];
            q = mpas_sub(q, mpas_mul(mpas_mul(
                weights_on_edge[offset], reference_u[E2(k, neighbor, nedges)]),
                f_edge[edge]));
        }
        const float gradient = mpas_mul(mpas_sub(
            kinetic[C2(k, c1, ncells)], kinetic[C2(k, c0, ncells)]),
            inv_dc_edge[edge]);
        result[index] = mpas_sub(mpas_mul(rho_edge[index],
            mpas_sub(q, gradient)), mpas_mul(mpas_mul(u[index], 0.5f),
                mpas_add(mass_divergence[C2(k, c0, ncells)],
                    mass_divergence[C2(k, c1, ncells)])));
    }
}
