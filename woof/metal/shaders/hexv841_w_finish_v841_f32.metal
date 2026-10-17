// Hex v8.4.1 Metal transcription of CUDA kernel w_finish_v841_f32 (woof/hex/cuda_dynamics_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_w_finish_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_w_finish_v841_f32(
    constant Hexv841WFinishV841F32Resources &r [[buffer(0)]],
    constant Hexv841WFinishV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges = p.max_edges;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *edges_on_cell = r.edges_on_cell;
    device const float *acoustic_sign = r.acoustic_sign;
    device const float *inv_area_cell = r.inv_area_cell;
    device const float *rdzu = r.rdzu;
    device const float *edge_flux = r.edge_flux;
    device const float *vertical_flux = r.vertical_flux;
    device float *result = r.result;
    const int element = int(tid);
    if (element >= ((nlev + 1) * ncells)) return;
    {

        const int cell = element % ncells;
        const int k = element / ncells;
        const int index = C2(k, cell, ncells);
        if (k == 0 || k == nlev) {
            result[index] = 0.0f;
        } else {
            float value = 0.0f;
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                value = mpas_sub(value, mpas_mul(
                    acoustic_sign[CES(cell, slot, max_edges)],
                    edge_flux[E2(k, edge, nedges)]));
            }
            value = mpas_mul(value, inv_area_cell[cell]);
            value = mpas_sub(value, mpas_mul(rdzu[k], mpas_sub(
                vertical_flux[C2(k + 1, cell, ncells)], vertical_flux[index])));
            result[index] = value;
        }
    }
}
