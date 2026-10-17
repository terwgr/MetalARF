// Hex v8.4.1 scalar transport: connectivity validation (cuda_transport_v841.py validate_transport_indices_v841).
#include <metal_stdlib>
#include "hexv841_validate_transport_indices_v841_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,width) ((e)*(width) + (s))

kernel void hexv841_validate_transport_indices_v841(
    constant Hexv841ValidateTransportIndicesV841Resources &r [[buffer(0)]],
    constant Hexv841ValidateTransportIndicesV841Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int ncells = p.ncells, nedges = p.nedges;
    const int max_edges = p.max_edges, width = p.width;
    device atomic_int *invalid =
        reinterpret_cast<device atomic_int *>(r.invalid);
    const int index = int(tid);
    if (index < ncells) {
        const int count = r.n_edges_on_cell[index];
        if (count < 0 || count > max_edges)
            atomic_exchange_explicit(invalid, 1, memory_order_relaxed);
        const int safe_count = count < 0 ? 0 : (count > max_edges ? max_edges : count);
        for (int slot = 0; slot < safe_count; ++slot) {
            const int edge = r.edges_on_cell[CES(index, slot, max_edges)];
            const int cell = r.cells_on_cell[CES(index, slot, max_edges)];
            if (edge < 0 || edge >= nedges || cell < 0 || cell >= ncells)
                atomic_exchange_explicit(invalid, 1, memory_order_relaxed);
        }
    }
    if (index < nedges) {
        const int c0 = r.cells_on_edge[2 * index];
        const int c1 = r.cells_on_edge[2 * index + 1];
        if (c0 < 0 || c0 >= ncells || c1 < 0 || c1 >= ncells)
            atomic_exchange_explicit(invalid, 1, memory_order_relaxed);
        const int count = r.n_adv[index];
        if (count < 0 || count > width)
            atomic_exchange_explicit(invalid, 1, memory_order_relaxed);
        const int safe_count = count < 0 ? 0 : (count > width ? width : count);
        for (int slot = 0; slot < safe_count; ++slot) {
            const int cell = r.adv_cells[ADV(index, slot, width)];
            if (cell < 0 || cell >= ncells)
                atomic_exchange_explicit(invalid, 1, memory_order_relaxed);
        }
    }
}
