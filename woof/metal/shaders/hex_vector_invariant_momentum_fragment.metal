// needs integration: Hex non-v8.4.1 vector-invariant momentum tendency.
#include <metal_stdlib>
#include "hex_vector_invariant_momentum_fragment_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void hex_vector_invariant_momentum_fragment(
    constant HexVectorInvariantMomentumFragmentResources &r [[buffer(0)]],
    constant HexVectorInvariantMomentumFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.nedges);
    if (tid >= count) return;
    const uint level = tid / uint(p.nedges);
    const uint edge = tid % uint(p.nedges);
    const uint edge_offset = level * uint(p.nedges) + edge;
    const int cell0 = r.cells_on_edge[2u * edge];
    const int cell1 = r.cells_on_edge[2u * edge + 1u];
    const uint cell_offset0 = level * uint(p.ncells) + uint(cell0);
    const uint cell_offset1 = level * uint(p.ncells) + uint(cell1);
    float q = 0.0f;
    for (uint slot = 0u; slot < uint(r.edge_counts[edge]); ++slot) {
        const uint neighbor = uint(r.edges_on_edge[edge * uint(p.max_edges) + slot]);
        const uint neighbor_offset = level * uint(p.nedges) + neighbor;
        const float work_pv = 0.5f *
            (r.pv_edge[edge_offset] + r.pv_edge[neighbor_offset]);
        q += r.weights_on_edge[edge * uint(p.max_edges) + slot] *
             r.normal_velocity[neighbor_offset] * work_pv;
    }
    const float kinetic_gradient =
        (r.kinetic_energy[cell_offset1] - r.kinetic_energy[cell_offset0]) *
        r.inv_dc[edge];
    const float mean_divergence = 0.5f *
        (r.horizontal_divergence[cell_offset0] +
         r.horizontal_divergence[cell_offset1]);
    r.tendency[edge_offset] = r.rho_edge[edge_offset] *
        (q - kinetic_gradient) - r.normal_velocity[edge_offset] * mean_divergence;
}
