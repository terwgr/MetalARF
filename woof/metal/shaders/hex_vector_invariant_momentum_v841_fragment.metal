#include <metal_stdlib>
#include "../generated/hex_vector_invariant_momentum_v841_fragment_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

kernel void hex_vector_invariant_momentum_v841_fragment(
    constant HexVectorInvariantMomentumV841FragmentResources &r [[buffer(0)]],
    constant HexVectorInvariantMomentumV841FragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.nedges);
    if (tid >= count) return;
    const uint level = tid / uint(p.nedges);
    const uint edge = tid % uint(p.nedges);
    const uint nlev = uint(p.nlev);
    const uint nedges = uint(p.nedges);
    const uint ncells = uint(p.ncells);
    const uint max_edges = uint(p.max_edges);
    const uint c0 = uint(r.cells_on_edge[2u * edge]);
    const uint c1 = uint(r.cells_on_edge[2u * edge + 1u]);
    const uint count_on_edge = uint(r.edge_counts[edge]);
    const uint level_edge = level * nedges + edge;

    float q = 0.0f;
    for (uint slot = 0u; slot < count_on_edge; ++slot) {
        const uint neighbor = uint(r.edges_on_edge[edge * max_edges + slot]);
        const float work_pv = 0.5f *
            (r.pv_edge[level_edge] + r.pv_edge[level * nedges + neighbor]);
        q += r.weights_on_edge[edge * max_edges + slot]
            * r.normal_velocity[level * nedges + neighbor] * work_pv;
    }
    for (uint slot = 0u; slot < count_on_edge; ++slot) {
        const uint neighbor = uint(r.edges_on_edge[edge * max_edges + slot]);
        const float reference_u = r.u_init[level]
                * precise::cos(r.angle_edge[neighbor])
            - r.v_init[level] * precise::sin(r.angle_edge[neighbor]);
        q -= r.weights_on_edge[edge * max_edges + slot]
            * reference_u * r.f_edge[edge];
    }
    const uint cell0_index = level * ncells + c0;
    const uint cell1_index = level * ncells + c1;
    const float kinetic_gradient =
        (r.kinetic_energy[cell1_index] - r.kinetic_energy[cell0_index])
        * r.inv_dc_edge[edge];
    const float divergence = 0.5f *
        (r.horizontal_divergence[cell0_index]
         + r.horizontal_divergence[cell1_index]);
    r.tendency[tid] = r.rho_edge[tid] * (q - kinetic_gradient)
        - r.normal_velocity[tid] * divergence;
}
