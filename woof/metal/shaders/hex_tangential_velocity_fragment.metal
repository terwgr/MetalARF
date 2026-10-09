// needs integration: Hex edge-neighbor tangential velocity reconstruction.
#include <metal_stdlib>
#include "hex_tangential_velocity_fragment_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void hex_tangential_velocity_fragment(
    constant HexTangentialVelocityFragmentResources &r [[buffer(0)]],
    constant HexTangentialVelocityFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.nedges);
    if (tid >= count) return;
    const uint level = tid / uint(p.nedges);
    const uint edge = tid % uint(p.nedges);
    float value = 0.0f;
    for (int slot = 0; slot < r.n_edges_on_edge[edge]; ++slot) {
        const int neighbor = r.edges_on_edge[edge * uint(p.max_edges) + uint(slot)];
        value += r.weights_on_edge[edge * uint(p.max_edges) + uint(slot)]
            * r.normal_velocity[level * uint(p.nedges) + uint(neighbor)];
    }
    r.tangential_velocity[tid] = value;
}
