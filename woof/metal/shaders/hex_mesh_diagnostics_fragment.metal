// needs integration: Hex mesh-owned edge diagnostics.
// Edge topology, boundary masks and exact diagnostic ownership belong to ABI integration.
#include <metal_stdlib>
#include "../generated/hex_mesh_diagnostics_fragment_bindings.metalh"
using namespace metal;
kernel void hex_mesh_diagnostics_fragment(
    constant HexMeshDiagnosticsFragmentResources &r [[buffer(0)]],
    constant HexMeshDiagnosticsFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint nlev = uint(p.nlev), ncells = uint(p.ncells);
    const uint nedges = uint(p.nedges);
    const uint count = nlev * nedges;
    if (tid >= count) return;
    const uint level = tid / nedges;
    const uint edge = tid % nedges;
    const int c0 = r.cells_on_edge[2u * edge];
    const int c1 = r.cells_on_edge[2u * edge + 1u];
    const float edge_density = 0.5f *
        (r.rho[level * ncells + uint(c0)] + r.rho[level * ncells + uint(c1)]);
    r.rho_edge[tid] = edge_density;
    r.normal_velocity[tid] = r.rho_u[tid] / edge_density;
}
