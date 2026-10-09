// needs integration: terrain-coupled Hex interface-velocity recovery.
#include <metal_stdlib>
#include "../generated/hex_interface_velocity_terrain_fragment_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void hex_interface_velocity_terrain_fragment(
    constant HexInterfaceVelocityTerrainFragmentResources &r [[buffer(0)]],
    constant HexInterfaceVelocityTerrainFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint nlev = uint(p.nlev), ncells = uint(p.ncells);
    const uint nedges = uint(p.nedges), max_edges = uint(p.max_edges);
    const uint count = (nlev + 1u) * ncells;
    if (nlev < 3u || ncells == 0u || max_edges == 0u || tid >= count)
        return;
    const uint level = tid / ncells;
    const uint cell = tid % ncells;
    if (level == nlev) {
        r.velocity[tid] = 0.0f;
        return;
    }

    float sum = 0.0f;
    const int active_edges = r.n_edges_on_cell[cell];
    for (int slot = 0; slot < active_edges; ++slot) {
        const int edge = r.edges_on_cell[cell * p.max_edges + uint(slot)];
        if (edge < 0 || uint(edge) >= nedges) continue;
        float flux;
        if (level == 0u) {
            flux = p.cf1 * r.rho_u[uint(edge)]
                 + p.cf2 * r.rho_u[nedges + uint(edge)]
                 + p.cf3 * r.rho_u[2u * nedges + uint(edge)];
        } else {
            flux = r.fzm[level] * r.rho_u[level * nedges + uint(edge)]
                 + r.fzp[level] * r.rho_u[(level - 1u) * nedges + uint(edge)];
        }
        const uint metric_at = level * ncells * max_edges
                             + cell * max_edges + uint(slot);
        const float terrain = r.zb_cell[metric_at]
                            + copysign(1.0f, flux) * r.zb3_cell[metric_at];
        sum = sum + r.edge_sign[cell * p.max_edges + uint(slot)] * terrain * flux;
    }

    float density;
    if (level == 0u) {
        const uint at = cell;
            density = p.cf1 * r.rho[at]
                + p.cf2 * r.rho[ncells + at]
                + p.cf3 * r.rho[2u * ncells + at];
        } else {
        density = r.fzm[level] * r.rho[level * ncells + cell]
                + r.fzp[level] * r.rho[(level - 1u) * ncells + cell];
    }
    r.velocity[tid] = sum / density;
}
