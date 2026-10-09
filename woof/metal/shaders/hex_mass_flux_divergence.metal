#include <metal_stdlib>
#include "hex_mass_flux_divergence_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// One output belongs to one (vertical level, MPAS cell) pair.  The slot loop
// retains the source kernel's ascending edgesOnCell accumulation order.
kernel void hex_mass_flux_divergence(
        constant HexMassFluxDivergenceResources &r [[buffer(0)]],
        constant HexMassFluxDivergenceParams &q [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong count = ulong(q.nlev) * ulong(q.ncells);
    if (q.nlev <= 0 || q.ncells <= 0 || q.nedges <= 0
            || q.max_edges <= 0 || gid >= count)
        return;

    const int cell = int(gid % uint(q.ncells));
    const int level = int(gid / uint(q.ncells));
    float sum = 0.0f;
    const int used = r.edge_counts[cell];
    if (used < 0 || used > q.max_edges) return;
    for (int slot = 0; slot < used; ++slot) {
        const int edge = r.edge_indices[cell * q.max_edges + slot];
        if (edge < 0 || edge >= q.nedges) return;
        const ulong ru_at = ulong(level) * ulong(q.nedges) + ulong(edge);
        const ulong mesh_at = ulong(cell) * ulong(q.max_edges) + ulong(slot);
        sum += r.edge_signs[mesh_at] * r.dv_edge[edge] * r.rho_u[ru_at];
    }

    const ulong out_at = ulong(level) * ulong(q.ncells) + ulong(cell);
    if (q.use_inverse != 0)
        r.divergence[out_at] = sum * r.inv_area_cell[cell];
    else
        r.divergence[out_at] = sum / r.area_cell[cell];
}
