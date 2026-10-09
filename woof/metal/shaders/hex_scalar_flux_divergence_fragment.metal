#include <metal_stdlib>
#include "hex_scalar_flux_divergence_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// Finish one already limited MPAS scalar flux column. Edge slots are visited
// in ascending MPAS order and each contribution is accumulated into the
// incoming tendency before the vertical-flux divergence is added.
kernel void hex_scalar_flux_divergence(
        constant HexScalarFluxDivergenceResources &r [[buffer(0)]],
        constant HexScalarFluxDivergenceParams &q [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong count = ulong(q.ntracers) * ulong(q.nlev) * ulong(q.ncells);
    if (q.ntracers <= 0 || q.nlev <= 0 || q.ncells <= 0
            || q.nedges <= 0 || q.max_edges <= 0 || ulong(gid) >= count)
        return;

    const int cell = int(gid % uint(q.ncells));
    const int level = int((gid / uint(q.ncells)) % uint(q.nlev));
    const int tracer = int(gid / (uint(q.ncells) * uint(q.nlev)));
    const ulong horiz_base = (ulong(tracer) * ulong(q.nlev) + ulong(level))
                           * ulong(q.nedges);
    const ulong vertical_base = ulong(tracer) * ulong(q.nlev + 1)
                              * ulong(q.ncells);
    const float inverse_area = 1.0f / r.area_cell[cell];
    float value = r.tendency[gid];
    const int used = r.edge_counts[cell];
    if (used < 0 || used > q.max_edges) return;
    for (int slot = 0; slot < used; ++slot) {
        const int edge = r.edge_indices[cell * q.max_edges + slot];
        if (edge < 0 || edge >= q.nedges) return;
        const ulong mesh_at = ulong(cell) * ulong(q.max_edges) + ulong(slot);
        value = value + r.edge_signs[mesh_at]
                      * r.horizontal_flux[horiz_base + ulong(edge)]
                      * inverse_area;
    }
    const ulong vertical_at = vertical_base + ulong(level) * ulong(q.ncells)
                            + ulong(cell);
    value = value + r.vertical_factor[level]
                  * (r.vertical_flux[vertical_at + ulong(q.ncells)]
                     - r.vertical_flux[vertical_at]);
    r.tendency[gid] = value;
}
