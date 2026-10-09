#include <metal_stdlib>
#include "../generated/hex_fct_apply_fragment_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

kernel void hex_fct_apply_fragment(
    constant HexFctApplyFragmentResources &r [[buffer(0)]],
    constant HexFctApplyFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.ncells);
    if (tid >= count) return;
    const uint level = tid / uint(p.ncells);
    const uint cell = tid % uint(p.ncells);
    const uint nlev = uint(p.nlev);
    const uint storage = uint(p.storage_cells);
    const uint max_edges = uint(p.max_edges);
    const uint mass_index = level * storage + cell;
    const float zero = 0.0f;
    const float one = 1.0f;

    const float inverse_area = p.use_inverse_area != 0
        ? r.inv_area_cell[cell] : one / r.area_cell[cell];
    float mass = r.mass[mass_index];
    const int n_edges = r.n_edges_on_cell[cell];
    for (int slot = 0; slot < n_edges; ++slot) {
        const uint edge = uint(r.edges_on_cell[cell * max_edges + uint(slot)]);
        const float sign = r.edge_signs[cell * max_edges + uint(slot)];
        mass += sign * r.horizontal_residual[level * uint(p.nedges) + edge]
            * inverse_area;
    }

    const uint below_index = level * storage + cell;
    const uint above_index = (level + 1u) * storage + cell;
    const float raw_below = r.vertical_residual[below_index];
    const float raw_above = r.vertical_residual[above_index];
    float limited_below = raw_below;
    float limited_above = raw_above;
    if (level > 0u) {
        limited_below = max(raw_below, zero)
            * min(r.scale_out[(level - 1u) * storage + cell],
                  r.scale_in[level * storage + cell])
            + min(raw_below, zero)
            * min(r.scale_in[(level - 1u) * storage + cell],
                  r.scale_out[level * storage + cell]);
    }
    if (level + 1u < nlev) {
        limited_above = max(raw_above, zero)
            * min(r.scale_out[level * storage + cell],
                  r.scale_in[(level + 1u) * storage + cell])
            + min(raw_above, zero)
            * min(r.scale_in[level * storage + cell],
                  r.scale_out[(level + 1u) * storage + cell]);
    }
    mass -= r.rdnw[level] * (limited_above - limited_below);
    r.updated[mass_index] = max(zero, mass / r.target_density[mass_index]);
}
