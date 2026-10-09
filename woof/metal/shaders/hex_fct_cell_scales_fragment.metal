#include <metal_stdlib>
#include "../generated/hex_fct_cell_scales_fragment_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

// One cell-level owner computes local extrema, low-order mass, and both
// Zalesak scale factors. Each cell writes only its own column.
kernel void hex_fct_cell_scales_fragment(
    constant HexFctCellScalesFragmentResources &r [[buffer(0)]],
    constant HexFctCellScalesFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.ncells);
    if (tid >= count) return;
    const uint level = tid / uint(p.ncells);
    const uint cell = tid % uint(p.ncells);
    const uint nlev = uint(p.nlev);
    const uint ncells = uint(p.ncells);
    const uint storage = uint(p.storage_cells);
    const uint max_edges = uint(p.max_edges);
    const uint cell_index = level * storage + cell;
    const float dt = p.dt;
    const float zero = 0.0f;
    const float one = 1.0f;
    const float eps = 1.0e-20f;

    float minimum = r.old_scalar[cell_index];
    float maximum = minimum;
    const uint lower = level > 0u ? level - 1u : 0u;
    const uint upper = min(nlev, level + 2u);
    for (uint k = lower; k < upper; ++k) {
        const float value = r.old_scalar[k * storage + cell];
        minimum = min(minimum, value);
        maximum = max(maximum, value);
    }
    const int n_edges = r.n_edges_on_cell[cell];
    for (int slot = 0; slot < n_edges; ++slot) {
        const int edge = r.edges_on_cell[cell * max_edges + uint(slot)];
        const int c0 = r.cells_on_edge[2u * uint(edge)];
        const int c1 = r.cells_on_edge[2u * uint(edge) + 1u];
        const uint neighbor = uint(c0 == int(cell) ? c1 : c0);
        const float value = r.old_scalar[level * storage + neighbor];
        minimum = min(minimum, value);
        maximum = max(maximum, value);
    }

    float mass = r.old_scalar[cell_index] * r.rho_old[cell_index];
    float vertical_in = zero;
    float vertical_out = zero;
    for (uint interface = 1u; interface < nlev; ++interface) {
        const float velocity = r.vertical_velocity[interface * storage + cell];
        const float flux = dt * (max(zero, velocity)
            * r.old_scalar[(interface - 1u) * storage + cell]
            + min(zero, velocity)
            * r.old_scalar[interface * storage + cell]);
        if (level + 1u == interface) mass -= flux * r.rdnw[level];
        if (level == interface) mass += flux * r.rdnw[level];
    }
    const uint vertical_index = level * storage + cell;
    const float below = r.vertical_velocity[level * storage + cell];
    const float above = r.vertical_velocity[(level + 1u) * storage + cell];
    const float flux_below = level == 0u ? zero : dt * (
        max(zero, below) * r.old_scalar[(level - 1u) * storage + cell]
        + min(zero, below) * r.old_scalar[level * storage + cell]);
    const float flux_above = level + 1u >= nlev ? zero : dt * (
        max(zero, above) * r.old_scalar[level * storage + cell]
        + min(zero, above) * r.old_scalar[(level + 1u) * storage + cell]);
    const float residual_below =
        dt * r.high_vertical[vertical_index] - flux_below;
    const float residual_above =
        dt * r.high_vertical[(level + 1u) * storage + cell] - flux_above;
    vertical_in = -r.rdnw[level] *
        (min(zero, residual_above) - max(zero, residual_below));
    vertical_out = -r.rdnw[level] *
        (max(zero, residual_above) - min(zero, residual_below));
    r.vertical_residual[vertical_index] =
        residual_below;
    if (level + 1u == nlev) {
        r.vertical_residual[nlev * storage + cell] =
            dt * r.high_vertical[nlev * storage + cell];
    }

    const float inverse_area = p.use_inverse_area != 0
        ? r.inv_area_cell[cell] : one / r.area_cell[cell];
    float scale_in = vertical_in;
    float scale_out = vertical_out;
    for (int slot = 0; slot < n_edges; ++slot) {
        const uint edge = uint(r.edges_on_cell[cell * max_edges + uint(slot)]);
        const float sign = r.edge_signs[cell * max_edges + uint(slot)];
        mass += sign * r.upwind_flux[level * uint(p.nedges) + edge] * inverse_area;
        const float signed_residual = -sign
            * r.horizontal_residual[level * uint(p.nedges) + edge];
        scale_out -= max(zero, signed_residual) * inverse_area;
        scale_in -= min(zero, signed_residual) * inverse_area;
    }
    const float target = r.target_density[cell_index];
    const float in_factor = (maximum * target - mass) / (scale_in + eps);
    const float out_factor = (minimum * target - mass) / (scale_out - eps);
    const float bounded_in = min(one, max(zero, in_factor));
    const float bounded_out = min(one, max(zero, out_factor));
    r.mass[cell_index] = mass;
    r.scale_in[cell_index] = bounded_in;
    r.scale_out[cell_index] = bounded_out;
}
