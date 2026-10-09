#include <metal_stdlib>
#include "hex_vertical_velocity_tendency_fragment_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

static inline float hex_w_flux3(float qim2, float qim1, float qi, float qip1,
                                float velocity) {
    const float flux4 = velocity * (7.0f * (qi + qim1) - (qip1 + qim2))
        / 12.0f;
    return flux4 + abs(velocity)
        * ((qip1 - qim2) - 3.0f * (qi - qim1)) / 12.0f;
}

static inline float hex_w_vertical_flux(
    constant HexVerticalVelocityTendencyFragmentResources &r,
    constant HexVerticalVelocityTendencyFragmentParams &p,
    uint interface, uint cell) {
    const uint nlev = uint(p.nlev), ncells = uint(p.ncells);
    if (interface == 0u || interface >= nlev) return 0.0f;
    const uint current = interface * ncells + cell;
    const uint below = (interface - 1u) * ncells + cell;
    const float rho_w_mid = 0.5f * (r.rho_w[current] + r.rho_w[below]);
    if (interface == 1u || interface + 1u == nlev) {
        return 0.25f * (r.rho_w[current] + r.rho_w[below])
            * (r.normal_velocity[current] + r.normal_velocity[below]);
    }
    const uint before = (interface - 2u) * ncells + cell;
    const uint after = (interface + 1u) * ncells + cell;
    return hex_w_flux3(r.normal_velocity[before], r.normal_velocity[below],
        r.normal_velocity[current], r.normal_velocity[after], rho_w_mid);
}

kernel void hex_vertical_velocity_tendency_fragment(
    constant HexVerticalVelocityTendencyFragmentResources &r [[buffer(0)]],
    constant HexVerticalVelocityTendencyFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = (uint(p.nlev) + 1u) * uint(p.ncells);
    if (tid >= count) return;
    const uint nlev = uint(p.nlev), ncells = uint(p.ncells);
    const uint level = tid / ncells, cell = tid % ncells;
    if (level == 0u || level >= nlev) {
        r.tendency[tid] = 0.0f;
        return;
    }

    float horizontal = 0.0f;
    if (r.skip_cell[cell] == 0) {
        const int n_edges = r.n_edges_on_cell[cell];
        for (int slot = 0; slot < n_edges; ++slot) {
            const uint edge = uint(r.edges_on_cell[
                cell * uint(p.max_edges) + uint(slot)]);
            const float ru_interface =
                r.fzm[level] * r.rho_u[level * uint(p.nedges) + edge]
                + r.fzp[level] * r.rho_u[(level - 1u) * uint(p.nedges) + edge];
            const float direction = copysign(1.0f, ru_interface);
            float edge_w = 0.0f;
            const int n_adv = r.n_adv_cells_for_edge[edge];
            for (int stencil = 0; stencil < n_adv; ++stencil) {
                const uint offset = edge * uint(p.max_adv) + uint(stencil);
                const uint adv_cell = uint(r.adv_cells_for_edge[offset]);
                const float weight = r.adv_coefs[offset]
                    + p.coefficient * direction * r.adv_coefs_3rd[offset];
                edge_w += weight * r.normal_velocity[level * ncells + adv_cell];
            }
            horizontal -= r.edge_signs[cell * uint(p.max_edges) + uint(slot)]
                * ru_interface * edge_w;
        }
    }
    if (p.use_inverse_area != 0) {
        horizontal *= r.inv_area_cell[cell];
    } else {
        horizontal /= r.area_cell[cell];
    }

    const float flux_above = hex_w_vertical_flux(r, p, level + 1u, cell);
    const float flux_below = hex_w_vertical_flux(r, p, level, cell);
    r.tendency[tid] = horizontal
        - r.rdzu[level] * (flux_above - flux_below);
}
