// needs integration: Hex horizontal rho-theta transport tendency.
#include <metal_stdlib>
#include "hex_horizontal_theta_tendency_fragment_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void hex_horizontal_theta_tendency_fragment(
    constant HexHorizontalThetaTendencyFragmentResources &r [[buffer(0)]],
    constant HexHorizontalThetaTendencyFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.ncells);
    if (tid >= count) return;
    const uint level = tid / uint(p.ncells);
    const uint cell = tid % uint(p.ncells);
    float tendency = 0.0f;
    if (r.skip_cell[cell] == 0) {
        for (uint slot = 0u; slot < uint(r.edge_counts[cell]); ++slot) {
            const uint edge = uint(r.edges_on_cell[cell * uint(p.max_cell_edges) + slot]);
            const int nstencil = r.n_adv_cells_for_edge[edge];
            const float velocity = r.rho_u[level * uint(p.nedges) + edge];
            const float sign_velocity = copysign(1.0f, velocity);
            float edge_theta = 0.0f;
            for (int stencil_slot = 0; stencil_slot < nstencil; ++stencil_slot) {
                const uint stencil = edge * uint(p.max_adv_cells) + uint(stencil_slot);
                const uint adv_cell = uint(r.adv_cells_for_edge[stencil]);
                const float weight = r.adv_coefs[stencil]
                    + p.third_order_coefficient * sign_velocity * r.adv_coefs_3rd[stencil];
                edge_theta += weight * r.theta[level * uint(p.ncells) + adv_cell];
            }
            tendency -= r.edge_signs[cell * uint(p.max_cell_edges) + slot]
                * velocity * edge_theta;
        }
        if (p.rk_step > 1) {
            for (uint slot = 0u; slot < uint(r.edge_counts[cell]); ++slot) {
                const uint edge = uint(r.edges_on_cell[cell * uint(p.max_cell_edges) + slot]);
                const uint cell0 = uint(r.cells_on_edge[2u * edge]);
                const uint cell1 = uint(r.cells_on_edge[2u * edge + 1u]);
                const uint at_edge = level * uint(p.nedges) + edge;
                const float correction =
                    r.edge_signs[cell * uint(p.max_cell_edges) + slot]
                    * r.dv_edge[edge]
                    * (r.rho_u_saved[at_edge] - r.rho_u[at_edge])
                    * 0.5f
                    * (r.theta_saved[level * uint(p.ncells) + cell1]
                       + r.theta_saved[level * uint(p.ncells) + cell0]);
                tendency -= correction;
            }
        }
        tendency *= r.inv_area_cell[cell];
    }
    r.tendency[tid] = tendency;
}
