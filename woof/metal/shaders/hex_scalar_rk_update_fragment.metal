#include <metal_stdlib>
#include "../generated/hex_scalar_rk_update_fragment_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

inline float hex_scalar_vertical_flux(
        constant HexScalarRkUpdateFragmentResources &r,
        constant HexScalarRkUpdateFragmentParams &p,
        uint tracer, uint cell, uint interface) {
    if (interface == 0u || interface >= uint(p.nlev)) return 0.0f;
    const ulong qbase = (ulong(tracer) * ulong(p.nlev) * ulong(p.ncells))
                      + ulong(cell);
    const ulong wbase = ulong(interface) * ulong(p.ncells) + ulong(cell);
    const float w = r.vertical_velocity[wbase];
    if (p.vertical_order == 2 || interface == 1u
            || interface == uint(p.nlev - 1)) {
        const ulong upper = qbase + ulong(interface) * ulong(p.ncells);
        const ulong lower = qbase + ulong(interface - 1u) * ulong(p.ncells);
        return w * (r.fzm[interface] * r.scalar_stage[upper]
                  + r.fzp[interface] * r.scalar_stage[lower]);
    }

    const ulong q0 = qbase + ulong(interface - 2u) * ulong(p.ncells);
    const ulong q1 = qbase + ulong(interface - 1u) * ulong(p.ncells);
    const ulong q2 = qbase + ulong(interface) * ulong(p.ncells);
    const ulong q3 = qbase + ulong(interface + 1u) * ulong(p.ncells);
    const float v0 = r.scalar_stage[q0];
    const float v1 = r.scalar_stage[q1];
    const float v2 = r.scalar_stage[q2];
    const float v3 = r.scalar_stage[q3];
    const float base = w * (7.0f * (v2 + v1) - (v3 + v0)) / 12.0f;
    if (p.vertical_order == 4) return base;
    const float correction = p.vertical_coefficient * abs(w)
        * ((v3 - v0) - 3.0f * (v2 - v1)) / 12.0f;
    return base + correction;
}

inline float hex_scalar_horizontal_value(
        constant HexScalarRkUpdateFragmentResources &r,
        constant HexScalarRkUpdateFragmentParams &p,
        uint tracer, uint level, uint edge) {
    const ulong stage_base = (ulong(tracer) * ulong(p.nlev) + ulong(level))
                           * ulong(p.ncells);
    if (p.horizontal_order == 2) {
        const uint c0 = uint(r.cells_on_edge[2u * edge]);
        const uint c1 = uint(r.cells_on_edge[2u * edge + 1u]);
        return r.dv_edge[edge] * 0.5f
             * (r.scalar_stage[stage_base + ulong(c0)]
                + r.scalar_stage[stage_base + ulong(c1)]);
    }
    const float velocity = r.horizontal_velocity[
        ulong(level) * ulong(p.nedges) + ulong(edge)];
    const float sign = velocity < 0.0f ? -1.0f : 1.0f;
    const float coefficient = p.horizontal_order == 3
                            ? p.horizontal_coefficient : 0.0f;
    const int count = r.n_adv_cells_for_edge[edge];
    if (count < 0 || count > p.max_adv) return 0.0f;
    float value = 0.0f;
    for (int slot = 0; slot < count; ++slot) {
        const ulong at = ulong(edge) * ulong(p.max_adv) + ulong(slot);
        const uint source_cell = uint(r.adv_cells_for_edge[at]);
        const float weight = r.adv_coefs[at]
                           + coefficient * sign * r.adv_coefs_3rd[at];
        value = value + weight * r.scalar_stage[
            stage_base + ulong(source_cell)];
    }
    return value;
}

kernel void hex_scalar_rk_update_fragment(
        constant HexScalarRkUpdateFragmentResources &r [[buffer(0)]],
        constant HexScalarRkUpdateFragmentParams &p [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong count = ulong(p.ntracers) * ulong(p.nlev) * ulong(p.ncells);
    if (p.ntracers <= 0 || p.nlev <= 1 || p.ncells <= 0 || p.nedges <= 0
            || p.max_edges <= 0 || p.max_adv <= 0 || ulong(gid) >= count)
        return;
    const uint cell = uint(gid % uint(p.ncells));
    const uint level = uint((gid / uint(p.ncells)) % uint(p.nlev));
    const uint tracer = uint(gid / (uint(p.ncells) * uint(p.nlev)));
    const int used = r.edge_counts[cell];
    if (used < 0 || used > p.max_edges) return;

    float tendency = r.source[gid];
    float horizontal_sum = 0.0f;
    for (int slot = 0; slot < used; ++slot) {
        const ulong edge_at = ulong(cell) * ulong(p.max_edges) + ulong(slot);
        const uint edge = uint(r.edges_on_cell[edge_at]);
        if (edge >= uint(p.nedges)) return;
        const float edge_value = hex_scalar_horizontal_value(r, p, tracer,
                                                              level, edge);
        const float velocity = r.horizontal_velocity[
            ulong(level) * ulong(p.nedges) + ulong(edge)];
        const float flux = r.edge_signs[edge_at] * velocity * edge_value;
        if (p.use_inverse_area != 0) {
            horizontal_sum = horizontal_sum + flux;
        } else {
            tendency = tendency + flux / r.area_cell[cell];
        }
    }
    if (p.use_inverse_area != 0) {
        tendency = horizontal_sum * r.inv_area_cell[cell] + tendency;
    }

    const ulong density_at = ulong(level) * ulong(p.ncells) + ulong(cell);
    const float flux_upper = hex_scalar_vertical_flux(r, p, tracer, cell,
                                                       level + 1u);
    const float flux_lower = hex_scalar_vertical_flux(r, p, tracer, cell,
                                                       level);
    const float numerator = r.scalar_old[gid] * r.rho_old[density_at]
        + p.dt * (tendency - r.rdzw[level] * (flux_upper - flux_lower));
    if (p.use_inverse_area != 0) {
        r.output[gid] = numerator * (1.0f / r.target_density[density_at]);
    } else {
        r.output[gid] = numerator / r.target_density[density_at];
    }
}
