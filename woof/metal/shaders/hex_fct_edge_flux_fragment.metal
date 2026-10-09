#include <metal_stdlib>
#include "../generated/hex_fct_edge_flux_fragment_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

kernel void hex_fct_edge_flux_fragment(
    constant HexFctEdgeFluxFragmentResources &r [[buffer(0)]],
    constant HexFctEdgeFluxFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.nedges);
    if (tid >= count) return;
    const uint level = tid / uint(p.nedges);
    const uint edge = tid % uint(p.nedges);
    const uint storage = uint(p.storage_cells);
    const uint max_adv = uint(p.max_adv);
    const float velocity = r.horizontal_velocity[tid];
    const float sign = velocity < 0.0f ? -1.0f : 1.0f;
    const float coefficient = p.order == 3 ? p.coefficient : 0.0f;
    const int n_adv = r.n_adv_cells_for_edge[edge];
    const uint out_index = tid;

    float value = 0.0f;
    if (p.order == 2) {
        const uint c0 = uint(r.cells_on_edge[2u * edge]);
        const uint c1 = uint(r.cells_on_edge[2u * edge + 1u]);
        value = r.dv_edge[edge] * 0.5f * (
            r.scalar_stage[uint(p.scalar) * uint(p.nlev) * storage + level * storage + c0]
            + r.scalar_stage[uint(p.scalar) * uint(p.nlev) * storage + level * storage + c1]);
        value *= velocity;
        r.high_flux[out_index] = value;
        return;
    }

    if (p.monotonic_source_flux != 0 && n_adv == 10) {
        const uint cell0 = uint(r.adv_cells_for_edge[edge * max_adv]);
        const uint stage_base = uint(p.scalar) * uint(p.nlev) * storage
            + level * storage;
        value = (r.adv_coefs[edge * max_adv]
            + coefficient * sign * r.adv_coefs_3rd[edge * max_adv])
            * r.scalar_stage[stage_base + cell0];
        for (int slot = 1; slot < 10; ++slot) {
            const uint offset = edge * max_adv + uint(slot);
            const uint cell = uint(r.adv_cells_for_edge[offset]);
            value += (r.adv_coefs[offset]
                + coefficient * sign * r.adv_coefs_3rd[offset])
                * r.scalar_stage[stage_base + cell];
        }
        value *= velocity;
    } else {
        const uint stage_base = uint(p.scalar) * uint(p.nlev) * storage
            + level * storage;
        for (int slot = 0; slot < n_adv; ++slot) {
            const uint offset = edge * max_adv + uint(slot);
            const uint cell = uint(r.adv_cells_for_edge[offset]);
            float weight = r.adv_coefs[offset]
                + coefficient * sign * r.adv_coefs_3rd[offset];
            if (p.monotonic_source_flux != 0) weight *= velocity;
            value += weight * r.scalar_stage[stage_base + cell];
        }
        if (p.monotonic_source_flux == 0) value *= velocity;
    }
    r.high_flux[out_index] = value;
}
