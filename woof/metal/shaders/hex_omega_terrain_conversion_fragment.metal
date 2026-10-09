// needs integration: Hex terrain-slope conversion from physical-w to omega.
#include <metal_stdlib>
#include "hex_omega_terrain_conversion_fragment_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void hex_omega_terrain_conversion_fragment(
    constant HexOmegaTerrainConversionFragmentResources &r [[buffer(0)]],
    constant HexOmegaTerrainConversionFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total = uint(p.nlev + 1) * uint(p.ncells);
    if (tid >= total) return;
    const uint level = tid / uint(p.ncells);
    const uint cell = tid % uint(p.ncells);
    float value = r.w_tendency[tid];
    if (level > 0u && level < uint(p.nlev)
        && r.boundary_mask[cell] <= p.relaxation_zone) {
        for (uint slot = 0u; slot < uint(r.edge_counts[cell]); ++slot) {
            const uint edge = uint(r.edges_on_cell[cell * uint(p.max_edges) + slot]);
            const float u = r.u_tendency[level * uint(p.nedges) + edge];
            const float flux = r.edge_signs[cell * uint(p.max_edges) + slot] *
                (r.fzm[level] * u + r.fzp[level] *
                 r.u_tendency[(level - 1u) * uint(p.nedges) + edge]);
            const float upwind = copysign(1.0f, u);
            value = value -
                (r.zb_cell[(level * uint(p.ncells) + cell) * uint(p.max_edges) + slot]
                 + upwind * r.zb3_cell[(level * uint(p.ncells) + cell) * uint(p.max_edges) + slot])
                * flux;
        }
        value = (r.fzm[level] * r.zz[level * uint(p.ncells) + cell]
                 + r.fzp[level] * r.zz[(level - 1u) * uint(p.ncells) + cell]) * value;
    }
    r.omega_tendency[tid] = value;
}
