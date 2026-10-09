#include <metal_stdlib>
#include "hex_moist_coefficients_v841_fragment_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

static inline float hex_moist_q(
    constant HexMoistCoefficientsV841FragmentResources &r,
    constant HexMoistCoefficientsV841FragmentParams &p,
    uint scalar, uint level, int cell) {
    if (cell < 0 || cell >= p.ncells) return 0.0f;
    const uint index = scalar * uint(p.nlev * p.ncells)
        + level * uint(p.ncells) + uint(cell);
    return r.scalars[index];
}

static inline float hex_moist_qtot(
    constant HexMoistCoefficientsV841FragmentResources &r,
    constant HexMoistCoefficientsV841FragmentParams &p,
    uint level, int cell) {
    float total = 0.0f;
    for (int moist = 0; moist < p.nmoist; ++moist) {
        const int scalar = r.moist_indices[uint(moist)];
        total += hex_moist_q(r, p, uint(scalar), level, cell);
    }
    return total;
}

kernel void hex_moist_coefficients_v841_fragment(
    constant HexMoistCoefficientsV841FragmentResources &r [[buffer(0)]],
    constant HexMoistCoefficientsV841FragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint nlev = uint(p.nlev), ncells = uint(p.ncells);
    const uint cell_values = nlev * ncells;
    const uint edge_values = nlev * uint(p.nedges);
    if (tid < cell_values) {
        const uint level = tid / ncells;
        const uint cell = tid % ncells;
        const float total = hex_moist_qtot(r, p, level, int(cell));
        r.qtot[tid] = total;
        if (level == 0u) {
            r.cqw[tid] = 1.0f;
        } else {
            const float interface_total = 0.5f * (
                total + hex_moist_qtot(r, p, level - 1u, int(cell)));
            r.cqw[tid] = 1.0f / (1.0f + interface_total);
        }
    }
    if (tid < edge_values) {
        const uint level = tid / uint(p.nedges);
        const uint edge = tid % uint(p.nedges);
        const int cell0 = r.cells_on_edge[2u * edge];
        const int cell1 = r.cells_on_edge[2u * edge + 1u];
        float total = 0.0f;
        for (int moist = 0; moist < p.nmoist; ++moist) {
            const uint scalar = uint(r.moist_indices[uint(moist)]);
            const float q0 = hex_moist_q(r, p, scalar, level, cell0);
            const float q1 = hex_moist_q(r, p, scalar, level, cell1);
            total += 0.5f * (q0 + q1);
        }
        r.cqu[tid] = 1.0f / (1.0f + total);
    }
}
