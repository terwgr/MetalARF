#include <metal_stdlib>
#include "hex_pressure_gradient_euler_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// Terrain-coordinate Euler pressure-gradient tendency, one lane per
// (vertical level, MPAS edge) pair.
kernel void hex_pressure_gradient_euler(
        constant HexPressureGradientEulerResources &r [[buffer(0)]],
        constant HexPressureGradientEulerParams &q [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong count = ulong(q.nlev) * ulong(q.nedges);
    if (q.nlev <= 0 || q.ncells <= 0 || q.nedges <= 0 || ulong(gid) >= count)
        return;
    const int level = int(gid / uint(q.nedges));
    const int edge = int(gid % uint(q.nedges));
    const int cell0 = r.cells_on_edge[2 * edge];
    const int cell1 = r.cells_on_edge[2 * edge + 1];
    if (cell0 < 0 || cell1 < 0 || cell0 >= q.ncells || cell1 >= q.ncells)
        return;
    const ulong at0 = ulong(level) * ulong(q.ncells) + ulong(cell0);
    const ulong at1 = ulong(level) * ulong(q.ncells) + ulong(cell1);
    volatile float pressure_difference = r.pressure_perturbation[at1]
                                      - r.pressure_perturbation[at0];
    volatile float scaled_pressure_difference = pressure_difference
                                             * r.inv_dc_edge[edge];
    const float mean_zz = 0.5f * (r.zz[at1] + r.zz[at0]);
    volatile float terrain_gradient = scaled_pressure_difference / mean_zz;
    const float vertical_pressure_gradient = 0.5f * r.zxu[gid]
                                          * (r.dpdz[at0] + r.dpdz[at1]);
    r.tendency[gid] = -r.cqu[gid]
                    * (terrain_gradient - vertical_pressure_gradient);
}
