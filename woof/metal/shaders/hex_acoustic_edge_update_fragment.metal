#include <metal_stdlib>
#include "hex_acoustic_edge_update_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// First-sweep mass-weighted normal momentum and its acoustic average. One
// lane owns an MPAS edge and visits vertical levels in source order.
kernel void hex_acoustic_edge_update(
        constant HexAcousticEdgeUpdateResources &r [[buffer(0)]],
        constant HexAcousticEdgeUpdateParams &q [[buffer(1)]],
        uint edge [[thread_position_in_grid]]) {
    if (q.nlev <= 0 || q.ncells <= 0 || q.nedges <= 0
            || edge >= uint(q.nedges)) return;
    const int cell0 = r.cells_on_edge[2 * edge];
    const int cell1 = r.cells_on_edge[2 * edge + 1];
    if (cell0 < 0 || cell1 < 0 || cell0 >= q.ncells || cell1 >= q.ncells)
        return;

    for (int level = 0; level < q.nlev; ++level) {
        const ulong edge_at = ulong(level) * ulong(q.nedges) + ulong(edge);
        if (q.small_step == 1) {
            r.ru_p[edge_at] = q.dts * r.tend_ru[edge_at];
            r.ru_avg[edge_at] = r.ru_p[edge_at];
            continue;
        }
        const ulong p0 = ulong(level) * ulong(q.ncells) + ulong(cell0);
        const ulong p1 = ulong(level) * ulong(q.ncells) + ulong(cell1);
        volatile float pressure_difference = r.rtheta_pp[p1] - r.rtheta_pp[p0];
        volatile float pressure_gradient = pressure_difference / r.dc_edge[edge];
        const float mean_zz = 0.5f * (r.zz[p0] + r.zz[p1]);
        pressure_gradient = pressure_gradient / mean_zz;
        pressure_gradient = r.cqu[edge_at] * 0.5f * q.c2
                          * (r.exner[p0] + r.exner[p1]) * pressure_gradient;
        pressure_gradient = pressure_gradient + 0.5f * q.gravity
                          * r.zxu[edge_at] * (r.rho_pp[p0] + r.rho_pp[p1]);
        r.ru_p[edge_at] += q.dts * (r.tend_ru[edge_at]
                           - (1.0f - r.specified_zone_edge[edge])
                           * pressure_gradient);
        r.ru_avg[edge_at] += r.ru_p[edge_at];
    }
}
