#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_pressure_gradient_v841_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx_v841(int level, int entity, int count) { return level * count + entity; }
inline float cell_sign_v841(int cell, int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_pressure_gradient_v841_f32(
    constant Hexv841PressureGradientV841F32Resources &r [[buffer(0)]],
    constant Hexv841PressureGradientV841F32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *pressure_p = r.pressure_p;
    device const float *dpdz = r.dpdz;
    device const float *cqu = r.cqu;
    device const float *zz = r.zz;
    device const float *zxu = r.zxu;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *inv_dc_edge = r.inv_dc_edge;
    device float *tendency = r.tendency;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;

    const int edge = int(gid);
    if (edge >= nedges) return;
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        const int index = lidx_v841(k, edge, nedges);
        float normal = mpas_mul(mpas_sub(
            pressure_p[lidx_v841(k, cell1, ncells)],
            pressure_p[lidx_v841(k, cell0, ncells)]), inv_dc_edge[edge]);
        normal = mpas_div(normal, mpas_mul(0.5f, mpas_add(
            zz[lidx_v841(k, cell1, ncells)],
            zz[lidx_v841(k, cell0, ncells)])));
        const float terrain = mpas_mul(mpas_mul(0.5f, zxu[index]), mpas_add(
            dpdz[lidx_v841(k, cell0, ncells)],
            dpdz[lidx_v841(k, cell1, ncells)]));
        tendency[index] = mpas_mul(mpas_sub(0.0f, cqu[index]),
            mpas_sub(normal, terrain));
    }
}
