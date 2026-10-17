#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_pv_apvm_v841_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx_v841(int level, int entity, int count) { return level * count + entity; }
inline float cell_sign_v841(int cell, int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_pv_apvm_v841_f32(
    constant Hexv841PvApvmV841F32Resources &r [[buffer(0)]],
    constant Hexv841PvApvmV841F32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *normal_velocity = r.normal_velocity;
    device const float *tangential_velocity = r.tangential_velocity;
    device const float *pv_vertex = r.pv_vertex;
    device const float *pv_cell = r.pv_cell;
    device const int *vertices_on_edge = r.vertices_on_edge;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *inv_dc_edge = r.inv_dc_edge;
    device const float *inv_dv_edge = r.inv_dv_edge;
    device float *pv_edge = r.pv_edge;
    device float *grad_pv_normal = r.grad_pv_normal;
    device float *grad_pv_tangential = r.grad_pv_tangential;
    const float scale = p.scale;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int nvertices = p.nvertices;

    if (gid >= uint(nlev * nedges)) return;
    {
        const int element = int(gid);
        const int edge = element % nedges;
        const int k = element / nedges;
        const int vertex0 = vertices_on_edge[2 * edge];
        const int vertex1 = vertices_on_edge[2 * edge + 1];
        const int cell0 = cells_on_edge[2 * edge];
        const int cell1 = cells_on_edge[2 * edge + 1];
        const int index = lidx_v841(k, edge, nedges);
        const float grad_t = mpas_mul(mpas_sub(
            pv_vertex[lidx_v841(k, vertex1, nvertices)],
            pv_vertex[lidx_v841(k, vertex0, nvertices)]), inv_dv_edge[edge]);
        const float grad_n = mpas_mul(mpas_sub(
            pv_cell[lidx_v841(k, cell1, ncells)],
            pv_cell[lidx_v841(k, cell0, ncells)]), inv_dc_edge[edge]);
        grad_pv_tangential[index] = grad_t;
        grad_pv_normal[index] = grad_n;
        pv_edge[index] = mpas_sub(pv_edge[index], mpas_mul(scale, mpas_add(
            mpas_mul(tangential_velocity[index], grad_t),
            mpas_mul(normal_velocity[index], grad_n))));
    }
}
