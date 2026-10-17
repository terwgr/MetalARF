#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_pv_cell_v841_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx_v841(int level, int entity, int count) { return level * count + entity; }
inline float cell_sign_v841(int cell, int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}

inline int kite_slot_v841(int vertex_, int cell, device const int *cells_on_vertex, int vertex_degree) {
    for (int slot = 0; slot < vertex_degree; ++slot) {
        if (cells_on_vertex[vertex_ * vertex_degree + slot] == cell) return slot;
    }
    return -1;
}


kernel void hexv841_pv_cell_v841_f32(
    constant Hexv841PvCellV841F32Resources &r [[buffer(0)]],
    constant Hexv841PvCellV841F32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *pv_vertex = r.pv_vertex;
    device const int *vertices_on_cell = r.vertices_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_vertex = r.cells_on_vertex;
    device const float *kite_areas = r.kite_areas;
    device const float *inv_area_cell = r.inv_area_cell;
    device float *pv_cell = r.pv_cell;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nvertices = p.nvertices;
    const int max_edges = p.max_edges;
    const int vertex_degree = p.vertex_degree;

    if (gid >= uint(nlev * ncells)) return;
    {
        const int element = int(gid);
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int vertex_ = vertices_on_cell[cell * max_edges + slot];
            const int kite = kite_slot_v841(
                vertex_, cell, cells_on_vertex, vertex_degree);
            if (kite >= 0) {
                float contribution = mpas_mul(
                    kite_areas[vertex_ * vertex_degree + kite],
                    pv_vertex[lidx_v841(k, vertex_, nvertices)]);
                contribution = mpas_mul(contribution, inv_area_cell[cell]);
                value = mpas_add(value, contribution);
            }
        }
        pv_cell[lidx_v841(k, cell, ncells)] = value;
    }
}
