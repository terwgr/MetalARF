#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_cell_diagnostics_v841_f32_bindings.metalh"
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


kernel void hexv841_cell_diagnostics_v841_f32(
    constant Hexv841CellDiagnosticsV841F32Resources &r [[buffer(0)]],
    constant Hexv841CellDiagnosticsV841F32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *normal_velocity = r.normal_velocity;
    device const float *ke_edge = r.ke_edge;
    device const float *ke_vertex = r.ke_vertex;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const int *vertices_on_cell = r.vertices_on_cell;
    device const int *cells_on_vertex = r.cells_on_vertex;
    device const float *dv_edge = r.dv_edge;
    device const float *inv_area_cell = r.inv_area_cell;
    device const float *kite_areas = r.kite_areas;
    device float *divergence = r.divergence;
    device float *kinetic_energy = r.kinetic_energy;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int nvertices = p.nvertices;
    const int max_edges = p.max_edges;
    const int vertex_degree = p.vertex_degree;

    if (gid >= uint(nlev * ncells)) return;
    {
        const int element = int(gid);
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        const float inv_area = inv_area_cell[cell];
        const float ke_fact = 0.625f;
        float div = 0.0f;
        float ke = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            div = mpas_add(div, mpas_mul(mpas_mul(
                cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]),
                normal_velocity[lidx_v841(k, edge, nedges)]));
            ke = mpas_add(ke, mpas_mul(
                0.25f, ke_edge[lidx_v841(k, edge, nedges)]));
        }
        div = mpas_mul(div, inv_area);
        ke = mpas_mul(ke, inv_area);
        ke = mpas_mul(ke, ke_fact);
        for (int slot = 0; slot < count; ++slot) {
            const int vertex_ = vertices_on_cell[cell * max_edges + slot];
            const int kite = kite_slot_v841(
                vertex_, cell, cells_on_vertex, vertex_degree);
            if (kite >= 0) {
                float correction = mpas_mul(mpas_sub(1.0f, ke_fact),
                    kite_areas[vertex_ * vertex_degree + kite]);
                correction = mpas_mul(correction,
                    ke_vertex[lidx_v841(k, vertex_, nvertices)]);
                correction = mpas_mul(correction, inv_area);
                ke = mpas_add(ke, correction);
            }
        }
        const int index = lidx_v841(k, cell, ncells);
        divergence[index] = div;
        kinetic_energy[index] = ke;
    }
}
