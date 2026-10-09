// needs integration: Hex absolute-vorticity and APVM diagnostics.
#include <metal_stdlib>
#include "hex_pv_diagnostics_fragment_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void hex_pv_vertex_fragment(
    constant HexPvVertexFragmentResources &r [[buffer(0)]],
    constant HexPvVertexFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.nvertices);
    if (tid >= count) return;
    r.pv_vertex[tid] = r.f_vertex[tid % uint(p.nvertices)] + r.vorticity[tid];
}

kernel void hex_pv_cell_fragment(
    constant HexPvCellFragmentResources &r [[buffer(0)]],
    constant HexPvCellFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.ncells);
    if (tid >= count) return;
    const uint level = tid / uint(p.ncells);
    const uint cell = tid % uint(p.ncells);
    if (p.apvm_upwinding <= 0.0f) { r.pv_cell[tid] = 0.0f; return; }
    float sum = 0.0f;
    for (int slot = 0; slot < r.n_edges_on_cell[cell]; ++slot) {
        const uint offset = cell * uint(p.max_cell_edges) + uint(slot);
        const int vertex_id = r.vertices_on_cell[offset];
        const int kite = r.kite_index[offset];
        const float contribution =
            r.kite_areas[uint(vertex_id) * uint(p.max_vertex_edges) + uint(kite)]
            * r.pv_vertex[level * uint(p.nvertices) + uint(vertex_id)];
        sum += p.use_inverse_area != 0
            ? contribution * r.inv_area_cell[cell]
            : contribution / r.area_cell[cell];
    }
    r.pv_cell[tid] = sum;
}

kernel void hex_pv_edge_fragment(
    constant HexPvEdgeFragmentResources &r [[buffer(0)]],
    constant HexPvEdgeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.nedges);
    if (tid >= count) return;
    const uint level = tid / uint(p.nedges);
    const uint edge = tid % uint(p.nedges);
    const int v0 = r.vertices_on_edge[2u * edge];
    const int v1 = r.vertices_on_edge[2u * edge + 1u];
    float value = 0.5f * (r.pv_vertex[level * uint(p.nvertices) + uint(v0)]
                        + r.pv_vertex[level * uint(p.nvertices) + uint(v1)]);
    float grad_t = 0.0f;
    float grad_n = 0.0f;
    if (p.apvm_upwinding > 0.0f) {
        const int c0 = r.cells_on_edge[2u * edge];
        const int c1 = r.cells_on_edge[2u * edge + 1u];
        const float delta_t = r.pv_vertex[level * uint(p.nvertices) + uint(v1)]
                            - r.pv_vertex[level * uint(p.nvertices) + uint(v0)];
        const float delta_n = r.pv_cell[level * uint(p.ncells) + uint(c1)]
                            - r.pv_cell[level * uint(p.ncells) + uint(c0)];
        grad_t = p.use_inverse_dv != 0 ? delta_t * r.inv_dv_edge[edge]
                                      : delta_t / r.dv_edge[edge];
        grad_n = p.use_inverse_dc != 0 ? delta_n * r.inv_dc_edge[edge]
                                      : delta_n / r.dc_edge[edge];
        const float scale = p.apvm_upwinding * p.dt;
        value -= scale * (r.tangential_velocity[tid] * grad_t
                        + r.normal_velocity[tid] * grad_n);
    }
    r.pv_edge[tid] = value;
    r.grad_pv_tangential[tid] = grad_t;
    r.grad_pv_normal[tid] = grad_n;
}
