// needs integration: Hex C-grid vorticity, divergence and kinetic energy.
#include <metal_stdlib>
#include "hex_kinematic_diagnostics_fragment_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void hex_vertex_kinematics_fragment(
    constant HexVertexKinematicsFragmentResources &r [[buffer(0)]],
    constant HexVertexKinematicsFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.nvertices);
    if (tid >= count) return;
    const uint level = tid / uint(p.nvertices);
    const uint vertex_id = tid % uint(p.nvertices);
    float circulation = 0.0f;
    float ke_sum = 0.0f;
    for (int slot = 0; slot < p.max_vertex_edges; ++slot) {
        const int edge = r.edges_on_vertex[vertex_id * uint(p.max_vertex_edges) + uint(slot)];
        if (edge < 0) continue;
        const float u = r.normal_velocity[level * uint(p.nedges) + uint(edge)];
        circulation += r.edge_signs[vertex_id * uint(p.max_vertex_edges) + uint(slot)]
            * r.dc_edge[uint(edge)] * u;
        ke_sum += r.dc_edge[uint(edge)] * r.dv_edge[uint(edge)] * u * u;
    }
    const float inv_area = p.use_inverse != 0
        ? r.inv_area_triangle[vertex_id] : (1.0f / r.area_triangle[vertex_id]);
    r.vorticity[tid] = p.use_inverse != 0
        ? circulation * inv_area : circulation / r.area_triangle[vertex_id];
    r.ke_vertex[tid] = p.use_inverse != 0
        ? ke_sum * (0.25f * inv_area)
        : (0.25f * ke_sum) / r.area_triangle[vertex_id];
}

kernel void hex_cell_kinematics_fragment(
    constant HexCellKinematicsFragmentResources &r [[buffer(0)]],
    constant HexCellKinematicsFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.ncells);
    if (tid >= count) return;
    const uint level = tid / uint(p.ncells);
    const uint cell = tid % uint(p.ncells);
    float flux = 0.0f;
    float ke_sum = 0.0f;
    for (int slot = 0; slot < r.n_edges_on_cell[cell]; ++slot) {
        const int edge = r.edges_on_cell[cell * uint(p.max_cell_edges) + uint(slot)];
        const float u = r.normal_velocity[level * uint(p.nedges) + uint(edge)];
        flux += r.cell_signs[cell * uint(p.max_cell_edges) + uint(slot)]
            * r.dv_edge[uint(edge)] * u;
        const float edge_ke = r.dc_edge[uint(edge)] * r.dv_edge[uint(edge)] * u * u;
        ke_sum += 0.25f * edge_ke;
    }
    r.divergence[tid] = p.use_inverse != 0
        ? flux * r.inv_area_cell[cell] : flux / r.area_cell[cell];
    const float ke_area = p.use_inverse != 0
        ? ke_sum * r.inv_area_cell[cell] : ke_sum / r.area_cell[cell];
    r.ke_raw[tid] = ke_area * 0.625f;
}

kernel void hex_cell_ke_finish_fragment(
    constant HexCellKeFinishFragmentResources &r [[buffer(0)]],
    constant HexCellKeFinishFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.ncells);
    if (tid >= count) return;
    const uint level = tid / uint(p.ncells);
    const uint cell = tid % uint(p.ncells);
    float correction = 0.0f;
    for (int slot = 0; slot < r.n_edges_on_cell[cell]; ++slot) {
        const uint offset = cell * uint(p.max_cell_edges) + uint(slot);
        const int vertex_id = r.vertices_on_cell[offset];
        const int kite = r.kite_index[offset];
        correction += 0.375f * r.kite_areas[uint(vertex_id) * uint(p.max_vertex_edges) + uint(kite)]
            * r.ke_vertex[level * uint(p.nvertices) + uint(vertex_id)];
    }
    const float area_correction = p.use_inverse != 0
        ? correction * r.inv_area_cell[cell] : correction / r.area_cell[cell];
    r.kinetic_energy[tid] = r.ke_raw[tid] + area_correction;
}
