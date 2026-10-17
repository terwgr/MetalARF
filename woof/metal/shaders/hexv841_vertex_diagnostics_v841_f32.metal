#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_vertex_diagnostics_v841_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx_v841(int level, int entity, int count) { return level * count + entity; }
inline float cell_sign_v841(int cell, int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}

inline float vertex_sign_v841(int vertex_, int edge, device const int *vertices_on_edge) {
    return vertices_on_edge[2 * edge] == vertex_ ? -1.0f : 1.0f;
}


kernel void hexv841_vertex_diagnostics_v841_f32(
    constant Hexv841VertexDiagnosticsV841F32Resources &r [[buffer(0)]],
    constant Hexv841VertexDiagnosticsV841F32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *normal_velocity = r.normal_velocity;
    device const float *ke_edge = r.ke_edge;
    device const int *edges_on_vertex = r.edges_on_vertex;
    device const int *vertices_on_edge = r.vertices_on_edge;
    device const float *dc_edge = r.dc_edge;
    device const float *inv_area_triangle = r.inv_area_triangle;
    device const float *f_vertex = r.f_vertex;
    device float *vorticity = r.vorticity;
    device float *ke_vertex = r.ke_vertex;
    device float *pv_vertex = r.pv_vertex;
    const int nlev = p.nlev;
    const int nedges = p.nedges;
    const int nvertices = p.nvertices;
    const int vertex_degree = p.vertex_degree;

    if (gid >= uint(nlev * nvertices)) return;
    {
        const int element = int(gid);
        const int vertex_ = element % nvertices;
        const int k = element / nvertices;
        float vort = 0.0f;
        float ke = 0.0f;
        for (int slot = 0; slot < vertex_degree; ++slot) {
            const int edge = edges_on_vertex[vertex_ * vertex_degree + slot];
            if (edge < 0) continue;
            vort = mpas_add(vort, mpas_mul(mpas_mul(
                vertex_sign_v841(vertex_, edge, vertices_on_edge), dc_edge[edge]),
                normal_velocity[lidx_v841(k, edge, nedges)]));
            ke = mpas_add(ke, ke_edge[lidx_v841(k, edge, nedges)]);
        }
        vort = mpas_mul(vort, inv_area_triangle[vertex_]);
        const float r = mpas_mul(0.25f, inv_area_triangle[vertex_]);
        ke = mpas_mul(ke, r);
        const int index = lidx_v841(k, vertex_, nvertices);
        vorticity[index] = vort;
        ke_vertex[index] = ke;
        pv_vertex[index] = mpas_add(f_vertex[vertex_], vort);
    }
}
