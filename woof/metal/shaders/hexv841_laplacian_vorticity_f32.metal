#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_laplacian_vorticity_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx(const int k, const int i, const int n) { return k * n + i; }
inline float cell_sign(const int cell, const int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}

inline float vertex_sign(const int vertex_, const int edge, device const int *vertices_on_edge) {
    return vertices_on_edge[2 * edge + 1] == vertex_ ? 1.0f : -1.0f;
}


kernel void hexv841_laplacian_vorticity_f32(
    constant Hexv841LaplacianVorticityF32Resources &r [[buffer(0)]],
    constant Hexv841LaplacianVorticityF32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *delsq_u = r.delsq_u;
    device const int *edges_on_vertex = r.edges_on_vertex;
    device const int *vertices_on_edge = r.vertices_on_edge;
    device const float *dc_edge = r.dc_edge;
    device const float *area_triangle = r.area_triangle;
    device float *delsq_vorticity = r.delsq_vorticity;
    const int nlev = p.nlev;
    const int nedges = p.nedges;
    const int nvertices = p.nvertices;
    const int vertex_degree = p.vertex_degree;

    const int vertex_ = int(gid);
    if (vertex_ >= nvertices) return;
    for (int k = 0; k < nlev; ++k) {
        float value = 0.0f;
        for (int slot = 0; slot < vertex_degree; ++slot) {
            const int edge = edges_on_vertex[vertex_ * vertex_degree + slot];
            if (edge < 0) continue;
            value = mpas_add(value, mpas_mul(mpas_div(mpas_mul(
                vertex_sign(vertex_, edge, vertices_on_edge), dc_edge[edge]),
                area_triangle[vertex_]), delsq_u[lidx(k, edge, nedges)]));
        }
        delsq_vorticity[lidx(k, vertex_, nvertices)] = value;
    }
}
