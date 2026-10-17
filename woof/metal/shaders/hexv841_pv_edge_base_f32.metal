#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_pv_edge_base_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx(const int k, const int i, const int n) { return k * n + i; }
inline float cell_sign(const int cell, const int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_pv_edge_base_f32(
    constant Hexv841PvEdgeBaseF32Resources &r [[buffer(0)]],
    constant Hexv841PvEdgeBaseF32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *pv_vertex = r.pv_vertex;
    device const int *vertices_on_edge = r.vertices_on_edge;
    device float *pv_edge = r.pv_edge;
    const int nlev = p.nlev;
    const int nedges = p.nedges;
    const int nvertices = p.nvertices;

    const int edge = int(gid);
    if (edge >= nedges) return;
    const int vertex0 = vertices_on_edge[2 * edge];
    const int vertex1 = vertices_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        pv_edge[lidx(k, edge, nedges)] = mpas_mul(0.5f, mpas_add(
            pv_vertex[lidx(k, vertex0, nvertices)],
            pv_vertex[lidx(k, vertex1, nvertices)]));
    }
}
