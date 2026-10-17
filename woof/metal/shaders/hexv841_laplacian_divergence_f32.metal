#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_laplacian_divergence_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx(const int k, const int i, const int n) { return k * n + i; }
inline float cell_sign(const int cell, const int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_laplacian_divergence_f32(
    constant Hexv841LaplacianDivergenceF32Resources &r [[buffer(0)]],
    constant Hexv841LaplacianDivergenceF32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *delsq_u = r.delsq_u;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *dv_edge = r.dv_edge;
    device const float *area_cell = r.area_cell;
    device float *delsq_divergence = r.delsq_divergence;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges = p.max_edges;

    if (gid >= uint(nlev * ncells)) return;
    {
        const int element = int(gid);
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int edge = edges_on_cell[cell * max_edges + slot];
            value = mpas_add(value, mpas_mul(mpas_div(mpas_mul(
                cell_sign(cell, edge, cells_on_edge), dv_edge[edge]),
                area_cell[cell]), delsq_u[lidx(k, edge, nedges)]));
        }
        delsq_divergence[lidx(k, cell, ncells)] = value;
    }
}
