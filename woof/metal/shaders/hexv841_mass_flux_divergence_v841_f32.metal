#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_mass_flux_divergence_v841_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx_v841(int level, int entity, int count) { return level * count + entity; }
inline float cell_sign_v841(int cell, int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_mass_flux_divergence_v841_f32(
    constant Hexv841MassFluxDivergenceV841F32Resources &r [[buffer(0)]],
    constant Hexv841MassFluxDivergenceV841F32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *rho_u = r.rho_u;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *dv_edge = r.dv_edge;
    device const float *inv_area_cell = r.inv_area_cell;
    device float *mass_divergence = r.mass_divergence;
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
            value = mpas_add(value, mpas_mul(mpas_mul(
                cell_sign_v841(cell, edge, cells_on_edge), dv_edge[edge]),
                rho_u[lidx_v841(k, edge, nedges)]));
        }
        mass_divergence[lidx_v841(k, cell, ncells)] = mpas_mul(
            value, inv_area_cell[cell]);
    }
}
