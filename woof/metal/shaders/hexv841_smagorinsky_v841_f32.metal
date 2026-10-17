#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_smagorinsky_v841_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx_v841(int level, int entity, int count) { return level * count + entity; }
inline float cell_sign_v841(int cell, int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_smagorinsky_v841_f32(
    constant Hexv841SmagorinskyV841F32Resources &r [[buffer(0)]],
    constant Hexv841SmagorinskyV841F32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *normal_velocity = r.normal_velocity;
    device const float *tangential_velocity = r.tangential_velocity;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const float *coef_c2 = r.coef_c2;
    device const float *coef_s2 = r.coef_s2;
    device const float *coef_cs = r.coef_cs;
    device float *kdiff = r.kdiff;
    const float strain_scale = p.strain_scale;
    const float ceiling = p.ceiling;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges = p.max_edges;

    // mpas_atm_dissipation_models.F:119-204 (smagorinsky_2d): full
    // velocity-gradient tensor from the precomputed v8.4.1 deformation
    // weights, D11 = 2 du/dx, D22 = 2 dv/dy, D12 = du/dy + dv/dx,
    // kdiff = (c_s*len)**2 * sqrt(0.25*(D11-D22)**2 + D12**2) capped at
    // (0.01*len**2)*invDt.
    if (gid >= uint(nlev * ncells)) return;
    {
        const int element = int(gid);
        const int cell = element % ncells;
        const int k = element / ncells;
        const int count = n_edges_on_cell[cell];
        float dudx = 0.0f;
        float dudy = 0.0f;
        float dvdx = 0.0f;
        float dvdy = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int offset = cell * max_edges + slot;
            const int edge = edges_on_cell[offset];
            const float u = normal_velocity[lidx_v841(k, edge, nedges)];
            const float v = tangential_velocity[lidx_v841(k, edge, nedges)];
            const float c2 = coef_c2[offset];
            const float s2 = coef_s2[offset];
            const float cs = coef_cs[offset];
            dudx = mpas_add(dudx,
                mpas_sub(mpas_mul(c2, u), mpas_mul(cs, v)));
            dudy = mpas_add(dudy,
                mpas_sub(mpas_mul(cs, u), mpas_mul(s2, v)));
            dvdx = mpas_add(dvdx,
                mpas_add(mpas_mul(cs, u), mpas_mul(c2, v)));
            dvdy = mpas_add(dvdy,
                mpas_add(mpas_mul(s2, u), mpas_mul(cs, v)));
        }
        const float d_11 = mpas_mul(2.0f, dudx);
        const float d_22 = mpas_mul(2.0f, dvdy);
        const float d_12 = mpas_add(dudy, dvdx);
        const float diff = mpas_sub(d_11, d_22);
        const float strain = mpas_sqrt(mpas_add(
            mpas_mul(0.25f, mpas_mul(diff, diff)),
            mpas_mul(d_12, d_12)));
        kdiff[lidx_v841(k, cell, ncells)] = mpas_min(
            mpas_mul(strain_scale, strain), ceiling);
    }
}
