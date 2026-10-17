#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_density_tendency_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx(const int k, const int i, const int n) { return k * n + i; }
inline float cell_sign(const int cell, const int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_density_tendency_f32(
    constant Hexv841DensityTendencyF32Resources &r [[buffer(0)]],
    constant Hexv841DensityTendencyF32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *mass_divergence = r.mass_divergence;
    device const float *rho_w = r.rho_w;
    device const float *rdzw = r.rdzw;
    device const float *physics_tendency = r.physics_tendency;
    device float *tendency = r.tendency;
    const int has_physics = p.has_physics;
    const int nlev = p.nlev;
    const int ncells = p.ncells;

    const int cell = int(gid);
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int tid = lidx(k, cell, ncells);
        float value = mpas_sub(mpas_sub(0.0f, mass_divergence[tid]),
            mpas_mul(rdzw[k], mpas_sub(
                rho_w[lidx(k + 1, cell, ncells)],
                rho_w[lidx(k, cell, ncells)])));
        if (has_physics) value = mpas_add(value, physics_tendency[tid]);
        tendency[tid] = value;
    }
}
