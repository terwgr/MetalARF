#include <metal_stdlib>
#include "hex_ieee32.metalh"
#include "../generated/hexv841_tangential_velocity_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

inline int lidx(const int k, const int i, const int n) { return k * n + i; }
inline float cell_sign(const int cell, const int edge, device const int *cells_on_edge) {
    return cells_on_edge[2 * edge] == cell ? 1.0f : -1.0f;
}


kernel void hexv841_tangential_velocity_f32(
    constant Hexv841TangentialVelocityF32Resources &r [[buffer(0)]],
    constant Hexv841TangentialVelocityF32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const float *normal_velocity = r.normal_velocity;
    device const int *edges_on_edge = r.edges_on_edge;
    device const int *n_edges_on_edge = r.n_edges_on_edge;
    device const float *weights_on_edge = r.weights_on_edge;
    device float *tangential_velocity = r.tangential_velocity;
    const int nlev = p.nlev;
    const int nedges = p.nedges;
    const int max_edges2 = p.max_edges2;

    if (gid >= uint(nlev * nedges)) return;
    {
        const int element = int(gid);
        const int edge = element % nedges;
        const int k = element / nedges;
        const int count = n_edges_on_edge[edge];
        float value = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            const int offset = edge * max_edges2 + slot;
            const int neighbor = edges_on_edge[offset];
            value = mpas_add(value, mpas_mul(weights_on_edge[offset],
                normal_velocity[lidx(k, neighbor, nedges)]));
        }
        tangential_velocity[lidx(k, edge, nedges)] = value;
    }
}
