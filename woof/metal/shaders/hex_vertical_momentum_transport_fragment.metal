// needs integration: Hex vertical normal-momentum transport.
#include <metal_stdlib>
#include "hex_vertical_momentum_transport_fragment_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float hex_flux3(float qim2, float qim1, float qi, float qip1,
                              float velocity, float coefficient) {
    const float flux4 = velocity *
        (7.0f * (qi + qim1) - (qip1 + qim2)) / 12.0f;
    return flux4 + coefficient * abs(velocity) *
        ((qip1 - qim2) - 3.0f * (qi - qim1)) / 12.0f;
}

static inline float hex_wduz(
    constant HexVerticalMomentumTransportFragmentResources &r,
    constant HexVerticalMomentumTransportFragmentParams &p,
    uint level, uint edge) {
    const uint nlev = uint(p.nlev), ncells = uint(p.ncells);
    if (level == 0u || level >= nlev) return 0.0f;
    const int c0 = r.cells_on_edge[2u * edge];
    const int c1 = r.cells_on_edge[2u * edge + 1u];
    const float rho_edge = 0.5f *
        (r.rho_w[level * ncells + uint(c0)] +
         r.rho_w[level * ncells + uint(c1)]);
    if (level == 1u || level + 1u == nlev) {
        return rho_edge *
            (r.fzm[level] * r.normal_velocity[level * uint(p.nedges) + edge]
             + r.fzp[level] * r.normal_velocity[(level - 1u) * uint(p.nedges) + edge]);
    }
    const uint stride = uint(p.nedges);
    return hex_flux3(
        r.normal_velocity[(level - 2u) * stride + edge],
        r.normal_velocity[(level - 1u) * stride + edge],
        r.normal_velocity[level * stride + edge],
        r.normal_velocity[(level + 1u) * stride + edge], rho_edge,
        p.coefficient);
}

kernel void hex_vertical_momentum_transport_fragment(
    constant HexVerticalMomentumTransportFragmentResources &r [[buffer(0)]],
    constant HexVerticalMomentumTransportFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.nedges);
    if (tid >= count) return;
    const uint level = tid / uint(p.nedges);
    const uint edge = tid % uint(p.nedges);
    r.tendency[tid] = -r.rdzw[level] *
        (hex_wduz(r, p, level + 1u, edge) - hex_wduz(r, p, level, edge));
}
