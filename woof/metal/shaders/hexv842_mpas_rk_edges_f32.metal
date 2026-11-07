#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_rk_edges_f32_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv842_mpas_rk_edges_f32(
    constant Hexv842MpasRkEdgesF32Resources &r [[buffer(0)]],
    constant Hexv842MpasRkEdgesF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int i = int(tid);
    if (i >= p.nlev * p.nedges) return;
    if (p.first) {
        if (p.pbl) r.tend_u_phys[i] = r.pbl_edge[i];
        if (p.cu_momentum) r.tend_u_phys[i] = mpas_add(r.tend_u_phys[i], r.cu_edge[i]);
    }
    if (i % p.nedges >= p.nsolve) return;
    float u = 0.0f;
    if (p.pbl) u = mpas_add(u, mpas_mul(r.pbl_edge[i], r.mass_edge[i]));
    if (p.cu_momentum) u = mpas_add(u, mpas_mul(r.cu_edge[i], r.mass_edge[i]));
    r.out_u[i] = u;
}
