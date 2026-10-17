// Hex v8.4.1 Metal transcription of CUDA kernel project_edges_v841_f32 (woof/hex/cuda_physics_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_project_edges_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)


kernel void hexv841_project_edges_v841_f32(
    constant Hexv841ProjectEdgesV841F32Resources &r [[buffer(0)]],
    constant Hexv841ProjectEdgesV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    device const int *cells = r.cells;
    device const float *east = r.east;
    device const float *north = r.north;
    device const float *normal = r.normal;
    device const float *du = r.du;
    device const float *dv = r.dv;
    device const float *rho_edge = r.rho_edge;
    device float *out_ru = r.out_ru;
    device int *invalid = r.invalid;

    const int edge = int(tid);
    if (edge >= nedges) return;
    const int c1 = cells[2 * edge];
    const int c2 = cells[2 * edge + 1];
    float e1 = mpas_mul(normal[3*edge], east[3*c1]);
    e1 = mpas_add(e1, mpas_mul(normal[3*edge+1], east[3*c1+1]));
    e1 = mpas_add(e1, mpas_mul(normal[3*edge+2], east[3*c1+2]));
    float n1 = mpas_mul(normal[3*edge], north[3*c1]);
    n1 = mpas_add(n1, mpas_mul(normal[3*edge+1], north[3*c1+1]));
    n1 = mpas_add(n1, mpas_mul(normal[3*edge+2], north[3*c1+2]));
    float e2 = mpas_mul(normal[3*edge], east[3*c2]);
    e2 = mpas_add(e2, mpas_mul(normal[3*edge+1], east[3*c2+1]));
    e2 = mpas_add(e2, mpas_mul(normal[3*edge+2], east[3*c2+2]));
    float n2 = mpas_mul(normal[3*edge], north[3*c2]);
    n2 = mpas_add(n2, mpas_mul(normal[3*edge+1], north[3*c2+1]));
    n2 = mpas_add(n2, mpas_mul(normal[3*edge+2], north[3*c2+2]));
    for (int k = 0; k < nlev; ++k) {
        const int ei = k * nedges + edge;
        const int c1i = k * ncells + c1;
        const int c2i = k * ncells + c2;
        float value = mpas_mul(mpas_mul(du[c1i], 0.5f), e1);
        value = mpas_add(value, mpas_mul(mpas_mul(dv[c1i], 0.5f), n1));
        value = mpas_add(value, mpas_mul(mpas_mul(du[c2i], 0.5f), e2));
        value = mpas_add(value, mpas_mul(mpas_mul(dv[c2i], 0.5f), n2));
        value = mpas_mul(value, rho_edge[ei]);
        if (!isfinite(value) || !isfinite(rho_edge[ei])) atomicExch(invalid, 1);
        out_ru[ei] = value;
    }
}
