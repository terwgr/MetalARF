// Hex v8.4.1 Metal transcription of CUDA kernel moist_edge_coefficients_v841_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_moist_edge_coefficients_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)


kernel void hexv841_moist_edge_coefficients_v841_f32(
    constant Hexv841MoistEdgeCoefficientsV841F32Resources &r [[buffer(0)]],
    constant Hexv841MoistEdgeCoefficientsV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int nmass = p.nmass;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *scalars = r.scalars;
    device float *cqu = r.cqu;
    device int *invalid = r.invalid;

    const int edge = int(tid);
    if (edge >= nedges) return;
    const int cell1 = cells_on_edge[2 * edge];
    const int cell2 = cells_on_edge[2 * edge + 1];
    for (int k = 0; k < nlev; ++k) {
        float total = 0.0f;
        // See moist_cell_coefficients_v841_f32: nmass is the loading run.
        // The per-species average is summed in species order, NOT summed
        // then averaged -- that ordering is FP32-observable and is what the
        // frozen WSM6 arm's bytes were measured under.
        for (int species = 0; species < nmass; ++species) {
            const float q1 = scalars[S3(species, k, cell1, nlev, ncells)];
            const float q2 = scalars[S3(species, k, cell2, nlev, ncells)];
            if (!isfinite(q1) || !isfinite(q2)) atomicExch(invalid, 1);
            total = mpas_add(total, mpas_mul(0.5f, mpas_add(q1, q2)));
        }
        const float value = mpas_div(1.0f, mpas_add(1.0f, total));
        if (!isfinite(total) || !isfinite(value)) atomicExch(invalid, 1);
        cqu[E2(k, edge, nedges)] = value;
    }
}
