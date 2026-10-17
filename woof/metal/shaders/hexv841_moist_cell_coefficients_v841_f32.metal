// Hex v8.4.1 Metal transcription of CUDA kernel moist_cell_coefficients_v841_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_moist_cell_coefficients_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)


kernel void hexv841_moist_cell_coefficients_v841_f32(
    constant Hexv841MoistCellCoefficientsV841F32Resources &r [[buffer(0)]],
    constant Hexv841MoistCellCoefficientsV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nmass = p.nmass;
    device const float *scalars = r.scalars;
    device float *qtot = r.qtot;
    device float *cqw = r.cqw;
    device int *invalid = r.invalid;

    const int cell = int(tid);
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        float total = 0.0f;
        // nmass, not nScalars: only the mass-LOADING species enter qtot.
        // The species row orders them as a contiguous leading run, so one
        // count is the whole rule (woof.hex.species_row).  A number
        // concentration folded in here would collapse 1/(1+qtot); a rime
        // MASS that is a component of another species would double-count it.
        for (int species = 0; species < nmass; ++species) {
            const float value = scalars[S3(species, k, cell, nlev, ncells)];
            if (!isfinite(value)) atomicExch(invalid, 1);
            total = mpas_add(total, value);
        }
        if (!isfinite(total)) atomicExch(invalid, 1);
        qtot[i] = total;
    }
    // Native Fortran assigns cqw only for k=2..nVertLevels.  The unused
    // k=1/zero-based boundary is intentionally left untouched here.
    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const float total = mpas_mul(0.5f, mpas_add(
            qtot[i], qtot[C2(k - 1, cell, ncells)]));
        const float value = mpas_div(1.0f, mpas_add(1.0f, total));
        if (!isfinite(total) || !isfinite(value)) atomicExch(invalid, 1);
        cqw[i] = value;
    }
}
