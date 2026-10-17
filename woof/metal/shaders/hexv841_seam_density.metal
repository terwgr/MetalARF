// Metal mirror of the MPAS column-batch seam arithmetic: run_phase1:(1 + qv) * rho_dry
// (woof/core/mpas_column_batch.py:1064).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_seam_density_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv841_seam_density(
    constant Hexv841SeamDensityResources &r [[buffer(0)]],
    constant Hexv841SeamDensityParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int gid = int(tid);
    if (gid >= p.n) return;
    r.rho[gid] = mpas_mul(mpas_add(1.0f, r.qv[gid]), r.rho_dry[gid]);
}
