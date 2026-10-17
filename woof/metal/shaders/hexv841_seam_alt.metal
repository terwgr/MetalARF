// Metal mirror of the MPAS column-batch seam arithmetic: run_phase2:cp.divide(1, rho_dry)
// (woof/core/mpas_column_batch.py:1236).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_seam_alt_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv841_seam_alt(
    constant Hexv841SeamAltResources &r [[buffer(0)]],
    constant Hexv841SeamAltParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int gid = int(tid);
    if (gid >= p.n) return;
    r.alt[gid] = mpas_div(1.0f, r.rho_dry[gid]);
}
