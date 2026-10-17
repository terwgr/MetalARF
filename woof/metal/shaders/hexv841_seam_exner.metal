// Metal mirror of the MPAS column-batch seam arithmetic: run_phase1:cp.power(p / P0, RCP)
// (woof/core/mpas_column_batch.py:1057).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "pressure_power.metalh"
#include "hexv841_seam_exner_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv841_seam_exner(
    constant Hexv841SeamExnerResources &r [[buffer(0)]],
    constant Hexv841SeamExnerParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int gid = int(tid);
    if (gid >= p.n) return;
    r.exner[gid] = fp32_pressure_power(r.pressure[gid], p.p0, p.rcp);
}
