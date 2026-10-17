// Metal mirror of the MPAS column-batch seam arithmetic: run_phase1:cp.multiply(theta, exner)
// (woof/core/mpas_column_batch.py:1061).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_seam_temperature_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv841_seam_temperature(
    constant Hexv841SeamTemperatureResources &r [[buffer(0)]],
    constant Hexv841SeamTemperatureParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int gid = int(tid);
    if (gid >= p.n) return;
    r.temperature[gid] = mpas_mul(r.theta[gid], r.exner[gid]);
}
