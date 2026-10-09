#include <metal_stdlib>
#include "surface_diagnostics_seed_bindings.metalh"
using namespace metal;

kernel void surface_diagnostics_seed(
    constant SurfaceDiagnosticsSeedResources &r [[buffer(0)]],
    constant SurfaceDiagnosticsSeedParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (ulong(tid) >= q.n) return;
    const int i = int(tid);
    const int x = i % q.nx;
    const int y = i / q.nx;
    const int ubase = y * (q.nx + 1) + x;
    const int vbase = y * q.nx + x;
    r.th2[i] = r.t2[i] * pow(100000.0f / r.psfc[i], 287.0f / 1004.0f);
    r.u10[i] = 0.5f * (r.u10_staggered[ubase] + r.u10_staggered[ubase + 1]);
    r.v10[i] = 0.5f * (r.v10_staggered[vbase] + r.v10_staggered[vbase + q.nx]);
}
