// Metal mirror of the MPAS column-batch seam arithmetic: run_phase1:cp.subtract(z_if[1:], z_if[:-1])
// (woof/core/mpas_column_batch.py:1062).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_seam_thickness_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv841_seam_thickness(
    constant Hexv841SeamThicknessResources &r [[buffer(0)]],
    constant Hexv841SeamThicknessParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int gid = int(tid);
    if (gid >= p.n) return;
    r.dz[gid] = mpas_sub(r.z_interface[gid + p.ncol], r.z_interface[gid]);
}
