// Metal mirror of the MPAS column-batch seam arithmetic: run_phase2:cp.multiply(z_interface, G)
// (woof/core/mpas_column_batch.py:1243).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_seam_geopotential_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv841_seam_geopotential(
    constant Hexv841SeamGeopotentialResources &r [[buffer(0)]],
    constant Hexv841SeamGeopotentialParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int gid = int(tid);
    if (gid >= p.n) return;
    r.php[gid] = mpas_mul(r.z_interface[gid], p.g);
}
