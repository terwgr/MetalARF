#include <metal_stdlib>
using namespace metal;

#include "real_total_geopotential_bindings.metalh"

// Keep setup geopotential addition as one ordinary FP32 add per element.
#pragma clang fp contract(off)

kernel void real_total_geopotential(
    constant RealTotalGeopotentialResources &r [[buffer(0)]],
    constant RealTotalGeopotentialParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const uint n = uint(p.nz + 1) * uint(p.ny) * uint(p.nx);
    if (gid >= n) return;

    const uint ncol = uint(p.ny) * uint(p.nx);
    const uint k = gid / ncol;
    const uint base_index = p.phb_3d ? gid : k;
    r.total_phi[gid] = r.phb[base_index] + r.php[gid];
}
