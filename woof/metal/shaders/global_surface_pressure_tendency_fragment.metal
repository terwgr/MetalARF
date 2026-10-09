#include <metal_stdlib>
#include "global_surface_pressure_tendency_fragment_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;

// needs integration: fused hybrid layer-mass and log-surface-pressure
// tendency preparation for Global dynamics.
#pragma clang fp contract(off)

kernel void global_surface_pressure_tendency_fragment(
    constant GlobalSurfacePressureTendencyFragmentResources &r [[buffer(0)]],
    constant GlobalSurfacePressureTendencyFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if(col>=uint(p.count)) return;
    r.log_surface_tendency[col]=soft64_div(
        r.surface_tendency[col],r.surface_pressure[col]);
    for(uint level=0;level<uint(p.nlevel);++level) {
        r.layer_tendency[ulong(level)*ulong(p.count)+ulong(col)] =
            soft64_mul(r.delta_b[level],r.surface_tendency[col]);
    }
}
