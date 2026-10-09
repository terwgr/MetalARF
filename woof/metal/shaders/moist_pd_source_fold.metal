#include <metal_stdlib>
using namespace metal;

#include "../generated/moist_pd_source_fold_bindings.metalh"

#pragma clang fp contract(off)

kernel void moist_pd_source_fold(
    device MoistPdSourceFoldResources& r [[buffer(0)]],
    constant MoistPdSourceFoldParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (uint(p.n) != uint(p.nz) * uint(p.ny) * uint(p.nx)
            || gid >= uint(p.n)) return;

    const uint plane = uint(p.ny) * uint(p.nx);
    const uint k = gid / plane;
    const uint cell = gid - k * plane;
    float chm0 = r.c1h[k] * r.mu0[cell];
    chm0 = chm0 + r.c2h[k];
    float value = r.q0[gid];
    if (p.has_fixed) {
        float source = p.dt_eff * r.fixed[gid];
        source = source / chm0;
        value = value + source;
    }
    if (p.has_physics) {
        float source = p.dt_eff * r.physics[gid];
        source = source / chm0;
        value = value + source;
    }
    if (p.has_lbc) {
        float source = p.dt_eff * r.lbc[gid];
        source = source / chm0;
        value = value + source;
    }
    r.q0_eff[gid] = value;
}
