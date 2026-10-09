#include <metal_stdlib>
using namespace metal;

#include "../generated/surface_cap_land_q2_bindings.metalh"

kernel void surface_cap_land_q2(
    device SurfaceCapLandQ2Resources& r [[buffer(0)]],
    constant SurfaceCapLandQ2Params& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (uint(p.n) != uint(p.nx) * uint(p.ny) || gid >= uint(p.n)) return;

    float q = r.q2[gid];
    if (r.xland[gid] < 1.5f) {
        float cap = 1.05f * r.qv1[gid];
        // numpy.minimum propagates either NaN. Keep that behavior explicit
        // instead of relying on the platform-specific fmin NaN convention.
        if (isnan(q)) {
            r.q2[gid] = q;
        } else if (isnan(cap)) {
            r.q2[gid] = cap;
        } else {
            r.q2[gid] = q < cap ? q : cap;
        }
    }
}
