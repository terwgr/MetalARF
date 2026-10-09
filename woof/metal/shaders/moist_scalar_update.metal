#include <metal_stdlib>
#include "moist_scalar_update_bindings.metalh"

using namespace metal;

#pragma clang fp contract(off)

kernel void moist_scalar_update(
    constant MoistScalarUpdateResources &r [[buffer(0)]],
    constant MoistScalarUpdateParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong flat = ulong(gid);
    const ulong nz = ulong(p.nz), ny = ulong(p.ny), nx = ulong(p.nx);
    const ulong plane = ny * nx;
    if (flat >= nz * plane) return;

    const ulong k = flat / plane;
    const ulong cell = flat - k * plane;
    float tendency = r.tend[flat];
    if (p.has_msf) tendency = tendency * r.msft[cell];
    if (p.has_lbc) {
        const int j = int(cell / nx);
        const int i = int(cell - ulong(j) * nx);
        const bool specified = p.spec_zone > 0
            && (j < p.spec_zone || j >= int(ny) - p.spec_zone
                || i < p.spec_zone || i >= int(nx) - p.spec_zone);
        if (specified) tendency = r.lbc[flat];
        else tendency = tendency + r.lbc[flat];
    }
    if (p.has_physics) tendency = tendency + r.physics[flat];
    if (p.has_fixed) tendency = tendency + r.fixed[flat];
    if (p.clear_specified_advection && p.spec_zone > 0) {
        const int j = int(cell / nx);
        const int i = int(cell - ulong(j) * nx);
        if (j < p.spec_zone || j >= int(ny) - p.spec_zone
                || i < p.spec_zone || i >= int(nx) - p.spec_zone) {
            // WRF computes but never applies PD vertical-divergence updates
            // on the specified ring. The held boundary source was already
            // folded into q0_eff before flux construction.
            tendency = 0.0f;
        }
    }

    const float old_mass = r.c1h[k] * r.mu0[cell] + r.c2h[k];
    const float new_mass = r.c1h[k] * r.mu[cell] + r.c2h[k];
    float value = old_mass * r.q0[flat];
    tendency = p.dt_eff * tendency;
    value = value + tendency;
    value = value / new_mass;
    if (p.clamp) {
        value = isnan(value) ? as_type<float>(0x7fffffffU)
                             : max(value, 0.0f);
    }
    r.q[flat] = value;
}
