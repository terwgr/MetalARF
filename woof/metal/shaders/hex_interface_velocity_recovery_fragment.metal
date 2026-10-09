#include <metal_stdlib>
#include "hex_interface_velocity_recovery_fragment_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

// Recover the intermediate interfaces in the driver from F:2793-2903.
// The bottom and top interfaces are zero in this non-terrain path. Preserve
// the driver's ordered interpolation and two divisions.
kernel void hex_interface_velocity_recovery_fragment(
    constant HexInterfaceVelocityRecoveryFragmentResources &r [[buffer(0)]],
    constant HexInterfaceVelocityRecoveryFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev_plus_one) * uint(p.ncells);
    if (tid >= count) return;
    const int level = int(tid / uint(p.ncells));
    const int cell = int(tid % uint(p.ncells));
    r.error[tid] = 0;
    if (level == 0 || level == p.nlev) {
        r.velocity[tid] = 0.0f;
        return;
    }
    const ulong here = ulong(level) * ulong(p.ncells) + ulong(cell);
    const ulong below = ulong(level - 1) * ulong(p.ncells) + ulong(cell);
    const float density = r.fzm[level] * r.rho[here]
                        + r.fzp[level] * r.rho[below];
    const float metric = r.fzm[level] * r.zz[here]
                       + r.fzp[level] * r.zz[below];
    if (!(density > 0.0f)) {
        r.error[tid] = 1;
        r.velocity[tid] = 0.0f;
        return;
    }
    if (metric == 0.0f) {
        r.error[tid] = 2;
        r.velocity[tid] = 0.0f;
        return;
    }
    const float momentum = r.rho_w[here] / metric;
    r.velocity[tid] = momentum / density;
}
