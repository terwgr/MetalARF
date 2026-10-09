#include <metal_stdlib>
#include "hex_acoustic_postsolve_damping_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// Apply the frozen interface momentum damping immediately after the vertical
// solve. Level zero and the lid are not prognostic interface velocities.
kernel void hex_acoustic_postsolve_damping(
        constant HexAcousticPostsolveDampingResources &r [[buffer(0)]],
        constant HexAcousticPostsolveDampingParams &q [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong count = ulong(q.nlev - 1) * ulong(q.ncells);
    if (q.nlev < 2 || q.ncells <= 0 || ulong(gid) >= count) return;

    const int cell = int(gid % uint(q.ncells));
    const int level = int(gid / uint(q.ncells)) + 1;
    const ulong at = ulong(level) * ulong(q.ncells) + ulong(cell);
    const ulong below = at - ulong(q.ncells);
    const float saved_delta = r.rw_save[at] - r.rw[at];
    const float density_at_interface =
        (r.fzm[level] * r.zz[at] + r.fzp[level] * r.zz[below])
        * (r.fzm[level] * r.rho_zz[at] + r.fzp[level] * r.rho_zz[below]);
    const float damping = r.dss[at];
    r.rw_p[at] =
        (r.rw_p[at] + saved_delta
         - q.dts * damping * density_at_interface * r.w[at])
        / (1.0f + q.dts * damping) - saved_delta;
}
