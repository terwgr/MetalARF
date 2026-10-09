// needs integration: FP64 Global four-sweep convective column adjustment.
//
// One thread owns each column; the four deterministic sweeps and the
// sequential layer-pair dependencies match ReferencePhysics._convective_adjust.
// `mass` is dp/g and `q` is [water species, level, column]. The ABI owner
// must register this fragment before dispatch.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_convective_adjust_fragment_bindings.metalh"

using namespace metal;

kernel void global_convective_adjust_fragment(
    constant GlobalConvectiveAdjustFragmentResources &r [[buffer(0)]],
    constant GlobalConvectiveAdjustFragmentParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(q.ncolumn)) return;
    uint count = 0u;
    for (uint sweep = 0u; sweep < 4u; ++sweep) {
        for (uint level = 0u; level + 1u < q.nlevel; ++level) {
            const uint lo = level * q.ncolumn + tid;
            const uint hi = (level + 1u) * q.ncolumn + tid;
            if (!soft64_less(r.theta[lo], r.theta[hi])) continue;
            const ulong total_mass = soft64_add(r.mass[lo], r.mass[hi]);
            const ulong safe_mass = soft64_greater(
                total_mass, 0x3d719799812dea11ul) // 1.0e-12
                ? total_mass : 0x3d719799812dea11ul;
            const ulong mixed_theta = soft64_div(
                soft64_add(soft64_mul(r.mass[lo], r.theta[lo]),
                           soft64_mul(r.mass[hi], r.theta[hi])),
                safe_mass);
            r.theta[lo] = mixed_theta;
            r.theta[hi] = mixed_theta;
            for (uint species = 0u; species < q.nspecies; ++species) {
                const uint qlo = (species * q.nlevel + level) * q.ncolumn + tid;
                const uint qhi = (species * q.nlevel + level + 1u) * q.ncolumn + tid;
                const ulong mixed = soft64_div(
                    soft64_add(soft64_mul(r.mass[lo], r.q[qlo]),
                               soft64_mul(r.mass[hi], r.q[qhi])),
                    safe_mass);
                r.q[qlo] = mixed;
                r.q[qhi] = mixed;
            }
            count += 1u;
        }
    }
    r.mixed_pairs[tid] = count;
}
