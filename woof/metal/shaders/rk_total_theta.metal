#include <metal_stdlib>
#include "rk_total_theta_bindings.metalh"
#include "rk_total_mu_bindings.metalh"

using namespace metal;

kernel void rk_total_theta(constant RkTotalThetaResources &r [[buffer(0)]],
                           constant RkTotalThetaParams &q [[buffer(1)]],
                           uint tid [[thread_position_in_grid]]) {
    const uint n = uint(q.nz * q.ny * q.nx);
    if (tid >= n) return;
    const uint plane = uint(q.ny * q.nx);
    const uint k = tid / plane;
    const ulong base = q.base3d ? ulong(tid) : ulong(k);
    r.out[tid] = r.thb[base] + r.thp[tid];
}

kernel void rk_total_mu(constant RkTotalMuResources &r [[buffer(0)]],
                        constant RkTotalMuParams &q [[buffer(1)]],
                        uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(q.ny * q.nx)) return;
    r.out[tid] = r.mub2d[tid] + r.mup[tid];
}
