#include <metal_stdlib>
using namespace metal;
#include "nssl2_runtime_prepare_bindings.metalh"
#include "nssl2_runtime_finish_bindings.metalh"
#include "nssl2_radius_microns_bindings.metalh"

kernel void nssl2_runtime_prepare(
    constant Nssl2RuntimePrepareResources& r [[buffer(0)]],
    constant Nssl2RuntimePrepareParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    const uint n = uint(p.nz * p.ncol);
    if (idx >= n) return;
    const uint k = idx / uint(p.ncol);
    const float thb = r.thb[p.thb_3d ? idx : k];
    const float phb0 = r.phb[p.phb_3d ? idx : k];
    const float phb1 = r.phb[p.phb_3d ? idx + uint(p.ncol) : k + 1u];
    const float th = thb + r.thp[idx];
    r.theta[idx] = th;
    r.rho[idx] = 1.0f / r.alt[idx];
    r.pii[idx] = pow(r.pressure[idx] / p.p0, p.rcp);
    r.h_diabatic[idx] = th;
    const float z0 = (phb0 + r.php[idx]) / p.gravity;
    const float z1 = (phb1 + r.php[idx + uint(p.ncol)]) / p.gravity;
    r.dz[idx] = z1 - z0;
}

kernel void nssl2_runtime_finish(
    constant Nssl2RuntimeFinishResources& r [[buffer(0)]],
    constant Nssl2RuntimeFinishParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    if (p.no_mp_heating) {
        r.heating[idx] = 0.0f;
        return;
    }
    float tendency = r.theta[idx] - r.heating[idx];
    if (!isnan(tendency)) {
        tendency = min(p.limit, tendency);
        tendency = max(-p.limit, tendency);
    }
    r.theta_perturbation[idx] += tendency;
    r.heating[idx] = tendency / p.dt;
}

kernel void nssl2_radius_microns(
    constant Nssl2RadiusMicronsResources& r [[buffer(0)]],
    constant Nssl2RadiusMicronsParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    r.cloud[idx] *= 1.0e6f;
    r.ice[idx] *= 1.0e6f;
    r.snow[idx] *= 1.0e6f;
}
