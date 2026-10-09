// Apply spherical metric factors to resident Global spectral gradients.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_gradient_metric_fragment_bindings.metalh"

using namespace metal;

kernel void global_semilag_gradient_metric_fragment(
    constant GlobalSemilagGradientMetricFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagGradientMetricFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const uint plane = uint(p.nlat) * uint(p.nlon);
    const uint row = (tid % plane) / uint(p.nlon);
    const ulong radius_coslat = soft64_mul(r.radius[0], r.coslat[row]);
    r.east[tid] = soft64_div(r.east_raw[tid], radius_coslat);
    r.north[tid] = soft64_div(r.north_raw[tid], r.radius[0]);
}
