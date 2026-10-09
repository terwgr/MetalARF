#include <metal_stdlib>
#include "total_theta_bindings.metalh"

using namespace metal;

// Form full theta = thb + thp for the periodic advection-only caller. thb is
// either a flat vertical profile or a per-column 3-D base field.
kernel void total_theta(constant TotalThetaResources &r [[buffer(0)]],
                        constant TotalThetaParams &p [[buffer(1)]],
                        uint flat [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny) * uint(p.nx);
    const uint count = uint(p.nz) * ncol;
    if (flat >= count) return;
    const uint k = flat / ncol;
    const float base = p.base3d ? r.thb[flat] : r.thb[k];
    r.theta[flat] = base + r.thp[flat];
}
