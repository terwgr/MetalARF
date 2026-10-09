#include <metal_stdlib>
#include "rayleigh_damp_bindings.metalh"

using namespace metal;

// CUDA applies one precomputed level factor to each contiguous field value.
kernel void rayleigh_damp(
    constant RayleighDampResources &r [[buffer(0)]],
    constant RayleighDampParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    const uint plane = uint(p.nys) * uint(p.nxs);
    const uint count = uint(p.nlev) * plane;
    if (idx >= count) return;
    r.f[idx] *= r.rdamp[idx / plane];
}
