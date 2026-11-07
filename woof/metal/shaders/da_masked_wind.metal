#include <metal_stdlib>
#include "da_masked_wind_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;

kernel void da_masked_wind_add(
    constant DaMaskedWindAddResources &r [[buffer(0)]],
    constant DaMaskedWindAddParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    const uint original = r.left[gid];
    r.out[gid] = r.mask[gid % uint(p.plane)]
        ? soft32_add(original, r.right[gid]) : original;
}
