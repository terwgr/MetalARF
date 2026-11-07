#include <metal_stdlib>
#include "da_state_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;

kernel void da_state_binary(
    constant DaStateBinaryResources &r [[buffer(0)]],
    constant DaStateBinaryParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    const uint left = as_type<uint>(r.left[gid]);
    const uint right = as_type<uint>(r.right[gid]);
    r.out[gid] = as_type<float>(p.subtract ? soft32_sub(left, right)
                                         : soft32_add(left, right));
}
