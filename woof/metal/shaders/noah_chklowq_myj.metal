#include <metal_stdlib>
#include "noah_chklowq_myj_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;

kernel void noah_chklowq_myj(
    constant NoahChklowqMyjResources &r [[buffer(0)]],
    constant NoahChklowqMyjParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.ny) * uint(p.nx)) return;
    const uint qv = as_type<uint>(r.qv1[gid]);
    const uint qgh = as_type<uint>(r.qgh[gid]);
    const uint q2k = soft32_div(qv, soft32_add(0x3f800000u, qv));
    const uint q2sat = soft32_div(qgh, soft32_add(0x3f800000u, qgh));
    const uint threshold = soft32_mul(q2sat, 0x3f733333u);
    if (soft32_greater_equal(q2k, threshold) &&
        soft32_less(q2k, as_type<uint>(r.qz0[gid])))
        r.chklowq[gid] = 0.0f;
}
