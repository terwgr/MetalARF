#include <metal_stdlib>
#include "da_child_positivity_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;

kernel void da_child_positivity(
    constant DaChildPositivityResources &r [[buffer(0)]],
    constant DaChildPositivityParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    uint background = as_type<uint>(r.background[gid]);
    uint increment = as_type<uint>(r.increment[gid]);
    if ((background & 0x7f800000u) == 0x7f800000u ||
        (increment & 0x7f800000u) == 0x7f800000u) {
        r.out[gid] = r.increment[gid];
        r.negative[gid] = 0x7ff8000000000000ul;
        return;
    }
    ulong analysis = soft64_add(soft64_from_f32_bits(background),
                                soft64_from_f32_bits(increment));
    bool negative = soft64_less(analysis, 0ul);
    r.out[gid] = as_type<float>(negative ? (background ^ 0x80000000u) : increment);
    r.negative[gid] = negative ? analysis : 0ul;
}
