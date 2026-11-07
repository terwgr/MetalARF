#include <metal_stdlib>
#include "da_iau_rate_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;

kernel void da_iau_rate(
    constant DaIauRateResources &r [[buffer(0)]],
    constant DaIauRateParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    uint term = soft32_mul(as_type<uint>(r.delta[gid]), as_type<uint>(p.factor));
    if (p.accumulate) term = soft32_add(as_type<uint>(r.rate[gid]), term);
    r.rate[gid] = as_type<float>(term);
}
