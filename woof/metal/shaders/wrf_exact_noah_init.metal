#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "wrf_exact_noah_init_check_bindings.metalh"
#include "wrf_exact_noah_init_copy_bindings.metalh"
using namespace metal;

kernel void wrf_exact_noah_init_check(
    constant WrfExactNoahInitCheckResources &r [[buffer(0)]],
    constant WrfExactNoahInitCheckParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const uint n=uint(p.ny)*uint(p.nx);
    if (gid>=n) return;
    const int raw_category=r.isltyp[gid];
    if (raw_category<1 || raw_category>p.slcats) {
        atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint*>(&r.status[0]),1u,memory_order_relaxed);
        return;
    }
    const int category=raw_category-1;
    device const uint *table=reinterpret_cast<device const uint*>(r.soiltbl);
    const int base=category*10;
    if (soft32_greater(table[base],0u) && soft32_greater(table[base+3],0u)
            && soft32_greater(table[base+5],0u)) {
        device const uint *temperature=reinterpret_cast<device const uint*>(r.tslb);
        for (uint k=0;k<4u;++k) {
            if (soft32_less(temperature[k*n+gid],as_type<uint>(273.149f))) {
                atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint*>(&r.status[0]),2u,memory_order_relaxed);
                break;
            }
        }
    }
}

kernel void wrf_exact_noah_init_copy(
    constant WrfExactNoahInitCopyResources &r [[buffer(0)]],
    constant WrfExactNoahInitCopyParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const uint n=uint(p.ny)*uint(p.nx);
    if (gid>=n) return;
    device const uint *moisture=reinterpret_cast<device const uint*>(r.smois);
    device uint *liquid=reinterpret_cast<device uint*>(r.sh2o);
    for (uint k=0;k<4u;++k) liquid[k*n+gid]=moisture[k*n+gid];
}
