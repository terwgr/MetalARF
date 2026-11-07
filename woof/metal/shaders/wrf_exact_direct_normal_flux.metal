#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "wrf_exact_direct_normal_flux_bindings.metalh"
using namespace metal;
kernel void wrf_exact_direct_normal_flux(
    constant WrfExactDirectNormalFluxResources &r [[buffer(0)]],
    constant WrfExactDirectNormalFluxParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    device const uint *direct=reinterpret_cast<device const uint *>(r.direct);
    device const uint *cosine=reinterpret_cast<device const uint *>(r.cosine);
    device uint *normal=reinterpret_cast<device uint *>(r.normal);
    const uint c=cosine[i];
    const uint floor=as_type<uint>(1.0e-6f);
    const uint divisor=soft32_less(c,floor) ? floor : c;
    normal[i]=soft32_less_equal(c,0u) ? 0u : soft32_div(direct[i],divisor);
}
