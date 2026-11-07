#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_moist_tendency_bindings.metalh"
using namespace metal;
kernel void wrf_exact_moist_tendency(constant WrfExactMoistTendencyResources &r [[buffer(0)]],
                                     constant WrfExactMoistTendencyParams &p [[buffer(1)]],
                                     uint gid [[thread_position_in_grid]]) {
    const ulong plane=ulong(p.ny)*ulong(p.nx), flat=ulong(gid);
    if(flat>=ulong(p.nz)*plane) return;
    device Exact32 *t=reinterpret_cast<device Exact32 *>(r.tend);
    device const Exact32 *source=reinterpret_cast<device const Exact32 *>(r.source);
    device const Exact32 *msft=reinterpret_cast<device const Exact32 *>(r.msft);
    Exact32 value=t[flat];
    if(p.has_msf) value=value*msft[flat%plane];
    t[flat]=value+source[flat];
}
