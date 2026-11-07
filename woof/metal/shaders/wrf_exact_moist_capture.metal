#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_moist_capture_bindings.metalh"
using namespace metal;
kernel void wrf_exact_moist_capture(constant WrfExactMoistCaptureResources &r [[buffer(0)]],
                                   constant WrfExactMoistCaptureParams &p [[buffer(1)]],
                                   uint gid [[thread_position_in_grid]]) {
    const ulong plane=ulong(p.ny)*ulong(p.nx), flat=ulong(gid);
    if(flat>=ulong(p.nz)*plane) return;
    device const Exact32 *t=reinterpret_cast<device const Exact32 *>(r.tend);
    device const Exact32 *mu=reinterpret_cast<device const Exact32 *>(r.mu0);
    device const Exact32 *c1=reinterpret_cast<device const Exact32 *>(r.c1h);
    device const Exact32 *c2=reinterpret_cast<device const Exact32 *>(r.c2h);
    device const Exact32 *msft=reinterpret_cast<device const Exact32 *>(r.msft);
    device Exact32 *out=reinterpret_cast<device Exact32 *>(r.dst);
    Exact32 rate=t[flat];
    rate=rate/(c1[flat/plane]*mu[flat%plane]+c2[flat/plane]);
    if(p.has_msf) rate=rate*msft[flat%plane];
    out[flat]=rate;
}
