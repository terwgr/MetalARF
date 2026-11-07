#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_lbc_interior_bindings.metalh"
using namespace metal;
kernel void wrf_exact_lbc_interior(constant WrfExactLbcInteriorResources &r [[buffer(0)]],
                                  constant WrfExactLbcInteriorParams &p [[buffer(1)]],
                                  uint gid [[thread_position_in_grid]]) {
    const ulong plane=ulong(p.ny)*ulong(p.nx), flat=ulong(gid);
    if(flat>=ulong(p.nz)*plane) return;
    const ulong col=flat%plane;const int j=int(col/ulong(p.nx)),i=int(col%ulong(p.nx));
    if(j<p.active_width || j>=p.ny-p.active_width || i<p.active_width || i>=p.nx-p.active_width) return;
    device Exact32 *t=reinterpret_cast<device Exact32 *>(r.field_tend);
    device const Exact32 *held=reinterpret_cast<device const Exact32 *>(r.held);
    device const Exact32 *msft=reinterpret_cast<device const Exact32 *>(r.msft);
    Exact32 value=t[flat];
    if(p.divide_msf) value=value/msft[col];
    if(p.add_held) value=value+held[flat];
    t[flat]=value;
}
