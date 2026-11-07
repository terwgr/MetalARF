#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_transport_rows_f32_bindings.metalh"
kernel void hexv842_mpas_transport_rows_f32(
    constant Hexv842MpasTransportRowsF32Resources &r [[buffer(0)]],
    constant Hexv842MpasTransportRowsF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    int i=int(tid), n=p.nrows*p.nlev*p.nsolve;
    if(i>=n) return;
    int cell=i%p.nsolve, level=(i/p.nsolve)%p.nlev, row=i/(p.nsolve*p.nlev);
    int full=(r.mapping[row]*p.nlev+level)*p.nfull+cell;
    if(p.scatter) r.dst[full]=r.src[i];
    else r.dst[i]=r.src[full];
}
