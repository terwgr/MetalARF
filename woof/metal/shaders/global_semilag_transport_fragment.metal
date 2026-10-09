// FP64 minimal-rotation transport from departure tangent plane to arrival.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_transport_fragment_bindings.metalh"

using namespace metal;

kernel void global_semilag_transport_fragment(
    constant GlobalSemilagTransportFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagTransportFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.count))return;
    const ulong dot=soft64_add(
        soft64_add(soft64_mul(r.px[tid],r.qx[tid]),
                   soft64_mul(r.py[tid],r.qy[tid])),
        soft64_mul(r.pz[tid],r.qz[tid]));
    const ulong den=soft64_greater(dot,0xbfeffffde7210be9ul)
        ? soft64_add(0x3ff0000000000000ul,dot)
        : 0x3eb0c6f7a0b5ed8dul;
    const ulong sx=soft64_add(r.px[tid],r.qx[tid]);
    const ulong sy=soft64_add(r.py[tid],r.qy[tid]);
    const ulong sz=soft64_add(r.pz[tid],r.qz[tid]);
    const ulong sv=soft64_div(
        soft64_add(soft64_add(soft64_mul(sx,r.vx[tid]),
                              soft64_mul(sy,r.vy[tid])),
                   soft64_mul(sz,r.vz[tid])),den);
    const ulong pv=soft64_mul(0x4000000000000000ul,
        soft64_add(soft64_add(soft64_mul(r.px[tid],r.vx[tid]),
                              soft64_mul(r.py[tid],r.vy[tid])),
                   soft64_mul(r.pz[tid],r.vz[tid])));
    const ulong wx=soft64_add(
        soft64_sub(r.vx[tid],soft64_mul(sv,sx)),
        soft64_mul(pv,r.qx[tid]));
    const ulong wy=soft64_add(
        soft64_sub(r.vy[tid],soft64_mul(sv,sy)),
        soft64_mul(pv,r.qy[tid]));
    const ulong wz=soft64_add(
        soft64_sub(r.vz[tid],soft64_mul(sv,sz)),
        soft64_mul(pv,r.qz[tid]));
    const ulong sla=r.sin_latlon[tid];
    const ulong cla=r.cos_latlon[tid];
    const ulong sph=r.sin_lat[tid];
    const ulong cph=r.cos_lat[tid];
    r.u_arrival[tid]=soft64_add(
        soft64_negate_bits(soft64_mul(wx,sla)),soft64_mul(wy,cla));
    r.v_arrival[tid]=soft64_add(
        soft64_sub(soft64_negate_bits(soft64_mul(soft64_mul(wx,sph),cla)),
                   soft64_mul(soft64_mul(wy,sph),sla)),
        soft64_mul(wz,cph));
}
