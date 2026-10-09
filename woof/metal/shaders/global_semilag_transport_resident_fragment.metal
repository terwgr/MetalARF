// Parallel-transport a resident vector using resident departure coordinates.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_transport_resident_fragment_bindings.metalh"
using namespace metal;

static inline void global_sincos_soft64(ulong input, thread ulong &sine,
                                        thread ulong &cosine) {
    const ulong half_pi = 0x3ff921fb54442d18ul;
    const ulong pi = 0x400921fb54442d18ul;
    ulong x = input;
    bool cosine_negative = false;
    if (soft64_greater(x, half_pi)) {
        x = soft64_sub(pi, x);
        cosine_negative = true;
    } else if (soft64_less(x, soft64_negate_bits(half_pi))) {
        x = soft64_sub(soft64_negate_bits(pi), x);
        cosine_negative = true;
    }
    const ulong x2 = soft64_mul(x, x);
    ulong sp = 0x3ce952c77030ad4aul;
    sp = soft64_add(0xbd6ae7f3e733b81ful, soft64_mul(x2, sp));
    sp = soft64_add(0x3de6124613a86d09ul, soft64_mul(x2, sp));
    sp = soft64_add(0xbe5ae64567f544e4ul, soft64_mul(x2, sp));
    sp = soft64_add(0x3ec71de3a556c734ul, soft64_mul(x2, sp));
    sp = soft64_add(0xbf2a01a01a01a01aul, soft64_mul(x2, sp));
    sp = soft64_add(0x3f81111111111111ul, soft64_mul(x2, sp));
    sp = soft64_add(0xbfc5555555555555ul, soft64_mul(x2, sp));
    sine = soft64_mul(x, soft64_add(0x3ff0000000000000ul,
                                      soft64_mul(x2, sp)));

    ulong cp = 0x3d2ae7f3e733b81ful;
    cp = soft64_add(0xbda93974a8c07c9dul, soft64_mul(x2, cp));
    cp = soft64_add(0x3e21eed8eff8d898ul, soft64_mul(x2, cp));
    cp = soft64_add(0xbe927e4fb7789f5cul, soft64_mul(x2, cp));
    cp = soft64_add(0x3efa01a01a01a01aul, soft64_mul(x2, cp));
    cp = soft64_add(0xbf56c16c16c16c17ul, soft64_mul(x2, cp));
    cp = soft64_add(0x3fa5555555555555ul, soft64_mul(x2, cp));
    cp = soft64_add(0xbfe0000000000000ul, soft64_mul(x2, cp));
    cosine = soft64_add(0x3ff0000000000000ul, soft64_mul(x2, cp));
    if (cosine_negative) cosine = soft64_negate_bits(cosine);
}

kernel void global_semilag_transport_resident_fragment(
    constant GlobalSemilagTransportResidentFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagTransportResidentFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const uint latitude = (tid / uint(p.nlon)) % uint(p.nlat);
    const uint longitude = tid % uint(p.nlon);
    const ulong lambda = soft64_mul(r.xi[tid], r.dlam[0]);
    ulong sl, cl, sp, cp;
    global_sincos_soft64(lambda, sl, cl);
    global_sincos_soft64(r.phi[tid], sp, cp);
    const ulong px = soft64_mul(cp, cl);
    const ulong py = soft64_mul(cp, sl);
    const ulong pz = sp;
    const ulong qx = soft64_mul(r.cos_lat[latitude], r.cos_lon[longitude]);
    const ulong qy = soft64_mul(r.cos_lat[latitude], r.sin_lon[longitude]);
    const ulong qz = r.sin_lat[latitude];
    const ulong dot = soft64_add(
        soft64_add(soft64_mul(px, qx), soft64_mul(py, qy)),
        soft64_mul(pz, qz));
    const ulong den = soft64_greater(dot, 0xbfeffffde7210be9ul)
        ? soft64_add(0x3ff0000000000000ul, dot)
        : 0x3eb0c6f7a0b5ed8dul;
    const ulong sx = soft64_add(px, qx);
    const ulong sy = soft64_add(py, qy);
    const ulong sz = soft64_add(pz, qz);
    const ulong sv = soft64_div(
        soft64_add(soft64_add(soft64_mul(sx, r.vx[tid]),
                              soft64_mul(sy, r.vy[tid])),
                   soft64_mul(sz, r.vz[tid])), den);
    const ulong pv = soft64_mul(0x4000000000000000ul,
        soft64_add(soft64_add(soft64_mul(px, r.vx[tid]),
                              soft64_mul(py, r.vy[tid])),
                   soft64_mul(pz, r.vz[tid])));
    const ulong wx = soft64_add(soft64_sub(r.vx[tid], soft64_mul(sv, sx)),
                                soft64_mul(pv, qx));
    const ulong wy = soft64_add(soft64_sub(r.vy[tid], soft64_mul(sv, sy)),
                                soft64_mul(pv, qy));
    const ulong wz = soft64_add(soft64_sub(r.vz[tid], soft64_mul(sv, sz)),
                                soft64_mul(pv, qz));
    const ulong sin_lat = r.sin_lat[latitude];
    const ulong cos_lat = r.cos_lat[latitude];
    const ulong sin_lon = r.sin_lon[longitude];
    const ulong cos_lon = r.cos_lon[longitude];
    r.u_arrival[tid] = soft64_add(
        soft64_negate_bits(soft64_mul(wx, sin_lon)), soft64_mul(wy, cos_lon));
    r.v_arrival[tid] = soft64_add(
        soft64_sub(soft64_negate_bits(soft64_mul(soft64_mul(wx, sin_lat), cos_lon)),
                   soft64_mul(soft64_mul(wy, sin_lat), sin_lon)),
        soft64_mul(wz, cos_lat));
}
