// Ring-local top sponge. Each thread owns one level/latitude ring.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "global_top_sponge_fragment_bindings.metalh"
using namespace metal;

kernel void global_top_sponge_fragment(
    constant GlobalTopSpongeFragmentResources &r [[buffer(0)]],
    constant GlobalTopSpongeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint rings = uint(p.nlevel) * uint(p.nlat);
    if (tid >= rings) return;
    const uint level = tid / uint(p.nlat);
    const uint latitude = tid % uint(p.nlat);
    const uint plane = uint(p.nlat) * uint(p.nlon);
    const uint base = level * plane + latitude * uint(p.nlon);
    ulong sum_p = 0ul, sum_lid = 0ul, sum_u = 0ul, sum_v = 0ul;
    for (uint lon = 0u; lon < uint(p.nlon); ++lon) {
        sum_p = soft64_add(sum_p, r.p_full[base + lon]);
        sum_u = soft64_add(sum_u, r.u[base + lon]);
        sum_v = soft64_add(sum_v, r.v[base + lon]);
        sum_lid = soft64_add(sum_lid, r.p_half[latitude * uint(p.nlon) + lon]);
    }
    const ulong divisor = soft64_from_i32(p.nlon);
    const ulong ring_p = soft64_div(sum_p, divisor);
    const ulong ring_lid = soft64_div(sum_lid, divisor);
    const ulong base_p = r.constants[0];
    if (!soft64_less(ring_p, base_p)) {
        for (uint lon = 0u; lon < uint(p.nlon); ++lon) {
            r.u_out[base + lon] = r.u[base + lon];
            r.v_out[base + lon] = r.v[base + lon];
        }
        return;
    }
    ulong denom = soft64_sub(base_p, ring_lid);
    if (soft64_less(denom, 0x3f50624dd2f1a9fcul)) denom = 0x3f50624dd2f1a9fcul;
    ulong ramp = soft64_div(soft64_sub(ring_p, ring_lid), denom);
    if (soft64_less(ramp, 0ul)) ramp = 0ul;
    if (soft64_greater(ramp, 0x3ff0000000000000ul)) ramp = 0x3ff0000000000000ul;
    const ulong x = soft64_mul(0x3ff921fb54442d18ul, ramp);
    const ulong x2 = soft64_mul(x, x);
    ulong cosine = soft64_sub(0x3ff0000000000000ul,
        soft64_mul(x2, 0x3fe0000000000000ul));
    const ulong x4 = soft64_mul(x2, x2);
    cosine = soft64_add(cosine, soft64_mul(x4, 0x3fa5555555555555ul));
    const ulong x6 = soft64_mul(x4, x2);
    cosine = soft64_sub(cosine, soft64_mul(x6, 0x3f56c16c16c16c17ul));
    const ulong x8 = soft64_mul(x6, x2);
    cosine = soft64_add(cosine, soft64_mul(x8, 0x3efa01a01a01a01aul));
    const ulong rate = soft64_div(
        soft64_mul(cosine, cosine), r.constants[1]);
    const ulong decay = soft64_plm_exp(
        soft64_negate_bits(soft64_mul(r.constants[2], rate)));
    const ulong mean_u = soft64_div(sum_u, divisor);
    const ulong mean_v = soft64_div(sum_v, divisor);
    for (uint lon = 0u; lon < uint(p.nlon); ++lon) {
        r.u_out[base + lon] = soft64_add(mean_u,
            soft64_mul(soft64_sub(r.u[base + lon], mean_u), decay));
        r.v_out[base + lon] = soft64_add(mean_v,
            soft64_mul(soft64_sub(r.v[base + lon], mean_v), decay));
    }
}
