// Solar zenith geometry and shortwave column inputs for Global physics.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_gray_solar_fragment_bindings.metalh"
using namespace metal;

static inline void global_gray_sincos(ulong input, thread ulong &sine,
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

static inline ulong global_gray_max(ulong a, ulong b) {
    return soft64_greater(a, b) ? a : b;
}

kernel void global_gray_solar_fragment(
    constant GlobalGraySolarFragmentResources &r [[buffer(0)]],
    constant GlobalGraySolarFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    ulong sin_solar, cos_solar;
    global_gray_sincos(r.solar_angle[0], sin_solar, cos_solar);
    const ulong cos_hour = soft64_sub(
        soft64_mul(sin_solar, r.sin_lon[tid]),
        soft64_mul(cos_solar, r.cos_lon[tid]));
    const ulong cosine_zenith = global_gray_max(
        soft64_mul(r.cos_lat[tid], cos_hour), 0ul);
    const ulong top = soft64_mul(r.solar_constant[0], cosine_zenith);
    const ulong sw_surface = soft64_mul(
        soft64_sub(0x3ff0000000000000ul, r.atmospheric_absorptivity[0]), top);
    r.cosine_zenith[tid] = cosine_zenith;
    r.sw_atmosphere[tid] = soft64_mul(
        r.atmospheric_absorptivity[0], top);
    r.absorbed_shortwave[tid] = soft64_mul(
        soft64_sub(0x3ff0000000000000ul, r.albedo[tid]), sw_surface);
}
