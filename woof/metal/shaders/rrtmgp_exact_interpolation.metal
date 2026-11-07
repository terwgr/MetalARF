// Strict RRTMGP pressure and temperature interpolation metadata.
#include <metal_stdlib>
#include "hexv842_mpas_thompson_column_math.metalh"
#include "rrtmgp_exact_interpolation_bindings.metalh"

using namespace metal;

// RRTMGP's pinned log uses fused double operations. The existing tables
// have the same words, but native_gfk_log rounds those operations separately.
static inline F rrtmgp_exact_log(F x) {
    uint ix = x.bits;
    if (ix == 0x3f800000u) return 0.0f;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {
        if (ix * 2u == 0u) return f32_word(0xff800000u);
        if (ix == 0x7f800000u) return x;
        if ((ix & 0x80000000u) || ix * 2u >= 0xff000000u) return f32_word(0x7fc00000u);
        ix = (x * F(8388608.0f)).bits - (23u << 23);
    }
    const uint tmp = ix - 0x3f330000u;
    const uint i = (tmp >> 19) & 15u;
    const int k = int(tmp) >> 23;
    const uint iz = ix - (tmp & 0xff800000u);
    const Soft8 z = SD8(soft64_from_f32_bits(iz));
    const Soft8 r = __fma_rn(z, native_tc_math::GFK_LOGF_INVC[i], SD8(0xbff0000000000000ul));
    const Soft8 y0 = __fma_rn(Soft8(k), SD8(0x3fe62e42fefa39eful), native_tc_math::GFK_LOGF_LOGC[i]);
    const Soft8 r2 = __dmul_rn(r, r);
    Soft8 y = __fma_rn(SD8(0x3fd5575b0be00b6aul), r, SD8(0xbfdffffef20a4123ul));
    y = __fma_rn(r2, SD8(0xbfd00ea348b88334ul), y);
    y = __fma_rn(r2, y, __dadd_rn(y0, r));
    return native_tc_math::gfk_d2f_rn(y);
}

kernel void rrtmgp_exact_interpolation(
    constant RrtmgpExactInterpolationResources &r [[buffer(0)]],
    constant RrtmgpExactInterpolationParams &p [[buffer(1)]],
    uint cell [[thread_position_in_grid]]) {
    if (cell >= uint(p.ncell)) return;
    const F pressure = r.play[cell], temperature = r.tlay[cell];
    const F tmin = r.temp_ref[0];
    const F dtemp = __fdiv_rn(F(r.temp_ref[p.ntemp - 1]) - tmin, F(float(p.ntemp - 1)));
    const F inv_temp = __fdiv_rn(1.0f, dtemp);
    const F log_first = rrtmgp_exact_log(r.press_ref[0]);
    const F dpress = __fdiv_rn(rrtmgp_exact_log(r.press_ref[p.npres - 1]) - log_first,
                              F(float(p.npres - 1)));
    const F inv_press = __fdiv_rn(1.0f, dpress);
    const F trop = native_tc_math::thompson_aa_expf(rrtmgp_exact_log(p.press_ref_trop));
    const int jtemp = max(1, min(p.ntemp - 1, int((temperature - (tmin - dtemp)) * inv_temp)));
    const F ftemp = (temperature - F(r.temp_ref[jtemp - 1])) * inv_temp;
    const F locpress = F(1.0f) + (rrtmgp_exact_log(pressure) - log_first) * inv_press;
    const int jpress = !(locpress > F(1.0f)) ? 1
        : locpress >= F(float(p.npres - 1)) ? p.npres - 1 : int(locpress);
    const int iatm = pressure > trop ? 0 : 1;
    r.iatm_out[cell] = as_type<float>(iatm);
    r.jt_out[cell] = as_type<float>(jtemp - 1);
    r.jp_out[cell] = as_type<float>((jpress - 1) + iatm);
    r.ftemp_out[cell] = as_type<float>(ftemp.bits);
    r.fpress_out[cell] = as_type<float>((locpress - F(float(jpress))).bits);
}
