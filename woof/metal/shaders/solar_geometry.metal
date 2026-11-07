// WRF radconst and calc_coszen, including the WRF 4.8 orbital phase.
// The arithmetic graph matches core/kernels/rrtmg_legacy_solar.cu. All
// arithmetic and storage bindings use integer IEEE bit patterns so RN-even
// operations, subnormals and the pinned glibc trig are backend independent.
#include <metal_stdlib>
#include "solar_geometry_bindings.metalh"
#include "soft_glibc_solar_trig.metalh"
using namespace metal;

// Exact finite float32 remainder by 1440. Integer significand subtraction
// computes the mathematical remainder before repacking, without a rounded
// quotient or a native FTZ operation. Sign follows the dividend (Fortran MOD).
static inline uint sgr_mod1440(uint x) {
    const uint sign = x & 0x80000000u;
    uint magnitude = x & 0x7fffffffu;
    const uint divisor = 0x44b40000u;
    if (magnitude < divisor) return x;
    if (magnitude == divisor) return sign;
    if (magnitude >= 0x7f800000u) return sgt_invalid(x);
    int exponent = int(magnitude >> 23);
    const int divisor_exponent = int(divisor >> 23);
    magnitude = (magnitude & 0x7fffffu) | 0x800000u;
    const uint significand = (divisor & 0x7fffffu) | 0x800000u;
    for (; exponent > divisor_exponent; --exponent) {
        if (magnitude >= significand) magnitude -= significand;
        if (magnitude == 0u) return sign;
        magnitude <<= 1;
    }
    if (magnitude >= significand) magnitude -= significand;
    if (magnitude == 0u) return sign;
    for (; magnitude < 0x800000u; --exponent) magnitude <<= 1;
    if (exponent > 0) magnitude = (magnitude - 0x800000u) | (uint(exponent) << 23);
    else magnitude >>= uint(1 - exponent);
    return magnitude | sign;
}

kernel void solar_radconst(
    constant SolarRadconstResources &r [[buffer(0)]],
    constant SolarRadconstParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= 1u) return;
    const uint julian = p.julian;
    const uint obecl = soft32_mul(0x41bc0000u, p.degrad);
    const uint sinob = soft_glibc_solar_sinf(obecl);
    uint sxlong = 0u;
    if (soft32_greater_equal(julian, 0x42a00000u))
        sxlong = soft32_mul(p.dpd, soft32_sub(julian, 0x42a00000u));
    if (soft32_less(julian, 0x42a00000u))
        sxlong = soft32_mul(p.dpd, soft32_add(julian, 0x438e8000u));
    sxlong = soft32_mul(sxlong, p.degrad);
    const uint argument = soft32_mul(sinob, soft_glibc_solar_sinf(sxlong));
    const uint declin = soft_glibc_solar_asinf(argument);
    const uint djul = soft32_div(soft32_mul(julian, 0x43b40000u), 0x43b68000u);
    const uint rjul = soft32_mul(djul, p.degrad);
    const uint two = soft32_mul(0x40000000u, rjul);
    uint eccfac = soft32_add(0x3f80039bu, soft32_mul(0x3d0c2b52u, soft_glibc_solar_cosf(rjul)));
    eccfac = soft32_add(eccfac, soft32_mul(0x3aa7c5acu, soft_glibc_solar_sinf(rjul)));
    eccfac = soft32_add(eccfac, soft32_mul(0x3a3c7b46u, soft_glibc_solar_cosf(two)));
    eccfac = soft32_add(eccfac, soft32_mul(0x38a17b0fu, soft_glibc_solar_sinf(two)));
    r.declin_solcon[0] = p.override_declin != 0 ? p.declin : declin;
    r.declin_solcon[1] = soft32_mul(0x44ab4000u, eccfac);
}

kernel void solar_coszen_hrang(
    constant SolarCoszenHrangResources &r [[buffer(0)]],
    constant SolarCoszenHrangParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.count)) return;
    const uint declin = r.declin_solcon[0];
    const uint phase = p.solar_480 != 0 ? 0x3f000000u : 0x3f800000u;
    const uint da = soft32_div(soft32_mul(0x40c90fdbu, soft32_sub(p.julian, phase)), 0x43b68000u);
    const uint two = soft32_mul(0x40000000u, da);
    uint eot = soft32_add(0x389d4952u, soft32_mul(0x3af4d7aeu, soft_glibc_solar_cosf(da)));
    eot = soft32_sub(eot, soft32_mul(0x3d03632cu, soft_glibc_solar_sinf(da)));
    eot = soft32_sub(eot, soft32_mul(0x3c6f73c1u, soft_glibc_solar_cosf(two)));
    eot = soft32_sub(eot, soft32_mul(p.solar_480 != 0 ? 0x3d275148u : 0x3d277c46u, soft_glibc_solar_sinf(two)));
    eot = soft32_mul(eot, 0x43652e14u);
    const uint xt24 = soft32_add(sgr_mod1440(p.xtime), eot);
    const uint local = soft32_add(soft32_add(p.gmt, soft32_div(xt24, 0x42700000u)), soft32_div(r.longitude[gid], 0x41700000u));
    const uint hour_angle = soft32_mul(soft32_mul(0x41700000u, soft32_sub(local, 0x41400000u)), p.degrad);
    const uint latitude = soft32_mul(r.latitude[gid], p.degrad);
    uint cosine = soft32_add(
        soft32_mul(soft_glibc_solar_sinf(latitude), soft_glibc_solar_sinf(declin)),
        soft32_mul(soft32_mul(soft_glibc_solar_cosf(latitude), soft_glibc_solar_cosf(declin)), soft_glibc_solar_cosf(hour_angle)));
    // CUDA fminf/fmaxf select the numeric operand if the other is NaN.
    if (soft32_is_nan(cosine) || soft32_less(cosine, 0xbf800000u)) cosine = 0xbf800000u;
    if (soft32_greater(cosine, 0x3f800000u)) cosine = 0x3f800000u;
    r.coszen[gid] = cosine;
    r.hrang[gid] = hour_angle;
}
