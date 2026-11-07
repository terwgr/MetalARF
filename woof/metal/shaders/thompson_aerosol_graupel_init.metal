// Classic aerosol graupel entry number, mp_gt_driver's raw-qv density.
// Every REAL(4)/DOUBLE operation follows thompson_aerosol_state.cu;
// integer words preserve RN rounding and subnormals across both precisions.
#include <metal_stdlib>
#include "thompson_aerosol_graupel_init_bindings.metalh"
#include "thompson_aerosol_exner_libm.metalh"
#include "soft_gfk_log.metalh"
#include "soft_glibc_pow64.metalh"
using namespace metal;

static inline uint agi_max(uint a, uint b) {
    if (soft32_is_nan(a)) return b;
    if (soft32_is_nan(b)) return a;
    if (soft32_is_zero(a) && soft32_is_zero(b)) return (a & 0x80000000u) ? b : a;
    return soft32_greater(a, b) ? a : b;
}
static inline uint agi_min(uint a, uint b) {
    if (soft32_is_nan(a)) return b;
    if (soft32_is_nan(b)) return a;
    if (soft32_is_zero(a) && soft32_is_zero(b)) return (a & 0x80000000u) ? a : b;
    return soft32_less(a, b) ? a : b;
}
// thompson_aa_log10f: its three float32 multiply/add boundaries surround
// the pinned gfk_log call; log(x)/log(10) changes these result words.
static inline uint agi_log10f(uint x) {
    uint hx = x;
    int k = 0;
    if (hx < 0x00800000u || (hx & 0x80000000u)) {
        if ((hx & 0x7fffffffu) == 0u) return 0xff800000u;
        if (hx & 0x80000000u) return 0x7fc00000u;
        k -= 25;
        hx = soft32_mul(x, 0x4c000000u);
    }
    if (hx > 0x7f7fffffu) return soft32_add(x, x);
    k += int(hx >> 23) - 127;
    const int i = int((uint(k) & 0x80000000u) >> 31);
    hx = (hx & 0x007fffffu) | (uint(0x7f - i) << 23);
    const uint y = soft64_to_f32_bits(soft64_from_i32(k + i));
    const uint z = soft32_add(soft32_mul(0x3ede5bd9u, soft_gfk_log(hx)),
                              soft32_mul(y, 0x355427dbu));
    return soft32_add(z, soft32_mul(y, 0x3e9a2080u));
}

kernel void thompson_aa_graupel_number_init(
    constant ThompsonAaGraupelNumberInitResources &r [[buffer(0)]],
    constant ThompsonAaGraupelNumberInitParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const uint q = r.qg[idx];
    if (!soft32_greater(q, 0x2b8cbcccu)) {
        r.graupel_number[idx] = 0u;
        return;
    }
    const uint rho = soft32_div(soft32_mul(0x3f1f3b64u, r.pressure[idx]),
        soft32_mul(soft32_mul(0x438f851fu, r.temperature[idx]), soft32_add(r.qv[idx], 0x3f1f3b64u)));
    const uint ygra1 = agi_log10f(agi_max(0x3089705fu, soft32_mul(q, rho)));
    uint zans1 = soft32_add(0x40400000u, soft32_mul(soft32_div(0x40000000u, 0x40e00000u), soft32_add(ygra1, 0x41000000u)));
    zans1 = agi_max(0x40000000u, agi_min(zans1, 0x40c00000u));
    const ulong n0_exp = soft64_from_f32_bits(thompson_exner_powf_bits(0x41200000u, zans1));
    const ulong am_g5 = soft64_from_f32_bits(0x43517084u);
    const ulong lam_exp = soft_glibc_pow64(soft64_div(soft64_mul(soft64_mul(n0_exp, am_g5),
        soft64_from_f32_bits(0x40c00000u)), soft64_from_f32_bits(soft32_mul(rho, q))),
        soft64_from_f32_bits(0x3e800000u));
    const uint factor = thompson_exner_powf_bits(soft32_mul(soft32_mul(0x40c00000u, 0x3f800000u), 0x3e2aaaabu), 0x3eaaaaabu);
    const ulong lamg = soft64_mul(lam_exp, soft64_from_f32_bits(factor));
    const uint prefix = soft32_mul(soft32_mul(soft32_mul(0x3f800000u, 0x3e2aaaabu), rho), q);
    const uint number = soft64_to_f32_bits(soft64_div(soft64_mul(soft64_from_f32_bits(prefix),
        soft_glibc_pow64(lamg, soft64_from_f32_bits(0x40400000u))), am_g5));
    r.graupel_number[idx] = agi_max(0x358637bdu, soft32_div(number, rho));
}
