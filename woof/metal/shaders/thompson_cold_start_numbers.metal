#include <metal_stdlib>
#include "cold_start_numbers_bindings.metalh"
#include "cold_start_temperature_bindings.metalh"
#include "cold_start_temperature_scatter_bindings.metalh"
#include "cold_start_seed_compact_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_libm64.metalh"

using namespace metal;

static inline bool cs_seed(uint mass, uint number);

kernel void cold_start_seed_compact(
    constant ColdStartSeedCompactResources &r [[buffer(0)]],
    constant ColdStartSeedCompactParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    if (!cs_seed(as_type<uint>(r.mass[tid]), as_type<uint>(r.number[tid]))) return;
    device atomic_int *counter = reinterpret_cast<device atomic_int *>(r.count);
    const int slot = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
    r.indices[slot] = long(p.base + int(tid));
}

kernel void cold_start_temperature_scatter(
    constant ColdStartTemperatureScatterResources &r [[buffer(0)]],
    constant ColdStartTemperatureScatterParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    const long cell = r.indices[tid];
    if (cell >= 0 && cell < long(p.n_total)) r.full_temperature[cell] = r.temperature[tid];
}

kernel void cold_start_temperature(
    constant ColdStartTemperatureResources &r [[buffer(0)]],
    constant ColdStartTemperatureParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    const long cell = r.indices[tid];
    const ulong ratio = soft64_div(r.pressure[cell], r.thermo_constants[0]);
    const ulong power = soft64_plm_pow(ratio, r.thermo_constants[1]);
    const ulong value = soft64_mul(r.theta[cell], power);
    r.temperature[tid] = as_type<float>(soft64_to_f32_bits(value));
}

static inline ulong cs_widen_f32(uint bits) { return soft64_from_f32_bits(bits); }
static inline uint cs_narrow_f64(ulong bits) { return soft64_to_f32_bits(bits); }
static inline uint cs_mul(uint a, uint b)
{ return cs_narrow_f64(soft64_mul(cs_widen_f32(a), cs_widen_f32(b))); }
static inline uint cs_div(uint a, uint b)
{ return cs_narrow_f64(soft64_div(cs_widen_f32(a), cs_widen_f32(b))); }
static inline uint cs_add(uint a, uint b)
{ return cs_narrow_f64(soft64_add(cs_widen_f32(a), cs_widen_f32(b))); }
static inline uint cs_sub(uint a, uint b)
{ return cs_narrow_f64(soft64_sub(cs_widen_f32(a), cs_widen_f32(b))); }
static inline bool cs_nonpositive(uint bits)
{
    const uint mag = bits & 0x7fffffffu;
    return mag == 0u || ((bits & 0x80000000u) != 0u && mag <= 0x7f800000u);
}
static inline bool cs_seed(uint mass, uint number)
{
    const uint ma = mass & 0x7fffffffu;
    const uint na = number & 0x7fffffffu;
    return (mass & 0x80000000u) == 0u && ma != 0u && ma <= 0x7f800000u
        && (na == 0u || ((number & 0x80000000u) != 0u && na <= 0x7f800000u));
}
static inline uint cs_gfk_pow(uint x, uint y) { return soft_gfk_pow_bits(x, y); }
static inline ulong cs_mul64(ulong a, ulong b) { return soft64_mul(a, b); }
static inline ulong cs_div64(ulong a, ulong b) { return soft64_div(a, b); }
static inline ulong cs_d32(uint bits) { return soft64_from_f32_bits(bits); }

// `number` is a working copy owned by the caller. Non-seeded cells retain
// their original value; the two int32 outputs are compact-compatible per-cell
// flags used for validation, receipt construction, and sparse cloud repair.
kernel void cold_start_numbers(
    constant ColdStartNumbersResources &r [[buffer(0)]],
    constant ColdStartNumbersParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    const uint q = as_type<uint>(r.mass[tid]);
    const uint old = as_type<uint>(r.number[tid]);
    const uint alt = as_type<uint>(r.alt[tid]);
    const bool seeded_cell = cs_seed(q, old);
    r.seeded[tid] = seeded_cell ? 1 : 0;
    const ulong r1bits = r.constants[4];
    const uint r1 = soft64_to_f32_bits(r1bits);
    const bool offender = seeded_cell && (q > r1)
        && cs_nonpositive(old);
    r.offender[tid] = offender ? 1 : 0;
    if (!seeded_cell) {
        r.final_number[tid] = as_type<float>(old);
        r.seed_volume[tid] = 0.0f;
        return;
    }

    const uint rho = cs_div(0x3f800000u, alt);
    const uint v = cs_mul(q, rho);
    const uint t = as_type<uint>(r.temperature[tid]);
    uint seeded = 0u;

    if (p.species == 0) { // make_RainNumber
        uint n0 = 0x4af42400u; // float(8.0e6)
        if (as_type<float>(t) <= 271.15f) n0 = 0x4e3ebc20u; // float(8.0e8)
        else if (as_type<float>(t) < 273.15f) {
            const uint exponent = cs_sub(0x438b9333u, t); // 279.15 - T
            n0 = cs_mul(0x41000000u, cs_gfk_pow(0x41200000u, exponent));
        }
        const ulong numerator = soft64_mul(
            soft64_mul(cs_d32(n0), r.constants[0]), 0x4018000000000000ul);
        ulong lambda = soft64_sqrt(soft64_sqrt(
            soft64_div(numerator, cs_d32(v))));
        const ulong nv = soft64_div(
            soft64_mul(soft64_mul(soft64_mul(cs_d32(cs_div(v, soft64_to_f32_bits(r.constants[1]))),
                                               lambda), lambda), lambda),
            r.constants[0]);
        seeded = soft64_to_f32_bits(nv);
    } else if (p.species == 1) { // make_IceNumber
        const uint offset = cs_sub(t, 0x43330000u); // T - 179
        const int raw_index = int(as_type<float>(offset));
        const int idx = max(1, min(raw_index, 94));
        const int ti = int(as_type<float>(t));
        const uint corr = cs_sub(t, as_type<uint>(float(ti)));
        const uint re = cs_add(
            cs_mul(as_type<uint>(r.retab[idx - 1]), cs_sub(0x3f800000u, corr)),
            cs_mul(as_type<uint>(r.retab[idx]), corr));
        const uint de = cs_mul(cs_mul(0x40000000u, re), 0x358637bdu);
        const ulong lambda = cs_d32(cs_div(0x40400000u, de));
        const ulong nv = soft64_div(
            soft64_mul(soft64_mul(soft64_mul(cs_d32(v), lambda), lambda), lambda),
            r.constants[2]);
        seeded = soft64_to_f32_bits(nv);
    } else { // make_DropletNumber
        const uint aerosol = as_type<uint>(r.aerosol[tid]);
        const uint nw = cs_mul(aerosol, rho);
        uint dc;
        int nu;
        if (cs_nonpositive(nw)) {
            const uint xland = as_type<uint>(r.xland[tid]);
            const bool ocean = as_type<float>(cs_sub(xland, 0x3fc00000u)) > 0.0f;
            dc = ocean ? 0x378e9b39u : 0x37388ca4u; // 17 or 11 um
            nu = ocean ? 12 : 4;
        } else {
            const float n_float = as_type<float>(nw);
            const float bounded_float = fmax(99.0e6f, fmin(n_float, 5.0e10f));
            const uint bounded = as_type<uint>(bounded_float);
            const ulong ratio = cs_d32(cs_div(0x50ba43b7u, bounded));
            const ulong nearest = soft64_add(ratio, 0x3fe0000000000000ul);
            const Soft64I32Result rounded = soft64_to_i32_rz(nearest);
            nu = max(2, min(15, rounded.valid ? rounded.value : 15));
            const uint scaled = cs_mul(bounded, 0x3089705fu);
            const float bounded_decade = fmin(as_type<float>(scaled), 10.0f);
            const uint x = cs_sub(as_type<uint>(fmax(1.0f, bounded_decade)),
                                  0x3f800000u);
            const uint numerator = cs_mul(x, 0x41a00000u);
            const uint slope = cs_div(numerator, 0x41100000u);
            dc = cs_mul(cs_sub(0x41f00000u, slope), 0x358637bdu);
        }
        const ulong lambda = soft64_div(
            soft64_add(0x4010000000000000ul, soft64_from_i32(nu)), cs_d32(dc));
        const uint ratio = cs_div(v, as_type<uint>(r.gratio[nu - 1]));
        const ulong nv = soft64_div(
            soft64_mul(soft64_mul(soft64_mul(cs_d32(ratio), lambda), lambda), lambda),
            r.constants[0]);
        seeded = soft64_to_f32_bits(nv);
    }

    r.seed_volume[tid] = as_type<float>(seeded);
    uint result = cs_div(seeded, rho);
    if (offender && p.species != 2) {
        const ulong density = soft64_div(0x3ff0000000000000ul, cs_d32(alt));
        const uint m = soft64_to_f32_bits(soft64_mul(cs_d32(q), density));
        const uint initial_num = soft64_to_f32_bits(
            soft64_mul(cs_d32(result), density));
        const uint r2 = soft64_to_f32_bits(r.constants[5]);
        const bool num_nan = ((initial_num & 0x7f800000u) == 0x7f800000u)
                          && ((initial_num & 0x007fffffu) != 0u);
        uint num = initial_num;
        if (!num_nan && as_type<float>(num) < as_type<float>(r2)) num = r2;
        if (p.species == 0) {
            const uint pref = cs_mul(as_type<uint>(as_type<float>(soft64_to_f32_bits(r.constants[7]))), m);
            if (!num_nan && as_type<float>(num) <= as_type<float>(r2))
                num = soft64_to_f32_bits(soft64_div(soft64_mul(cs_d32(pref), r.constants[8]), r.constants[9]));
            const uint arg = cs_div(cs_mul(cs_mul(as_type<uint>(as_type<float>(soft64_to_f32_bits(r.constants[9]))),
                                                  0x40c00000u), num), m);
            const uint lam = cs_gfk_pow(arg, as_type<uint>(as_type<float>(soft64_to_f32_bits(r.constants[6]))));
            const uint mvd = soft64_to_f32_bits(soft64_div(r.constants[10], cs_d32(lam)));
            if (as_type<float>(mvd) > as_type<float>(soft64_to_f32_bits(r.constants[11])))
                num = soft64_to_f32_bits(soft64_div(soft64_mul(cs_d32(pref), r.constants[12]), r.constants[9]));
            if (as_type<float>(mvd) < as_type<float>(soft64_to_f32_bits(r.constants[13])))
                num = soft64_to_f32_bits(soft64_div(soft64_mul(cs_d32(pref), r.constants[14]), r.constants[9]));
        } else {
            const uint pref = cs_div(cs_mul(as_type<uint>(as_type<float>(soft64_to_f32_bits(r.constants[7]))), m),
                                     as_type<uint>(as_type<float>(soft64_to_f32_bits(r.constants[15]))));
            const ulong small = soft64_less(r.constants[16], soft64_mul(cs_d32(pref), r.constants[17]))
                ? r.constants[16] : soft64_mul(cs_d32(pref), r.constants[17]);
            if (as_type<float>(num) <= as_type<float>(soft64_to_f32_bits(r.constants[5])))
                num = soft64_to_f32_bits(small);
            const uint arg = cs_div(cs_mul(cs_mul(as_type<uint>(as_type<float>(soft64_to_f32_bits(r.constants[15]))),
                                                  0x40c00000u), num), m);
            const uint lam = cs_gfk_pow(arg, as_type<uint>(as_type<float>(soft64_to_f32_bits(r.constants[6]))));
            const ulong di64 = soft64_mul(0x4010000000000000ul,
                                          soft64_div(0x3ff0000000000000ul, cs_d32(lam)));
            const uint di = soft64_to_f32_bits(di64);
            if (as_type<float>(di) < as_type<float>(soft64_to_f32_bits(r.constants[18])))
                num = soft64_to_f32_bits(small);
            else if (as_type<float>(di) > as_type<float>(soft64_to_f32_bits(r.constants[19])))
                num = soft64_to_f32_bits(soft64_mul(cs_d32(pref), r.constants[20]));
        }
        result = soft64_to_f32_bits(soft64_div(cs_d32(num), density));
    }
    r.final_number[tid] = as_type<float>(result);
}
