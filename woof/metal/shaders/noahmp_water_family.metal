#include <metal_stdlib>
#include "noahmp_water_family_bindings.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_pow.metalh"

using namespace metal;
#pragma clang fp contract(off)

// NON-PRODUCTION DRAFT ONLY. This packet-local software binary64/libm path
// overlaps the shared binary64 and pinned-exp owners. It is disabled and must
// not be exposed through descriptors or callers before their interface and
// arithmetic review are complete. Retain it only for reconciliation.
#if 0

struct W64 {
    ulong bits;
};

struct W128 {
    ulong hi;
    ulong lo;
};

static constexpr ulong W64_SIGN = 0x8000000000000000ul;
static constexpr ulong W64_FRAC = 0x000ffffffffffffful;
static constexpr ulong W64_EXP = 0x7ff0000000000000ul;
static constexpr ulong W64_QNAN = 0x7ff8000000000000ul;

// Bitwise copies of glibc 2.39 __exp2f_data and the pinned expf limits.
// Keep table values as integers: converting source decimal literals would
// introduce an unrelated host/compiler rounding step.
constant ulong W_EXP2_TAB[32] = {
    0x3FF0000000000000ul, 0x3FEFD9B0D3158574ul, 0x3FEFB5586CF9890Ful, 0x3FEF9301D0125B51ul,
    0x3FEF72B83C7D517Bul, 0x3FEF54873168B9AAul, 0x3FEF387A6E756238ul, 0x3FEF1E9DF51FDEE1ul,
    0x3FEF06FE0A31B715ul, 0x3FEEF1A7373AA9CBul, 0x3FEEDEA64C123422ul, 0x3FEECE086061892Dul,
    0x3FEEBFDAD5362A27ul, 0x3FEEB42B569D4F82ul, 0x3FEEAB07DD485429ul, 0x3FEEA47EB03A5585ul,
    0x3FEEA09E667F3BCDul, 0x3FEE9F75E8EC5F74ul, 0x3FEEA11473EB0187ul, 0x3FEEA589994CCE13ul,
    0x3FEEACE5422AA0DBul, 0x3FEEB737B0CDC5E5ul, 0x3FEEC49182A3F090ul, 0x3FEED503B23E255Dul,
    0x3FEEE89F995AD3ADul, 0x3FEEFF76F2FB5E47ul, 0x3FEF199BDD85529Cul, 0x3FEF3720DCEF9069ul,
    0x3FEF5818DCFBA487ul, 0x3FEF7C97337B9B5Ful, 0x3FEFA4AFA2A490DAul, 0x3FEFD0765B6E4540ul
};
constant ulong W_EXP2_MISC[3] = {
    0x42E8000000000000ul, 0x4338000000000000ul, 0x40471547652B82FEul
};
constant ulong W_EXP2_POLY[3] = {
    0x3FAC6AF84B912394ul, 0x3FCEBFCE50FAC4F3ul, 0x3FE62E42FF0C52D6ul
};
constant ulong W_EXP2_POLY_SCALED[3] = {
    0x3EBC6AF84B912394ul, 0x3F2EBFCE50FAC4F3ul, 0x3F962E42FF0C52D6ul
};
constant ulong W_LIMITS[3] = {
    0x40562E42E0000000ul, 0xC059FE3680000000ul, 0x405FFFFFFFD1D571ul
};
constant ulong W_POW_LOG2_TAB[32] = {
    0x3FF661EC79F8F3BEul, 0xBFDEFEC65B963019ul,
    0x3FF571ED4AAF883Dul, 0xBFDB0B6832D4FCA4ul,
    0x3FF49539F0F010B0ul, 0xBFD7418B0A1FB77Bul,
    0x3FF3C995B0B80385ul, 0xBFD39DE91A6DCF7Bul,
    0x3FF30D190C8864A5ul, 0xBFD01D9BF3F2B631ul,
    0x3FF25E227B0B8EA0ul, 0xBFC97C1D1B3B7AF0ul,
    0x3FF1BB4A4A1A343Ful, 0xBFC2F9E393AF3C9Ful,
    0x3FF12358F08AE5BAul, 0xBFB960CBBF788D5Cul,
    0x3FF0953F419900A7ul, 0xBFAA6F9DB6475FCEul,
    0x3FF0000000000000ul, 0x0000000000000000ul,
    0x3FEE608CFD9A47ACul, 0x3FB338CA9F24F53Dul,
    0x3FECA4B31F026AA0ul, 0x3FC476A9543891BAul,
    0x3FEB2036576AFCE6ul, 0x3FCE840B4AC4E4D2ul,
    0x3FE9C2D163A1AA2Dul, 0x3FD40645F0C6651Cul,
    0x3FE886E6037841EDul, 0x3FD88E9C2C1B9FF8ul,
    0x3FE767DCF5534862ul, 0x3FDCE0A44EB17BCCul
};
constant ulong W_POW_LOG2_POLY[5] = {
    0x3FD27616C9496E0Bul, 0xBFD71969A075C67Aul,
    0x3FDEC70A6CA7BADDul, 0xBFE7154748BEF6C8ul,
    0x3FF71547652AB82Bul
};

static inline bool w128_zero(W128 a) { return a.hi == 0ul && a.lo == 0ul; }
static inline int w128_msb(W128 a) {
    if (a.hi != 0ul) return 127 - int(clz(a.hi));
    if (a.lo != 0ul) return 63 - int(clz(a.lo));
    return -1;
}
static inline int w128_cmp(W128 a, W128 b) {
    if (a.hi < b.hi) return -1;
    if (a.hi > b.hi) return 1;
    if (a.lo < b.lo) return -1;
    if (a.lo > b.lo) return 1;
    return 0;
}
static inline W128 w128_add(W128 a, W128 b) {
    W128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + ulong(r.lo < a.lo);
    return r;
}
static inline W128 w128_sub(W128 a, W128 b) {
    W128 r;
    r.lo = a.lo - b.lo;
    r.hi = a.hi - b.hi - ulong(a.lo < b.lo);
    return r;
}
static inline W128 w128_shl(W128 a, uint n) {
    if (n == 0u) return a;
    if (n >= 128u) return W128{0ul, 0ul};
    if (n >= 64u) return W128{a.lo << (n - 64u), 0ul};
    return W128{(a.hi << n) | (a.lo >> (64u - n)), a.lo << n};
}
static inline bool w128_low_nonzero(W128 a, uint count) {
    if (count == 0u) return false;
    if (count < 64u) return (a.lo & ((1ul << count) - 1ul)) != 0ul;
    if (count == 64u) return a.lo != 0ul;
    if (count < 128u) {
        const uint high_count = count - 64u;
        return a.lo != 0ul || (a.hi & ((1ul << high_count) - 1ul)) != 0ul;
    }
    return !w128_zero(a);
}
static inline bool w128_bit(W128 a, uint bit) {
    if (bit >= 128u) return false;
    if (bit >= 64u) return ((a.hi >> (bit - 64u)) & 1ul) != 0ul;
    return ((a.lo >> bit) & 1ul) != 0ul;
}
static inline W128 w128_shr_jam(W128 a, uint n) {
    if (n == 0u) return a;
    if (n >= 128u) return W128{0ul, ulong(!w128_zero(a))};
    W128 r;
    if (n >= 64u) {
        const uint s = n - 64u;
        r.hi = 0ul;
        r.lo = s == 0u ? a.hi : (a.hi >> s);
    } else {
        r.hi = a.hi >> n;
        r.lo = (a.lo >> n) | (a.hi << (64u - n));
    }
    if (w128_low_nonzero(a, n)) r.lo |= 1ul;
    return r;
}
static inline ulong w128_shr_to_ulong(W128 a, uint n) {
    if (n == 0u) return a.lo;
    if (n >= 128u) return 0ul;
    if (n >= 64u) return a.hi >> (n - 64u);
    return (a.lo >> n) | (a.hi << (64u - n));
}
static inline bool w128_round_increment(W128 a, uint shift, ulong q) {
    if (shift == 0u) return false;
    const bool guard = w128_bit(a, shift - 1u);
    const bool sticky = shift > 1u && w128_low_nonzero(a, shift - 1u);
    return guard && (sticky || ((q & 1ul) != 0ul));
}
static inline ulong w128_round_shift(W128 a, int shift) {
    if (shift <= 0) return a.lo << uint(-shift);
    const uint s = uint(shift);
    ulong q = w128_shr_to_ulong(a, s);
    if (w128_round_increment(a, s, q)) ++q;
    return q;
}

// Round an unsigned integer significand times 2^exp2 to an IEEE binary format.
// All binary64 arithmetic below uses this single round-to-nearest-even path.
static inline ulong w_round_pack(bool sign, W128 mag, int exp2,
                                 uint precision, int emin, int emax,
                                 int bias) {
    const uint sign_position = precision == 53u ? 63u : 31u;
    const ulong sign_bit = sign ? (1ul << sign_position) : 0ul;
    const uint exp_bits = precision == 53u ? 11u : 8u;
    const uint frac_bits = precision - 1u;
    const ulong exp_mask = (1ul << exp_bits) - 1ul;
    const ulong hidden = 1ul << frac_bits;
    const int top = w128_msb(mag);
    if (top < 0) return sign_bit;
    int top_exp = top + exp2;
    if (top_exp > emax)
        return sign_bit | (exp_mask << frac_bits);

    if (top_exp >= emin) {
        const int shift = top - int(frac_bits);
        ulong q = w128_round_shift(mag, shift);
        if (q >= (hidden << 1u)) {
            q >>= 1u;
            ++top_exp;
        }
        if (top_exp > emax)
            return sign_bit | (exp_mask << frac_bits);
        return sign_bit | (ulong(top_exp + bias) << frac_bits)
             | (q - hidden);
    }

    const int unit_exp = emin - int(frac_bits);
    const int shift = unit_exp - exp2;
    const ulong q = w128_round_shift(mag, shift);
    if (q >= hidden) return sign_bit | (1ul << frac_bits);
    return sign_bit | q;
}

static inline bool w64_nan(W64 a) {
    return (a.bits & W64_EXP) == W64_EXP && (a.bits & W64_FRAC) != 0ul;
}
static inline bool w64_inf(W64 a) {
    return (a.bits & ~W64_SIGN) == W64_EXP;
}
static inline bool w64_zero(W64 a) { return (a.bits & ~W64_SIGN) == 0ul; }
static inline W64 w64_quiet(W64 a) {
    a.bits |= 0x0008000000000000ul;
    return a;
}
static inline W64 w64_from_f32(float x) {
    const uint b = as_type<uint>(x);
    const bool sign = (b & 0x80000000u) != 0u;
    const uint e = (b >> 23u) & 0xffu;
    const uint f = b & 0x007fffffu;
    if (e == 0xffu) {
        if (f == 0u) return W64{(sign ? W64_SIGN : 0ul) | W64_EXP};
        return W64{(sign ? W64_SIGN : 0ul) | W64_QNAN
                  | (ulong(f) << 29u)};
    }
    const ulong sig = e == 0u ? ulong(f) : ulong(0x00800000u | f);
    const int exp2 = e == 0u ? -149 : int(e) - 127 - 23;
    return W64{w_round_pack(sign, W128{0ul, sig}, exp2, 53u,
                            -1022, 1023, 1023)};
}
static inline W128 w64_sig(W64 a) {
    const ulong e = (a.bits >> 52u) & 0x7fful;
    const ulong f = a.bits & W64_FRAC;
    return W128{0ul, e == 0ul ? f : (0x0010000000000000ul | f)};
}
static inline int w64_exp2(W64 a) {
    const int e = int((a.bits >> 52u) & 0x7fful);
    return e == 0 ? -1074 : e - 1075;
}
static inline int w64_top_exp(W64 a) {
    return w128_msb(w64_sig(a)) + w64_exp2(a);
}
static inline int w64_cmp_abs(W64 a, W64 b) {
    const ulong aa = a.bits & ~W64_SIGN;
    const ulong bb = b.bits & ~W64_SIGN;
    return aa < bb ? -1 : (aa > bb ? 1 : 0);
}
static inline int w64_cmp(W64 a, W64 b) {
    const bool sa = (a.bits & W64_SIGN) != 0ul;
    const bool sb = (b.bits & W64_SIGN) != 0ul;
    if (sa != sb) return sa ? -1 : 1;
    const int mag = w64_cmp_abs(a, b);
    return sa ? -mag : mag;
}
static inline W128 w64_scaled(W128 sig, int exp2, int base_exp) {
    const int shift = exp2 - base_exp;
    return shift >= 0 ? w128_shl(sig, uint(shift))
                      : w128_shr_jam(sig, uint(-shift));
}

static inline W64 w64_add(W64 a, W64 b) {
    if (w64_nan(a)) return w64_quiet(a);
    if (w64_nan(b)) return w64_quiet(b);
    if (w64_inf(a) || w64_inf(b)) {
        if (w64_inf(a) && w64_inf(b)
                && ((a.bits ^ b.bits) & W64_SIGN) != 0ul)
            return W64{W64_QNAN};
        return w64_inf(a) ? a : b;
    }
    if (w64_zero(a) && w64_zero(b))
        return W64{(a.bits & b.bits) & W64_SIGN};
    if (w64_zero(a)) return b;
    if (w64_zero(b)) return a;

    const bool sa = (a.bits & W64_SIGN) != 0ul;
    const bool sb = (b.bits & W64_SIGN) != 0ul;
    const int base_exp = max(w64_top_exp(a), w64_top_exp(b)) - 120;
    const W128 xa = w64_scaled(w64_sig(a), w64_exp2(a), base_exp);
    const W128 xb = w64_scaled(w64_sig(b), w64_exp2(b), base_exp);
    W128 sum;
    bool sign;
    if (sa == sb) {
        sum = w128_add(xa, xb);
        sign = sa;
    } else {
        const int cmp = w128_cmp(xa, xb);
        if (cmp == 0) return W64{0ul};
        if (cmp > 0) { sum = w128_sub(xa, xb); sign = sa; }
        else { sum = w128_sub(xb, xa); sign = sb; }
    }
    return W64{w_round_pack(sign, sum, base_exp, 53u, -1022, 1023, 1023)};
}
static inline W64 w64_sub(W64 a, W64 b) {
    b.bits ^= W64_SIGN;
    return w64_add(a, b);
}

static inline W128 w128_mul53(ulong a, ulong b) {
    const ulong a0 = a & 0xfffffffful, a1 = a >> 32u;
    const ulong b0 = b & 0xfffffffful, b1 = b >> 32u;
    const ulong p00 = a0 * b0;
    const ulong p01 = a0 * b1;
    const ulong p10 = a1 * b0;
    const ulong p11 = a1 * b1;
    W128 r = W128{0ul, p00};
    r = w128_add(r, W128{p01 >> 32u, p01 << 32u});
    r = w128_add(r, W128{p10 >> 32u, p10 << 32u});
    r.hi += p11;
    return r;
}
static inline W64 w64_mul(W64 a, W64 b) {
    if (w64_nan(a)) return w64_quiet(a);
    if (w64_nan(b)) return w64_quiet(b);
    const bool sign = ((a.bits ^ b.bits) & W64_SIGN) != 0ul;
    if ((w64_inf(a) && w64_zero(b)) || (w64_inf(b) && w64_zero(a)))
        return W64{W64_QNAN};
    if (w64_inf(a) || w64_inf(b))
        return W64{(sign ? W64_SIGN : 0ul) | W64_EXP};
    if (w64_zero(a) || w64_zero(b)) return W64{sign ? W64_SIGN : 0ul};
    const ulong sa = w64_sig(a).lo, sb = w64_sig(b).lo;
    const W128 product = w128_mul53(sa, sb);
    return W64{w_round_pack(sign, product, w64_exp2(a) + w64_exp2(b),
                            53u, -1022, 1023, 1023)};
}
static inline W64 w64_fma(W64 a, W64 b, W64 c) {
    if (w64_nan(a)) return w64_quiet(a);
    if (w64_nan(b)) return w64_quiet(b);
    if (w64_nan(c)) return w64_quiet(c);
    const bool ps = ((a.bits ^ b.bits) & W64_SIGN) != 0ul;
    if ((w64_inf(a) && w64_zero(b)) || (w64_inf(b) && w64_zero(a)))
        return W64{W64_QNAN};
    if (w64_inf(a) || w64_inf(b)) {
        const W64 product_inf{(ps ? W64_SIGN : 0ul) | W64_EXP};
        if (w64_inf(c) && ((c.bits ^ product_inf.bits) & W64_SIGN) != 0ul)
            return W64{W64_QNAN};
        return product_inf;
    }
    if (w64_inf(c)) return c;
    if (w64_zero(a) || w64_zero(b))
        return w64_add(W64{ps ? W64_SIGN : 0ul}, c);

    const W128 product = w128_mul53(w64_sig(a).lo, w64_sig(b).lo);
    const int product_exp = w64_exp2(a) + w64_exp2(b);
    if (w64_zero(c))
        return W64{w_round_pack(ps, product, product_exp,
                                53u, -1022, 1023, 1023)};

    const bool cs = (c.bits & W64_SIGN) != 0ul;
    const int product_top = w128_msb(product) + product_exp;
    const int c_top = w64_top_exp(c);
    const int base_exp = max(product_top, c_top) - 120;
    const W128 xp = w64_scaled(product, product_exp, base_exp);
    const W128 xc = w64_scaled(w64_sig(c), w64_exp2(c), base_exp);
    W128 sum;
    bool sign;
    if (ps == cs) {
        sum = w128_add(xp, xc);
        sign = ps;
    } else {
        const int cmp = w128_cmp(xp, xc);
        if (cmp == 0) return W64{0ul};
        if (cmp > 0) { sum = w128_sub(xp, xc); sign = ps; }
        else { sum = w128_sub(xc, xp); sign = cs; }
    }
    return W64{w_round_pack(sign, sum, base_exp, 53u, -1022, 1023, 1023)};
}

static inline float w64_to_f32(W64 value) {
    const bool sign = (value.bits & W64_SIGN) != 0ul;
    const ulong e = (value.bits >> 52u) & 0x7fful;
    const ulong f = value.bits & W64_FRAC;
    uint out;
    if (e == 0x7fful) {
        if (f == 0ul) out = (sign ? 0x80000000u : 0u) | 0x7f800000u;
        else {
            uint payload = uint(f >> 29u) | 0x00400000u;
            out = (sign ? 0x80000000u : 0u) | 0x7f800000u
                | (payload & 0x007fffffu);
        }
        return as_type<float>(out);
    }
    const W128 sig = w64_sig(value);
    const ulong bits = w_round_pack(sign, sig, w64_exp2(value),
                                    24u, -126, 127, 127);
    return as_type<float>(uint(bits));
}

static inline ulong w64_to_u64_trunc(W64 value) {
    const W128 sig = w64_sig(value);
    const int exp2 = w64_exp2(value);
    if (exp2 >= 0) return sig.lo << uint(exp2);
    const uint shift = uint(-exp2);
    return w128_shr_to_ulong(sig, shift);
}

static inline W64 w64_from_f32_bits(uint bits) {
    return w64_from_f32(as_type<float>(bits));
}
static inline W64 w64_from_i32(int value) {
    const bool sign = value < 0;
    const ulong mag = sign ? ulong(uint(-(value + 1))) + 1ul : ulong(uint(value));
    return W64{w_round_pack(sign, W128{0ul, mag}, 0, 53u,
                            -1022, 1023, 1023)};
}

static inline float w_glibc_expf(float x) {
    const uint ix = as_type<uint>(x);
    const uint abstop = (ix >> 20u) & 0x7ffu;
    if (abstop >= 0x42bu) {
        if (ix == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return x + x;
        const W64 xd = w64_from_f32(x);
        if (w64_cmp(xd, W64{W_LIMITS[0]}) > 0)
            return as_type<float>(0x7f800000u);
        if (w64_cmp(xd, W64{W_LIMITS[1]}) < 0)
            return 0.0f;
    }

    const W64 shift = W64{W_EXP2_MISC[1]};
    const W64 xd = w64_from_f32(x);
    const W64 z = w64_mul(W64{W_EXP2_MISC[2]}, xd);
    W64 kd = w64_add(z, shift);
    const ulong ki = w64_to_u64_trunc(kd);
    kd = w64_sub(kd, shift);
    const W64 r = w64_sub(z, kd);

    const ulong t = W_EXP2_TAB[ki & 31ul] + (ki << 47u);
    const W64 s = W64{t};
    const W64 c0 = W64{W_EXP2_POLY_SCALED[0]};
    const W64 c1 = W64{W_EXP2_POLY_SCALED[1]};
    const W64 c2 = W64{W_EXP2_POLY_SCALED[2]};
    const W64 z2 = w64_fma(c0, r, c1);
    const W64 r2 = w64_mul(r, r);
    W64 y = w64_fma(c2, r, W64{0x3ff0000000000000ul});
    y = w64_fma(z2, r2, y);
    y = w64_mul(y, s);
    return w64_to_f32(y);
}

static inline W64 w_powf_log2_inline(uint ix) {
    const uint tmp = ix - 0x3f330000u;
    const uint i = (tmp >> 19u) & 15u;
    const uint top = tmp & 0xff800000u;
    const uint iz = ix - top;
    const int k = int(top) >> 23;
    const W64 invc = W64{W_POW_LOG2_TAB[2u * i]};
    const W64 logc = W64{W_POW_LOG2_TAB[2u * i + 1u]};
    const W64 z = w64_from_f32(as_type<float>(iz));
    const W64 r = w64_fma(z, invc, W64{W64_SIGN | 0x3ff0000000000000ul});
    const W64 y0 = w64_add(logc, w64_from_i32(k));
    const W64 a0 = W64{W_POW_LOG2_POLY[0]};
    const W64 a1 = W64{W_POW_LOG2_POLY[1]};
    const W64 a2 = W64{W_POW_LOG2_POLY[2]};
    const W64 a3 = W64{W_POW_LOG2_POLY[3]};
    const W64 a4 = W64{W_POW_LOG2_POLY[4]};
    const W64 r2 = w64_mul(r, r);
    W64 y = w64_fma(a0, r, a1);
    const W64 p = w64_fma(a2, r, a3);
    const W64 r4 = w64_mul(r2, r2);
    W64 q = w64_fma(a4, r, y0);
    q = w64_fma(p, r2, q);
    y = w64_fma(y, r4, q);
    return y;
}

static inline float w_powf_exp2_inline(W64 xd, ulong sign_bias) {
    const W64 shift = W64{W_EXP2_MISC[0]};
    W64 kd = w64_add(xd, shift);
    const ulong ki = w64_to_u64_trunc(kd);
    kd = w64_sub(kd, shift);
    const W64 r = w64_sub(xd, kd);
    const ulong ski = ki + sign_bias;
    const ulong t = W_EXP2_TAB[ki & 31ul] + (ski << 47u);
    const W64 s = W64{t};
    const W64 c0 = W64{W_EXP2_POLY[0]};
    const W64 c1 = W64{W_EXP2_POLY[1]};
    const W64 c2 = W64{W_EXP2_POLY[2]};
    const W64 z = w64_fma(c0, r, c1);
    const W64 r2 = w64_mul(r, r);
    W64 y = w64_fma(c2, r, W64{0x3ff0000000000000ul});
    y = w64_fma(z, r2, y);
    y = w64_mul(y, s);
    return w64_to_f32(y);
}

static inline float w_glibc_powf(float x, float y) {
    const uint ix = as_type<uint>(x);
    const uint iy = as_type<uint>(y);
    if (ix == 0x3f800000u) return 1.0f;
    if (ix == 0u && (iy & 0x80000000u) == 0u
            && (iy & 0x7f800000u) != 0x7f800000u && iy != 0u)
        return 0.0f;
    if ((ix - 0x00800000u) >= 0x7f000000u
            || (((2u * iy) - 1u) >= 0xfeffffffu))
        return as_type<float>(0x7fc00000u);

    const W64 logx = w_powf_log2_inline(ix);
    const W64 ylogx = w64_mul(w64_from_f32(y), logx);
    const ulong ab = ylogx.bits;
    const ulong lim = 0x405f800000000000ul >> 47u;
    if (((ab >> 47u) & 0xfffful) >= lim) {
        if (w64_cmp(ylogx, W64{W_LIMITS[2]}) > 0)
            return as_type<float>(0x7f800000u);
        if (w64_cmp(ylogx, W64{0xc062c00000000000ul}) <= 0)
            return 0.0f;
    }
    return w_powf_exp2_inline(ylogx, 0ul);
}
#endif

// Public-caller WATER composition with source snow-grid remapping and pinned
// glibc transcendental helpers for COMPACT and SOILWATER.
constant uint W_NSOIL=4u,W_NSNOW=3u,W_NFULL=7u,W_NSTATE=37u;
constant float W_ZERO=0.0f,W_ONE=1.0f,W_TFRZ=273.16f,W_HVAP=2.5104e6f,W_HSUB=2.8440e6f;
constant float W_CWAT=4.188e6f,W_CICE=2.094e6f,W_HFUS=0.3336e6f,W_DENH2O=1000.0f,W_DENICE=917.0f;
constant float W_MPE=1.0e-6f,W_1000=1000.0f,W_MILLI=0.001f,W_WSLMAX=5000.0f;
constant float W_ONEEM4=1.0e-4f,W_WATMIN=0.01f;
static inline float wmax(float a,float b){return a<b?b:a;}
static inline float wmin(float a,float b){return b<a?b:a;}
static inline float wabs(float a){return fabs(a);}
static inline float wexp(float x){return as_type<float>(soft_gfk_exp(as_type<uint>(x)));}
static inline float wpow(float x,float y){return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x),as_type<uint>(y)));}
static inline float wdf1(float smc,float fcr,float smcmax,float bexp,float dwsat,float dksat,thread float &wcnd){
  float factr=wmax(0.01f,smc/smcmax),expon=bexp+2.0f,w=dwsat*wpow(factr,expon);
  w*=1.0f-fcr;expon=2.0f*bexp+3.0f;wcnd=dksat*wpow(factr,expon)*(1.0f-fcr);return w;
}
static inline float wdf2(float smc,float sice,float smcmax,float bexp,float dwsat,float dksat,thread float &wcnd){
  float factr1=wmin(0.05f/smcmax,wmax(0.01f,smc/smcmax)),factr2=wmax(0.01f,smc/smcmax),expon=bexp+2.0f;
  float w=dwsat*wpow(factr2,expon);
  if(sice>0.0f){float vkwgt=1.0f/(1.0f+wpow(500.0f*sice,3.0f));w=vkwgt*w+(1.0f-vkwgt)*dwsat*wpow(factr1,expon);}
  expon=2.0f*bexp+3.0f;wcnd=dksat*wpow(factr2,expon);return w;
}
static inline void wrosr12(thread const float *a,thread const float *b,thread float *c,thread const float *d,thread float *p,thread float *delta){
  c[3]=0.0f;p[0]=-c[0]/b[0];delta[0]=d[0]/b[0];
  for(int k=1;k<4;k++){float recip=1.0f/(b[k]+a[k]*p[k-1]);p[k]=-c[k]*recip;delta[k]=(d[k]-a[k]*delta[k-1])*recip;}
  p[3]=delta[3];for(int k=2;k>=0;k--)p[k]=p[k]*p[k+1]+delta[k];
}
static inline void winfil(device const float *p,float dt,thread const float *zsoil,thread const float *sh2o,thread const float *sice,float sicemax,float qinsur,thread float &pddum,thread float &runsrf){
 if(!(qinsur>0.0f))return;float dt1=dt/86400.0f,smcav=p[0]-p[4];float dmax[4],dice=-zsoil[0]*sice[0];
 dmax[0]=(-zsoil[0])*smcav;dmax[0]*=1.0f-((sh2o[0]+sice[0]-p[4])/smcav);float dd=dmax[0];
 for(int k=1;k<4;k++){dice+=(zsoil[k-1]-zsoil[k])*sice[k];dmax[k]=(zsoil[k-1]-zsoil[k])*smcav;dmax[k]*=1.0f-((sh2o[k]+sice[k]-p[4+k])/smcav);dd+=dmax[k];}
 float val=1.0f-wexp(-p[20]*dt1),ddt=dd*val,px=wmax(0.0f,qinsur*dt),infmax=(px*(ddt/(px+ddt)))/dt;
 float fcr=1.0f;if(dice>0.01f){float acrt=(3.0f*p[21])/dice,ssum=1.0f;ssum+=(acrt*acrt)/2.0f;ssum+=acrt;fcr=1.0f-wexp(-acrt)*ssum;}
 infmax*=fcr;float wcnd;wdf2(sh2o[0],sicemax,p[0],p[8],p[16],p[12],wcnd);infmax=wmax(infmax,wcnd);infmax=wmin(infmax,px/dt);
 runsrf=wmax(0.0f,qinsur-infmax);pddum=qinsur-runsrf;
}
static inline void wsrt(device const float *p,thread const float *zsoil,float pddum,thread const float *etrani,float qseva,thread const float *smc,thread const float *fcr,thread float *rhstt,thread float *ai,thread float *bi,thread float *ci,thread float &qdrain,thread float *wcnd){
 float wdf[4],smx[4],denom[4],ddz[4],dsmdz[4],wflux[4];for(int k=0;k<4;k++){wdf[k]=wdf1(smc[k],fcr[k],p[k],p[8+k],p[16+k],p[12+k],wcnd[k]);smx[k]=smc[k];ddz[k]=dsmdz[k]=0.0f;}qdrain=0.0f;
 for(int k=0;k<4;k++){if(k==0){denom[k]=-zsoil[k];float tmp=-zsoil[k+1];ddz[k]=2.0f/tmp;dsmdz[k]=2.0f*(smx[k]-smx[k+1])/tmp;wflux[k]=(((wdf[k]*dsmdz[k]+wcnd[k])-pddum)+etrani[k])+qseva;}
 else if(k<3){denom[k]=zsoil[k-1]-zsoil[k];float tmp=zsoil[k-1]-zsoil[k+1];ddz[k]=2.0f/tmp;dsmdz[k]=2.0f*(smx[k]-smx[k+1])/tmp;wflux[k]=(((wdf[k]*dsmdz[k]+wcnd[k])-wdf[k-1]*dsmdz[k-1]-wcnd[k-1])+etrani[k]);}
 else{denom[k]=zsoil[k-1]-zsoil[k];qdrain=p[22]*wcnd[k];wflux[k]=(-wdf[k-1]*dsmdz[k-1]-wcnd[k-1]+etrani[k])+qdrain;}}
 for(int k=0;k<4;k++){if(k==0){ai[k]=0.0f;bi[k]=wdf[k]*ddz[k]/denom[k];ci[k]=-bi[k];}else if(k<3){ai[k]=-(wdf[k-1]*ddz[k-1]/denom[k]);ci[k]=-(wdf[k]*ddz[k]/denom[k]);bi[k]=-(ai[k]+ci[k]);}else{ai[k]=-(wdf[k-1]*ddz[k-1]/denom[k]);ci[k]=0.0f;bi[k]=-(ai[k]+ci[k]);}rhstt[k]=wflux[k]/(-denom[k]);}
}
static inline float wsstep(device const float *p,float dt,thread const float *dz,thread const float *sice,thread float *sh2o,thread float *ai,thread float *bi,thread float *ci,thread float *rhstt,thread float *smc){
 float wp=0.0f;for(int k=0;k<4;k++){rhstt[k]*=dt;ai[k]*=dt;bi[k]=1.0f+bi[k]*dt;ci[k]*=dt;}float ri[4],ciin[4],del[4];for(int k=0;k<4;k++){ri[k]=rhstt[k];ciin[k]=ci[k];}wrosr12(ai,bi,ciin,ri,ci,del);for(int k=0;k<4;k++)sh2o[k]+=ci[k];
 for(int k=3;k>=1;k--){float ep=wmax(W_ONEEM4,p[k]-sice[k]);wp=wmax(sh2o[k]-ep,0.0f)*dz[k];sh2o[k]=wmin(ep,sh2o[k]);sh2o[k-1]+=wp/dz[k-1];}
 float ep=wmax(W_ONEEM4,p[0]-sice[0]);wp=wmax(sh2o[0]-ep,0.0f)*dz[0];sh2o[0]=wmin(ep,sh2o[0]);if(wp>0.0f){sh2o[1]+=wp/dz[1];for(int k=1;k<3;k++){float e=wmax(W_ONEEM4,p[k]-sice[k]);wp=wmax(sh2o[k]-e,0.0f)*dz[k];sh2o[k]=wmin(e,sh2o[k]);sh2o[k+1]+=wp/dz[k+1];}float e=wmax(W_ONEEM4,p[3]-sice[3]);wp=wmax(sh2o[3]-e,0.0f)*dz[3];sh2o[3]=wmin(e,sh2o[3]);}for(int k=0;k<4;k++)smc[k]=sh2o[k]+sice[k];return wp;
}
static inline void wsoilwater(device const float *p,int urban,float dt,thread const float *zsoil,thread const float *dz,float qinsur,float qseva,thread const float *etrani,thread const float *sice,thread float *sh2o,thread float *smc,thread float &runsub,thread float &runsrf,thread float &qdrain,thread float *wcnd){
 float rsat=0.0f;for(int k=0;k<4;k++){float ep=wmax(W_ONEEM4,p[k]-sice[k]);rsat+=wmax(0.0f,sh2o[k]-ep)*dz[k];sh2o[k]=wmin(ep,sh2o[k]);}
 float fcr[4],sicemax=0.0f;for(int k=0;k<4;k++){float fi=wmin(1.0f,sice[k]/p[k]);fcr[k]=wmax(0.0f,(wexp(-4.0f*(1.0f-fi))-wexp(-4.0f))/(1.0f-wexp(-4.0f)));sicemax=wmax(sicemax,sice[k]);}if(urban)fcr[0]=0.95f;
 float pddum=0.0f,run0=0.0f;winfil(p,dt,zsoil,sh2o,sice,sicemax,qinsur,pddum,run0);
 int niter=(pddum*dt>dz[0]*p[0])?6:3;float dtfine=dt/float(niter),qdrain_save=0.0f,runsrf_save=0.0f,ai[4],bi[4],ci[4],rhstt[4];runsub=0.0f;
 for(int it=0;it<niter;it++){if(qinsur>0.0f)winfil(p,dtfine,zsoil,sh2o,sice,sicemax,qinsur,pddum,run0);wsrt(p,zsoil,pddum,etrani,qseva,smc,fcr,rhstt,ai,bi,ci,qdrain,wcnd);float wp=wsstep(p,dtfine,dz,sice,sh2o,ai,bi,ci,rhstt,smc);rsat+=wp;qdrain_save+=qdrain;runsrf_save+=run0;}
 qdrain=(qdrain_save/float(niter))*1000.0f;runsrf=(runsrf_save/float(niter))*1000.0f+(rsat*1000.0f)/dt;
 float mliq[4];for(int k=0;k<4;k++)mliq[k]=sh2o[k]*dz[k]*1000.0f;for(int iz=0;iz<3;iz++){float xs=mliq[iz]<0.0f?W_WATMIN-mliq[iz]:0.0f;mliq[iz]+=xs;mliq[iz+1]-=xs;}float xs=mliq[3]<W_WATMIN?W_WATMIN-mliq[3]:0.0f;mliq[3]+=xs;runsub-=xs/dt;for(int iz=0;iz<4;iz++)sh2o[iz]=mliq[iz]/(dz[iz]*1000.0f);
}
static inline void wcanwater(device const float *par,float dt,float fcev,float fctr,float elai,float esai,float fveg,float bdfall,bool frozen,thread float &canliq,thread float &canice,thread float &tv,thread float *o){
 float lsai=elai+esai,maxliq=(fveg*par[23])*lsai,etran,qevac,qdewc,qsubc,qfroc;
 if(!frozen){etran=wmax(fctr/2.5104e6f,0.0f);qevac=wmax(fcev/2.5104e6f,0.0f);qdewc=fabs(wmin(fcev/2.5104e6f,0.0f));qsubc=qfroc=0.0f;}else{etran=wmax(fctr/2.844e6f,0.0f);qevac=qdewc=0.0f;qsubc=wmax(fcev/2.844e6f,0.0f);qfroc=fabs(wmin(fcev/2.844e6f,0.0f));}
 qevac=wmin(canliq/dt,qevac);canliq=wmax(0.0f,canliq+(qdewc-qevac)*dt);if(canliq<=1.0e-6f)canliq=0.0f;
 float maxsno=((fveg*6.6f)*(0.27f+46.0f/bdfall))*lsai;qsubc=wmin(canice/dt,qsubc);canice=wmax(0.0f,canice+(qfroc-qsubc)*dt);if(canice<=1.0e-6f)canice=0.0f;
 float fwet=canice>0.0f&&canice>=canliq?wmax(canice,maxsno>1e-6f?maxsno:1e-6f):wmax(canliq,maxliq>1e-6f?maxliq:1e-6f);
 if(canice>0.0f&&canice>=canliq)fwet=canice/wmax(maxsno,1.0e-6f);else fwet=canliq/wmax(maxliq,1.0e-6f);fwet=wpow(wmin(fwet,1.0f),0.667f);
 float cmc=canliq+canice,qmeltc=0.0f,qfrzc=0.0f;
 if(canice>1.0e-6f&&tv>W_TFRZ){qmeltc=wmin(canice/dt,(((tv-W_TFRZ)*W_CICE*canice)/W_DENICE)/(dt*W_HFUS));canice=wmax(0.0f,canice-qmeltc*dt);canliq=wmax(0.0f,cmc-canice);tv=fwet*W_TFRZ+(1.0f-fwet)*tv;}
 if(canliq>1.0e-6f&&tv<W_TFRZ){qfrzc=wmin(canliq/dt,(((W_TFRZ-tv)*W_CWAT*canliq)/W_DENH2O)/(dt*W_HFUS));canliq=wmax(0.0f,canliq-qfrzc*dt);canice=wmax(0.0f,cmc-canliq);tv=fwet*W_TFRZ+(1.0f-fwet)*tv;}
 cmc=canliq+canice;float ecan=(qevac+qsubc-qdewc)-qfroc;o[0]=cmc;o[1]=ecan;o[2]=etran;o[3]=fwet;o[4]=qsubc;o[5]=qfroc;o[6]=qfrzc;o[7]=qmeltc;o[8]=qevac;o[9]=qdewc;
}

constant float SN_ZERO=0.0f,SN_ONE=1.0f,SN_HALF=0.5f,SN_TWO=2.0f;
constant float SN_C2=0.021f,SN_C3=2.5e-6f,SN_C4=0.04f,SN_DM=100.0f;
constant float SN_ETA0=1.33e6f,SN_NEG46EM3=-0.046f,SN_NEG008=-0.08f;
constant float SN_P001=0.001f,SN_P01=0.01f,SN_NEGHALF=-0.5f;
constant float SN_P025=0.025f,SN_P05=0.05f,SN_P1=0.1f,SN_P20=0.2f;
constant float SN_1EM8=1.0e-8f,SN_1EM6=1.0e-6f,SN_500=500.0f,SN_50=50.0f;
constant float SN_MAXLIQ=0.4f,SN_5000=5000.0f;
constant float SN_DZMIN[3]={0.025f,0.025f,0.1f};

struct WaterSnowCol {
 int isnow; float snowh,sneqv; float snice[3],snliq[3];
 float stc[7],zsnso[7],dzsnso[7]; float sh2o[4],sice[4];
};
static inline uint wsi(int j){return uint(j+2);}
static inline uint wfi(int j){return uint(j+2);}
static inline float wsice(thread WaterSnowCol &c,int j){return c.snice[wsi(j)];}
static inline float wsliq(thread WaterSnowCol &c,int j){return c.snliq[wsi(j)];}
static inline float wsdz(thread WaterSnowCol &c,int j){return c.dzsnso[wfi(j)];}
static inline float wstc(thread WaterSnowCol &c,int j){return c.stc[wfi(j)];}
static inline void wsputice(thread WaterSnowCol &c,int j,float v){c.snice[wsi(j)]=v;}
static inline void wsputliq(thread WaterSnowCol &c,int j,float v){c.snliq[wsi(j)]=v;}
static inline void wsputdz(thread WaterSnowCol &c,int j,float v){c.dzsnso[wfi(j)]=v;}
static inline void wsputtc(thread WaterSnowCol &c,int j,float v){c.stc[wfi(j)]=v;}
static inline float wsmax(float a,float b){return a<b?b:a;}
static inline float wsmin(float a,float b){return b<a?b:a;}

static inline void wscombo(thread float &dz,thread float &wliq,thread float &wice,thread float &t,float dz2,float wliq2,float wice2,float t2){
 float dzc=dz+dz2,wicec=wice+wice2,wliqc=wliq+wliq2;
 float h=((W_CICE*wice+W_CWAT*wliq)*(t-W_TFRZ))+W_HFUS*wliq;
 float h2=((W_CICE*wice2+W_CWAT*wliq2)*(t2-W_TFRZ))+W_HFUS*wliq2;
 float hc=h+h2,tc;
 if(hc<0.0f)tc=W_TFRZ+hc/(W_CICE*wicec+W_CWAT*wliqc);
 else if(hc<=W_HFUS*wliqc)tc=W_TFRZ;
 else tc=W_TFRZ+(hc-W_HFUS*wliqc)/(W_CICE*wicec+W_CWAT*wliqc);
 dz=dzc;wice=wicec;wliq=wliqc;t=tc;
}
static inline void wssnowfall(thread WaterSnowCol &c,float dt,float qsnow,float snowhin,float sfctmp){
 int newnode=0;
 if(c.isnow==0&&qsnow>0.0f){c.snowh+=snowhin*dt;c.sneqv+=qsnow*dt;}
 if(c.isnow==0&&c.snowh>=SN_P025){c.isnow=-1;newnode=1;wsputdz(c,0,c.snowh);c.snowh=0.0f;wsputtc(c,0,wsmin(W_TFRZ,sfctmp));wsputice(c,0,c.sneqv);wsputliq(c,0,0.0f);}
 if(c.isnow<0&&newnode==0&&qsnow>0.0f){wsputice(c,c.isnow+1,wsice(c,c.isnow+1)+qsnow*dt);wsputdz(c,c.isnow+1,wsdz(c,c.isnow+1)+snowhin*dt);}
}
static inline void wscompact(thread WaterSnowCol &c,float dt,thread const int *imelt,thread const float *ficeold){
 float fice[3],burden=0.0f;
 for(int j=c.isnow+1;j<=0;j++){
  int k=j+2;float wx=wsice(c,j)+wsliq(c,j);fice[k]=wsice(c,j)/wx;
  float voidf=1.0f-((wsice(c,j)/W_DENICE+wsliq(c,j)/W_DENH2O)/wsdz(c,j));
  if(voidf>SN_P001&&wsice(c,j)>SN_P1){
   float bi=wsice(c,j)/wsdz(c,j),td=wsmax(0.0f,W_TFRZ-wstc(c,j));
   float ddz1=-SN_C3*wexp(-SN_C4*td);
   if(bi>SN_DM)ddz1*=wexp(SN_NEG46EM3*(bi-SN_DM));
   if(wsliq(c,j)>SN_P01*wsdz(c,j))ddz1*=SN_TWO;
   float ddz2=(-(burden+SN_HALF*wx)*wexp(SN_NEG008*td-SN_C2*bi))/SN_ETA0;
   float ddz3=0.0f;
   if(imelt[k]==1){ddz3=wsmax(0.0f,(ficeold[k]-fice[k])/wsmax(SN_1EM6,ficeold[k]));ddz3=-ddz3/dt;}
   float pdzdtc=((ddz1+ddz2)+ddz3)*dt;pdzdtc=wsmax(SN_NEGHALF,pdzdtc);
   wsputdz(c,j,wsdz(c,j)*(1.0f+pdzdtc));
   wsputdz(c,j,wsmax(wsdz(c,j),wsice(c,j)/W_DENICE+wsliq(c,j)/W_DENH2O));
   wsputdz(c,j,wsmin(wsmax(wsdz(c,j),(wsice(c,j)+wsliq(c,j))/SN_500),(wsice(c,j)+wsliq(c,j))/SN_50));
  }
  burden+=wx;
 }
}
static inline void wssnowcombine(thread WaterSnowCol &c,thread float &ponding1,thread float &ponding2){
 int isnow_old=c.isnow;
 for(int j=isnow_old+1;j<=0;j++)if(wsice(c,j)<=SN_P1){
  if(j!=0){wsputliq(c,j+1,wsliq(c,j+1)+wsliq(c,j));wsputice(c,j+1,wsice(c,j+1)+wsice(c,j));wsputdz(c,j+1,wsdz(c,j+1)+wsdz(c,j));}
  else if(c.isnow<-1){wsputliq(c,j-1,wsliq(c,j-1)+wsliq(c,j));wsputice(c,j-1,wsice(c,j-1)+wsice(c,j));wsputdz(c,j-1,wsdz(c,j-1)+wsdz(c,j));}
  else if(wsice(c,j)>=0.0f){ponding1=wsliq(c,j);c.sneqv=wsice(c,j);c.snowh=wsdz(c,j);}
  else{ponding1=wsliq(c,j)+wsice(c,j);if(ponding1<0.0f){c.sice[0]+=ponding1/(wsdz(c,1)*W_DENH2O);ponding1=0.0f;}c.sneqv=0.0f;c.snowh=0.0f;}
  if(j==0&&c.isnow>=-1){wsputliq(c,j,0.0f);wsputice(c,j,0.0f);wsputdz(c,j,0.0f);}
  if(j>c.isnow+1&&c.isnow<-1)for(int i=j;i>=c.isnow+2;i--){wsputtc(c,i,wstc(c,i-1));wsputliq(c,i,wsliq(c,i-1));wsputice(c,i,wsice(c,i-1));wsputdz(c,i,wsdz(c,i-1));}
  c.isnow++;
 }
 if(c.sice[0]<0.0f){c.sh2o[0]+=c.sice[0];c.sice[0]=0.0f;}
 if(c.isnow==0)return;
 c.sneqv=0.0f;c.snowh=0.0f;float zwice=0.0f,zwliq=0.0f;
 for(int j=c.isnow+1;j<=0;j++){c.sneqv=(c.sneqv+wsice(c,j))+wsliq(c,j);c.snowh+=wsdz(c,j);zwice+=wsice(c,j);zwliq+=wsliq(c,j);}
 if(c.snowh<SN_P025&&c.isnow<0){c.isnow=0;c.sneqv=zwice;ponding2=zwliq;if(c.sneqv<=0.0f)c.snowh=0.0f;}
 if(c.isnow<-1){isnow_old=c.isnow;int mssi=1;for(int i=isnow_old+1;i<=0;i++)if(wsdz(c,i)<SN_DZMIN[mssi-1]){
  int neighbor;if(i==c.isnow+1)neighbor=i+1;else if(i==0)neighbor=i-1;else{neighbor=i+1;if(wsdz(c,i-1)+wsdz(c,i)<wsdz(c,i+1)+wsdz(c,i))neighbor=i-1;}
  int j,l;if(neighbor>i){j=neighbor;l=i;}else{j=i;l=neighbor;}
  float dz=wsdz(c,j),liq=wsliq(c,j),ice=wsice(c,j),temp=wstc(c,j);wscombo(dz,liq,ice,temp,wsdz(c,l),wsliq(c,l),wsice(c,l),wstc(c,l));
  wsputdz(c,j,dz);wsputliq(c,j,liq);wsputice(c,j,ice);wsputtc(c,j,temp);
  if(j-1>c.isnow+1)for(int k=j-1;k>=c.isnow+2;k--){wsputtc(c,k,wstc(c,k-1));wsputice(c,k,wsice(c,k-1));wsputliq(c,k,wsliq(c,k-1));wsputdz(c,k,wsdz(c,k-1));}
  c.isnow++;if(c.isnow>=-1)break;
 }else mssi++;}
}
static inline void wssnowdivide(thread WaterSnowCol &c){
 float dz[3],ice[3],liq[3],temp[3];for(uint k=0;k<3;k++){dz[k]=as_type<float>(0x7fc00000u);ice[k]=dz[k];liq[k]=dz[k];temp[k]=dz[k];}
 int isn=c.isnow;for(int j=1;j<=3;j++)if(j<=abs(isn)){dz[j-1]=wsdz(c,j+isn);ice[j-1]=wsice(c,j+isn);liq[j-1]=wsliq(c,j+isn);temp[j-1]=wstc(c,j+isn);}
 int msno=abs(isn);float drr,propor,zwice,zwliq,dtdz;
 if(msno==1&&dz[0]>SN_P05){msno=2;dz[0]/=SN_TWO;ice[0]/=SN_TWO;liq[0]/=SN_TWO;dz[1]=dz[0];ice[1]=ice[0];liq[1]=liq[0];temp[1]=temp[0];}
 if(msno>1&&dz[0]>SN_P05){drr=dz[0]-SN_P05;propor=drr/dz[0];zwice=propor*ice[0];zwliq=propor*liq[0];propor=SN_P05/dz[0];ice[0]*=propor;liq[0]*=propor;dz[0]=SN_P05;wscombo(dz[1],liq[1],ice[1],temp[1],drr,zwliq,zwice,temp[0]);
  if(msno<=2&&dz[1]>SN_P20){msno=3;dtdz=(temp[0]-temp[1])/((dz[0]+dz[1])/SN_TWO);dz[1]/=SN_TWO;ice[1]/=SN_TWO;liq[1]/=SN_TWO;dz[2]=dz[1];ice[2]=ice[1];liq[2]=liq[1];temp[2]=temp[1]-dtdz*dz[1]/SN_TWO;if(temp[2]>=W_TFRZ)temp[2]=temp[1];else temp[1]+=dtdz*dz[1]/SN_TWO;}}
 if(msno>2&&dz[1]>SN_P20){drr=dz[1]-SN_P20;propor=drr/dz[1];zwice=propor*ice[1];zwliq=propor*liq[1];propor=SN_P20/dz[1];ice[1]*=propor;liq[1]*=propor;dz[1]=SN_P20;wscombo(dz[2],liq[2],ice[2],temp[2],drr,zwliq,zwice,temp[1]);}
 c.isnow=-msno;for(int j=c.isnow+1;j<=0;j++){uint k=uint(j-c.isnow-1);wsputdz(c,j,dz[k]);wsputice(c,j,ice[k]);wsputliq(c,j,liq[k]);wsputtc(c,j,temp[k]);}
}
static inline float wssnowh2o(thread WaterSnowCol &c,float dt,float qsnfro,float qsnsub,float qrain,float ssi,float snow_ret_fac,thread float &ponding1,thread float &ponding2){
 float vol_liq[3]={0.0f,0.0f,0.0f},vol_ice[3]={0.0f,0.0f,0.0f},epore[3]={0.0f,0.0f,0.0f};
 if(c.sneqv==0.0f){c.sice[0]+=(qsnfro-qsnsub)*dt/(wsdz(c,1)*W_DENH2O);if(c.sice[0]<0.0f){c.sh2o[0]+=c.sice[0];c.sice[0]=0.0f;}}
 if(c.isnow==0&&c.sneqv>0.0f){float tmp=c.sneqv;c.sneqv=(c.sneqv-qsnsub*dt)+qsnfro*dt;float propor=c.sneqv/tmp;c.snowh=wsmax(0.0f,propor*c.snowh);c.snowh=wsmin(wsmax(c.snowh,c.sneqv/SN_500),c.sneqv/SN_50);
  if(c.sneqv<0.0f){c.sice[0]+=c.sneqv/(wsdz(c,1)*W_DENH2O);c.sneqv=0.0f;c.snowh=0.0f;}if(c.sice[0]<0.0f){c.sh2o[0]+=c.sice[0];c.sice[0]=0.0f;}}
 if(c.snowh<=SN_1EM8||c.sneqv<=SN_1EM6){c.snowh=0.0f;c.sneqv=0.0f;}
 if(c.isnow<0){float wgdif=(wsice(c,c.isnow+1)-qsnsub*dt)+qsnfro*dt;wsputice(c,c.isnow+1,wgdif);if(wgdif<SN_1EM6&&c.isnow<0)wssnowcombine(c,ponding1,ponding2);if(c.isnow<0)wsputliq(c,c.isnow+1,wsmax(0.0f,wsliq(c,c.isnow+1)+qrain*dt));}
 for(int j=c.isnow+1;j<=0;j++){uint k=wsi(j);vol_ice[k]=wsmin(1.0f,wsice(c,j)/(wsdz(c,j)*W_DENICE));epore[k]=1.0f-vol_ice[k];}
 float qin=0.0f,qout=0.0f;
 for(int j=c.isnow+1;j<=0;j++){uint k=wsi(j);wsputliq(c,j,wsliq(c,j)+qin);vol_liq[k]=wsliq(c,j)/(wsdz(c,j)*W_DENH2O);qout=wsmax(0.0f,(vol_liq[k]-ssi*epore[k])*wsdz(c,j));if(j==0)qout=wsmax((vol_liq[k]-epore[k])*wsdz(c,j),snow_ret_fac*dt*qout);qout*=W_DENH2O;wsputliq(c,j,wsliq(c,j)-qout);
  if(wsliq(c,j)/(wsice(c,j)+wsliq(c,j))>SN_MAXLIQ){qout+=wsliq(c,j)-(SN_MAXLIQ/(1.0f-SN_MAXLIQ))*wsice(c,j);wsputliq(c,j,(SN_MAXLIQ/(1.0f-SN_MAXLIQ))*wsice(c,j));}qin=qout;}
 for(int j=c.isnow+1;j<=0;j++)wsputdz(c,j,wsmax(wsdz(c,j),wsliq(c,j)/W_DENH2O+wsice(c,j)/W_DENICE));
 return qout/dt;
}
static inline void wssnowwater(thread WaterSnowCol &c,float dt,thread const float *zsoil,thread const int *imelt,thread const float *ficeold,float sfctmp,float snowhin,float qsnow,float qsnfro,float qsnsub,float qrain,float ssi,float snow_ret_fac,thread float &qsnbot,thread float &snoflow,thread float &ponding1,thread float &ponding2){
 snoflow=0.0f;ponding1=0.0f;ponding2=0.0f;wssnowfall(c,dt,qsnow,snowhin,sfctmp);
 if(c.isnow<0)wscompact(c,dt,imelt,ficeold);if(c.isnow<0)wssnowcombine(c,ponding1,ponding2);if(c.isnow<0)wssnowdivide(c);
 qsnbot=wssnowh2o(c,dt,qsnfro,qsnsub,qrain,ssi,snow_ret_fac,ponding1,ponding2);
 for(int iz=-2;iz<=c.isnow;iz++){wsputice(c,iz,0.0f);wsputliq(c,iz,0.0f);wsputtc(c,iz,0.0f);wsputdz(c,iz,0.0f);c.zsnso[wfi(iz)]=0.0f;}
 if(c.sneqv>SN_5000){float bdsnow=wsice(c,0)/wsdz(c,0);snoflow=c.sneqv-SN_5000;wsputice(c,0,wsice(c,0)-snoflow);wsputdz(c,0,wsdz(c,0)-snoflow/bdsnow);snoflow/=dt;}
 if(c.isnow<0){c.sneqv=0.0f;for(int iz=c.isnow+1;iz<=0;iz++)c.sneqv=(c.sneqv+wsice(c,iz))+wsliq(c,iz);}
 for(int iz=c.isnow+1;iz<=0;iz++)wsputdz(c,iz,-wsdz(c,iz));
 c.dzsnso[3]=zsoil[0];for(int iz=2;iz<=4;iz++)c.dzsnso[uint(iz+2)]=zsoil[uint(iz-1)]-zsoil[uint(iz-2)];
 c.zsnso[wfi(c.isnow+1)]=wsdz(c,c.isnow+1);for(int iz=c.isnow+2;iz<=4;iz++)c.zsnso[wfi(iz)]=c.zsnso[wfi(iz-1)]+wsdz(c,iz);
 for(int iz=c.isnow+1;iz<=4;iz++)wsputdz(c,iz,-wsdz(c,iz));
 if(c.isnow<0){c.snowh=0.0f;for(int iz=c.isnow+1;iz<=0;iz++)c.snowh+=wsdz(c,iz);}
}

kernel void noahmp_water_batch(
    constant NoahmpWaterBatchResources &r [[buffer(0)]],
    constant NoahmpWaterBatchParams &q [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
  if(idx>=uint(q.n))return;ulong pb=ulong(idx)*26ul,xb=ulong(idx)*76ul,ib=ulong(idx)*9ul,ob=ulong(idx)*73ul;
  int urban=r.iin[ib+1],nroot=r.iin[ib+2],ist=r.iin[ib+3];
  WaterSnowCol c;c.isnow=r.iin[ib];c.snowh=r.fin[xb];c.sneqv=r.fin[xb+1];
  for(uint k=0;k<3;k++){c.snice[k]=r.fin[xb+2+k];c.snliq[k]=r.fin[xb+5+k];}
  for(uint k=0;k<7;k++){c.stc[k]=r.fin[xb+8+k];c.zsnso[k]=r.fin[xb+15+k];c.dzsnso[k]=r.fin[xb+22+k];}
  for(uint k=0;k<4;k++){c.sh2o[k]=r.fin[xb+29+k];c.sice[k]=r.fin[xb+33+k];}
  float sh2o[4],sice[4],smc[4],zsoil[4],btrani[4],acc_e[4],dz[4],ficeold[3];int imelt[3];
  for(uint k=0;k<4;k++){sh2o[k]=c.sh2o[k];sice[k]=c.sice[k];smc[k]=r.fin[xb+69+k];zsoil[k]=r.fin[xb+57+k];btrani[k]=r.fin[xb+61+k];acc_e[k]=r.fin[xb+65+k];}
  for(uint k=0;k<3;k++){ficeold[k]=r.fin[xb+73+k];imelt[k]=r.iin[ib+6+k];}
  float dt=r.fin[xb+37],fcev=r.fin[xb+38],fctr=r.fin[xb+39],elai=r.fin[xb+40],esai=r.fin[xb+41],fveg=r.fin[xb+42],bdfall=r.fin[xb+43];
  float sfctmp=r.fin[xb+44],qvap=r.fin[xb+45],qdew=r.fin[xb+46],qsnow=r.fin[xb+47],qrain=r.fin[xb+48],snowhin=r.fin[xb+49],ponding=r.fin[xb+50],canliq=r.fin[xb+51],canice=r.fin[xb+52],tv=r.fin[xb+53],wslake=r.fin[xb+54];
  float acc_qinsur=r.fin[xb+55],acc_qseva=r.fin[xb+56];
  float cw[10];wcanwater(r.par+pb,dt,fcev,fctr,elai,esai,fveg,bdfall,r.iin[ib+4]!=0,canliq,canice,tv,cw);
  float cmc=cw[0],ecan=cw[1],etran=cw[2],fwet=cw[3],qsubc=cw[4],qfroc=cw[5],qfrzc=cw[6],qmeltc=cw[7],qevac=cw[8],qdewc=cw[9];
  float qsnsub=0.0f;if(c.sneqv>0.0f)qsnsub=wmin(qvap,c.sneqv/dt);float qseva=qvap-qsnsub;
  float qsnfro=0.0f;if(c.sneqv>0.0f)qsnfro=qdew;float qsdew=qdew-qsnfro;
  float qsnbot=0.0f,snoflow=0.0f,ponding1=0.0f,ponding2=0.0f;
  wssnowwater(c,dt,zsoil,imelt,ficeold,sfctmp,snowhin,qsnow,qsnfro,qsnsub,qrain,r.par[pb+24],r.par[pb+25],qsnbot,snoflow,ponding1,ponding2);
  for(uint k=0;k<4;k++){sh2o[k]=c.sh2o[k];sice[k]=c.sice[k];}
  float qinsur=((ponding+ponding1+ponding2)/dt)*0.001f;
  if(c.isnow==0)qinsur+=(qsnbot+qsdew+qrain)*0.001f;else qinsur+=(qsnbot+qsdew)*0.001f;
  if(r.iin[ib+5]!=0){float dz0=c.dzsnso[3];sice[0]+=((qsdew-qseva)*dt)/(dz0*1000.0f);qsdew=0.0f;qseva=0.0f;if(sice[0]<0.0f){sh2o[0]+=sice[0];sice[0]=0.0f;}smc[0]=sh2o[0]+sice[0];}
  qseva*=0.001f;
  for(int iz=0;iz<nroot;iz++)acc_e[iz]+=etran*btrani[iz]*0.001f;
  acc_qinsur+=qinsur;acc_qseva+=qseva;float runsub=0.0f,runsrf=0.0f,qdrain=0.0f,wcnd[4];
  for(uint k=0;k<4;k++)dz[k]=c.dzsnso[3+k];
  if(ist==2){runsrf=0.0f;if(wslake>=W_WSLMAX)runsrf=acc_qinsur*1000.0f*dt;wslake=wslake+(acc_qinsur-acc_qseva)*1000.0f*dt-runsrf;}
  else {wsoilwater(r.par+pb,urban,dt,zsoil,dz,acc_qinsur,acc_qseva,acc_e,sice,sh2o,smc,runsub,runsrf,qdrain,wcnd);runsub+=qdrain;for(uint k=0;k<4;k++)smc[k]=sh2o[k]+sice[k];runsrf*=dt;runsub*=dt;}
  runsub+=snoflow*dt;
  for(uint k=0;k<4;k++){c.sh2o[k]=sh2o[k];c.sice[k]=sice[k];r.fout[ob+37+k]=smc[k];r.fout[ob+47+k]=acc_e[k];}
  for(uint k=0;k<37;k++)r.fout[ob+k]=r.fin[xb+k];
  r.fout[ob]=c.snowh;r.fout[ob+1]=c.sneqv;
  for(uint k=0;k<3;k++){r.fout[ob+2+k]=c.snice[k];r.fout[ob+5+k]=c.snliq[k];}
  for(uint k=0;k<7;k++){r.fout[ob+8+k]=c.stc[k];r.fout[ob+15+k]=c.zsnso[k];r.fout[ob+22+k]=c.dzsnso[k];}
  for(uint k=0;k<4;k++){r.fout[ob+29+k]=sh2o[k];r.fout[ob+33+k]=sice[k];r.fout[ob+18+k]=zsoil[k];r.fout[ob+25+k]=dz[k];}
  r.fout[ob+41]=canliq;r.fout[ob+42]=canice;r.fout[ob+43]=tv;r.fout[ob+44]=wslake;r.fout[ob+45]=acc_qinsur;r.fout[ob+46]=acc_qseva;
  r.fout[ob+51]=cmc;r.fout[ob+52]=ecan;r.fout[ob+53]=etran;r.fout[ob+54]=fwet;r.fout[ob+55]=runsrf;r.fout[ob+56]=runsub;
  r.fout[ob+57]=0.0f;r.fout[ob+58]=ponding1;r.fout[ob+59]=ponding2;r.fout[ob+60]=qsnbot;r.fout[ob+61]=qsnsub;r.fout[ob+62]=qsnfro;
  r.fout[ob+63]=qsubc;r.fout[ob+64]=qfroc;r.fout[ob+65]=qfrzc;r.fout[ob+66]=qmeltc;r.fout[ob+67]=qevac;r.fout[ob+68]=qdewc;
  r.fout[ob+69]=qseva;r.fout[ob+70]=qsdew;r.fout[ob+71]=qinsur;r.fout[ob+72]=snoflow;
  r.iout[idx]=c.isnow;
}

#if 0 // Disabled with the non-production packet-local math draft above.
kernel void noahmp_water_expf(
    constant NoahmpWaterExpfResources &r [[buffer(0)]],
    constant NoahmpWaterExpfParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    r.y[tid] = w_glibc_expf(r.x[tid]);
}

kernel void noahmp_water_powf(
    constant NoahmpWaterPowfResources &r [[buffer(0)]],
    constant NoahmpWaterPowfParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    r.y[tid] = w_glibc_powf(r.x[tid], r.e[tid]);
}
#endif
