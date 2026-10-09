#include <metal_stdlib>
#include "../generated/rrtmgp_mcica_prepare_seed_bindings.metalh"
#include "../generated/rrtmgp_mcica_maxran_bindings.metalh"
#include "soft_binary64.metalh"

using namespace metal;
#pragma clang fp contract(off)

constant uint MCICA_P3 = 1179647999u;
constant uint MCICA_P4 = 2025259007u;
constant ulong MCICA_ZERO = 0x0000000000000000ul;
constant ulong MCICA_ONE = 0x3ff0000000000000ul;
constant ulong MCICA_CLDMIN = 0x3bc79ca10c924223ul; // binary64 1.0e-20
constant ulong MCICA_DRAW_SCALE = 0x3defffff9b574dbcul; // binary64 2.328306e-10
constant ulong MCICA_HALF = 0x3fe0000000000000ul;
constant ulong MCICA_SEED_SCALE = 0x41cdcd6500000000ul; // binary64 1.0e9

struct McicaState { uint s1, s2, s3, s4; };

static inline void mcica_advance(thread McicaState &s) {
    s.s1 = 69069u * s.s1 + 1327217885u;
    s.s2 ^= s.s2 << 13u;
    s.s2 ^= s.s2 >> 17u;
    s.s2 ^= s.s2 << 5u;
    s.s3 = 18000u * (s.s3 & 65535u) + (s.s3 >> 16u);
    s.s4 = 30903u * (s.s4 & 65535u) + (s.s4 >> 16u);
}

static inline int mcica_draw(thread McicaState &s) {
    mcica_advance(s);
    return as_type<int>(s.s1 + s.s2 + (s.s3 << 16u) + s.s4);
}

static inline uint mcica_seed_word(float value) {
    const ulong pmid = soft64_from_f32_bits(as_type<uint>(value));
    const Soft64I32Result integer = soft64_to_i32_rz(pmid);
    if (!integer.valid) return 0u;
    const ulong fraction = soft64_sub(pmid, soft64_from_i32(integer.value));
    const Soft64I32Result scaled = soft64_to_i32_rz(
        soft64_mul(fraction, MCICA_SEED_SCALE));
    return scaled.valid ? uint(scaled.value) : 0u;
}

kernel void rrtmgp_mcica_prepare_seed(
    constant RrtmgpMcicaPrepareSeedResources &r [[buffer(0)]],
    constant RrtmgpMcicaPrepareSeedParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol) || p.nlay < 4) return;
    McicaState s;
    s.s1 = mcica_seed_word(r.play[col * uint(p.nlay)]);
    s.s2 = mcica_seed_word(r.play[col * uint(p.nlay) + 1u]);
    s.s3 = mcica_seed_word(r.play[col * uint(p.nlay) + 2u]);
    s.s4 = mcica_seed_word(r.play[col * uint(p.nlay) + 3u]);
    // WRF calls kissvec permuteseed times and discards each draw. Only the
    // integer state transition matters here; no binary64 draw is consumed.
    for (int i = 0; i < p.permuteseed; ++i) mcica_advance(s);
    const uint offset = col * 4u;
    r.seed[offset] = s.s1;
    r.seed[offset + 1u] = s.s2;
    r.seed[offset + 2u] = s.s3;
    r.seed[offset + 3u] = s.s4;
}

static inline uint mcica_gf2_apply(device const uint *matrix, uint value,
                                   uint stride) {
    uint out = 0u;
    while (value != 0u) {
        const uint bit = ctz(value);
        out ^= matrix[bit * stride];
        value &= value - 1u;
    }
    return out;
}

static inline uint mcica_mod_p3(ulong value) {
    return uint(value % ulong(MCICA_P3));
}

static inline uint mcica_mod_p4(ulong value) {
    return uint(value % ulong(MCICA_P4));
}

static inline McicaState mcica_jump_state(
    McicaState seed, device const uint *jump_s1,
    device const uint *jump_s2, device const uint *jump_s3,
    device const uint *jump_s4, uint g, uint nlay, uint ngpt) {
    if (g == 0u) return seed;
    const uint r3 = mcica_mod_p3(ulong(seed.s3 % MCICA_P3)
                                 * ulong(jump_s3[g]));
    const uint r4 = mcica_mod_p4(ulong(seed.s4 % MCICA_P4)
                                 * ulong(jump_s4[g]));
    const uint a1 = jump_s1[g * 2u] * seed.s1 + jump_s1[g * 2u + 1u];
    const uint a2 = mcica_gf2_apply(jump_s2 + g, seed.s2, ngpt);
    if (r3 > 1u && r4 > 1u) return McicaState{a1, a2, r3, r4};

    // Residues 0/1 alias the MWC state. Keep the exact CUDA fallback walk for
    // both MWC components; the independent LCG/xorshift jumps remain exact.
    McicaState walked = seed;
    const uint steps = g * nlay;
    for (uint i = 0u; i < steps; ++i) {
        walked.s3 = 18000u * (walked.s3 & 65535u) + (walked.s3 >> 16u);
        walked.s4 = 30903u * (walked.s4 & 65535u) + (walked.s4 >> 16u);
    }
    walked.s1 = a1;
    walked.s2 = a2;
    return walked;
}

static inline ulong mcica_complement(float fraction) {
    const ulong cf = soft64_from_f32_bits(as_type<uint>(fraction));
    if (soft64_less(cf, MCICA_CLDMIN)) return MCICA_ONE;
    return soft64_sub(MCICA_ONE, cf);
}

kernel void rrtmgp_mcica_maxran(
    constant RrtmgpMcicaMaxranResources &r [[buffer(0)]],
    constant RrtmgpMcicaMaxranParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total = uint(p.ncol) * uint(p.ngpt);
    if (tid >= total || p.nlay < 4 || p.ngpt <= 0) return;
    const uint col = tid / uint(p.ngpt);
    const uint g = tid - col * uint(p.ngpt);
    const uint layer_base = col * uint(p.nlay);
    const uint mask_base = col * uint(p.nlay) * uint(p.ngpt) + g;
    device uchar *mask = reinterpret_cast<device uchar *>(r.mask_words);

    bool any_cloud = false;
    for (uint k = 0u; k < uint(p.nlay); ++k)
        any_cloud |= !soft64_equal(
            mcica_complement(r.cldfra[layer_base + k]), MCICA_ONE);
    if (!any_cloud) {
        for (uint k = 0u; k < uint(p.nlay); ++k)
            mask[mask_base + k * uint(p.ngpt)] = uchar(0u);
        return;
    }

    const uint seed_offset = col * 4u;
    McicaState initial{r.seed[seed_offset], r.seed[seed_offset + 1u],
                       r.seed[seed_offset + 2u], r.seed[seed_offset + 3u]};
    McicaState state = mcica_jump_state(
        initial, r.jump_s1, r.jump_s2, r.jump_s3, r.jump_s4,
        g, uint(p.nlay), uint(p.ngpt));

    ulong cdf = MCICA_ZERO;
    ulong omb = MCICA_ZERO;
    for (uint k = 0u; k < uint(p.nlay); ++k) {
        const int kiss = mcica_draw(state);
        const ulong omc = mcica_complement(r.cldfra[layer_base + k]);
        const uint out_index = mask_base + k * uint(p.ngpt);
        if (soft64_equal(omc, MCICA_ONE)) {
            mask[out_index] = uchar(0u);
            cdf = MCICA_ZERO;
            omb = MCICA_ONE;
            continue;
        }
        const ulong draw = soft64_fma(
            soft64_from_i32(kiss), MCICA_DRAW_SCALE, MCICA_HALF);
        if (k == 0u) {
            cdf = draw;
        } else if (!soft64_greater(cdf, omb)) {
            cdf = soft64_mul(draw, omb);
        }
        mask[out_index] = uchar(soft64_greater_equal(cdf, omc) ? 1u : 0u);
        omb = omc;
    }
}
