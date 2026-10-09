#include <metal_stdlib>
#include "cold_start_alt_bindings.metalh"

using namespace metal;

// Exact RN-even conversion of finite binary64 values to binary32 using only
// the two stored uint32 words. The MSL shader performs no FP64 operations;
// input resource is a float32 view over the original 8-byte values.
static inline ulong cs_round_shift(ulong value, uint shift) {
    const ulong quotient = value >> shift;
    const ulong remainder = value & ((1ul << shift) - 1ul);
    const ulong halfway = 1ul << (shift - 1u);
    return quotient + ulong(remainder > halfway
        || (remainder == halfway && (quotient & 1ul)));
}

static inline uint cs_binary64_to_binary32(uint lo, uint hi) {
    const uint sign = (hi >> 31u) << 31u;
    const uint exponent = (hi >> 20u) & 0x7ffu;
    const ulong fraction = (ulong(hi & 0x000fffffu) << 32u) | ulong(lo);

    if (exponent == 0x7ffu) {
        if (fraction == 0ul) return sign | 0x7f800000u;
        // Preserve the high payload bits and quiet signaling NaNs. The actual
        // caller validates NaNs immediately after this conversion.
        uint payload = uint(fraction >> 29u) | 0x00400000u;
        return sign | 0x7f800000u | (payload & 0x007fffffu);
    }
    if (exponent == 0u) return sign; // binary64 subnormals are below float32.

    int e = int(exponent) - 1023;
    const ulong significand = (1ul << 52u) | fraction;
    if (e > 127) return sign | 0x7f800000u;

    if (e >= -126) {
        ulong rounded = cs_round_shift(significand, 29u);
        if (rounded == (1ul << 24u)) {
            rounded >>= 1u;
            ++e;
            if (e > 127) return sign | 0x7f800000u;
        }
        return sign | (uint(e + 127) << 23u)
                    | (uint(rounded) & 0x007fffffu);
    }

    if (e < -150) return sign;
    const uint shift = uint(-e - 97);
    const ulong rounded = cs_round_shift(significand, shift);
    if (rounded >= (1ul << 23u)) return sign | 0x00800000u;
    return sign | uint(rounded);
}

kernel void cold_start_alt(
    constant ColdStartAltResources &r [[buffer(0)]],
    constant ColdStartAltParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(q.n)) return;
    const device uint *words = reinterpret_cast<device const uint *>(r.input_words);
    const uint bits = cs_binary64_to_binary32(words[2u * tid],
                                               words[2u * tid + 1u]);
    r.output[tid] = as_type<float>(bits);
}
