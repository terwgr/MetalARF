#include <metal_stdlib>
using namespace metal;
#include "soft_gfk_log.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_pow.metalh"

kernel void gfk_probe(device const uint2 *input [[buffer(0)]],
                      device ulong *output [[buffer(1)]],
                      constant uint &count [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint2 row = input[gid];
    output[gid * 5] = ulong(soft_gfk_log(row.x));
    output[gid * 5 + 1] = ulong(soft_gfk_exp(row.x));
    output[gid * 5 + 2] = ulong(soft_gfk_pow_bits(row.x, row.y));
    const uint normal = 0x3f000000u | (row.x & 0x00ffffffu);
    output[gid * 5 + 3] = soft_gfk_powf_log2_bits(normal);
    const ulong scaled = soft64_mul(0x40471547652b82feul,
                                    soft64_from_f32_bits(normal));
    output[gid * 5 + 4] = soft_gfk_exp2_core(scaled,
        0x4338000000000000ul, 0x3ebc6af84b912394ul,
        0x3f2ebfce50fac4f3ul, 0x3f962e42ff0c52d6ul, 0ul);
}
