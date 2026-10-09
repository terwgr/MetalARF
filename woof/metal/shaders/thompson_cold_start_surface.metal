#include <metal_stdlib>
#include "cold_start_surface_bindings.metalh"
#include "soft_binary64.metalh"

using namespace metal;

// CUDA cold_start_surface: cs_div(1, alt), then cs_mul(aerosol, recip),
// where each cs operation widens the exact FP32 input to binary64, rounds
// the binary64 operation, and narrows with gradual-underflow RN-even.
// The output carries the same 0/1/2 codes as the CUDA uint8 resource using
// int32 storage, available in the current typed Metal ABI.
kernel void cold_start_surface(
    constant ColdStartSurfaceResources &r [[buffer(0)]],
    constant ColdStartSurfaceParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(q.n)) return;
    const uint alt_bits = as_type<uint>(r.alt[tid]);
    const uint aerosol_bits = as_type<uint>(r.aerosol[tid]);
    const ulong reciprocal = soft64_div(soft64_from_f32_bits(0x3f800000u),
                                         soft64_from_f32_bits(alt_bits));
    const uint reciprocal32 = soft64_to_f32_bits(reciprocal);
    const ulong nw64 = soft64_mul(soft64_from_f32_bits(aerosol_bits),
                                  soft64_from_f32_bits(reciprocal32));
    const uint nw = soft64_to_f32_bits(nw64);
    const uint magnitude = nw & 0x7fffffffu;
    const bool nan = (magnitude & 0x7f800000u) == 0x7f800000u
                  && (magnitude & 0x007fffffu) != 0u;
    const bool nonpositive = magnitude == 0u
        || ((nw & 0x80000000u) != 0u && magnitude <= 0x7f800000u);
    r.surface[tid] = nan ? 2 : (nonpositive ? 1 : 0);
}
