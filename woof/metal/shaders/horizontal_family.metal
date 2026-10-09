#include <metal_stdlib>
#include "horizontal_envelope_bindings.metalh"
using namespace metal;

// This entry point uses integer bit ordering only. The bounds resource is
// float32 storage for ABI compatibility, reinterpreted as three atomic words.
kernel void horizontal_envelope(
        constant HorizontalEnvelopeResources &r [[buffer(0)]],
        constant HorizontalEnvelopeParams &p [[buffer(1)]],
        uint flat [[thread_position_in_grid]],
        uint lane [[thread_index_in_threadgroup]],
        uint3 groups [[threadgroups_per_grid]]) {
    threadgroup uint lo[256];
    threadgroup uint hi[256];
    threadgroup uint bad[256];

    uint lower = 0xffffffffu;
    uint upper = 0u;
    uint invalid = 0u;
    const uint stride = groups.x * 256u;
    for (uint k = flat; k < uint(p.count); k += stride) {
        const uint bits = as_type<uint>(r.src[k]);
        invalid |= uint((bits & 0x7f800000u) == 0x7f800000u);
        const uint ordered = (bits >> 31) ? ~bits : (bits ^ 0x80000000u);
        lower = min(lower, ordered);
        upper = max(upper, ordered);
    }
    lo[lane] = lower;
    hi[lane] = upper;
    bad[lane] = invalid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128u; offset != 0u; offset >>= 1u) {
        if (lane < offset) {
            lo[lane] = min(lo[lane], lo[lane + offset]);
            hi[lane] = max(hi[lane], hi[lane + offset]);
            bad[lane] |= bad[lane + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        device atomic_uint *words = reinterpret_cast<device atomic_uint *>(r.bounds);
        atomic_fetch_min_explicit(words + 0, lo[0], memory_order_relaxed);
        atomic_fetch_max_explicit(words + 1, hi[0], memory_order_relaxed);
        atomic_fetch_or_explicit(words + 2, bad[0], memory_order_relaxed);
    }
}
