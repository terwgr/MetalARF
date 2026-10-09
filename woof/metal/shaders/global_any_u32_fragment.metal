// Associative nonzero reduction for small device-side diagnostic flags.
#include <metal_stdlib>
#include "global_any_u32_fragment_bindings.metalh"
using namespace metal;

kernel void global_any_u32_fragment(
    constant GlobalAnyU32FragmentResources &r [[buffer(0)]],
    constant GlobalAnyU32FragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count) || r.flags[tid] == 0u) return;
    atomic_fetch_or_explicit(
        reinterpret_cast<device atomic_uint *>(r.any), 1u,
        memory_order_relaxed);
}
