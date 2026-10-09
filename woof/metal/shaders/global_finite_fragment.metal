// Chunked finite-value reduction for resident Global guards.
#include <metal_stdlib>
#include "global_finite_fragment_bindings.metalh"
#include "global_finite_finalize_fragment_bindings.metalh"
using namespace metal;

kernel void global_finite_fragment(
    constant GlobalFiniteFragmentResources &r [[buffer(0)]],
    constant GlobalFiniteFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.partial_count)) return;
    const uint start = tid * uint(p.chunk_size);
    const uint end = min(uint(p.count), start + uint(p.chunk_size));
    const ulong exponent = 0x7ff0000000000000ul;
    uint nonfinite = 0u;
    for (uint i = start; i < end; ++i) {
        if ((r.values[i] & exponent) == exponent) nonfinite = 1u;
    }
    r.partial_nonfinite[tid] = nonfinite;
}

kernel void global_finite_finalize_fragment(
    constant GlobalFiniteFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalFiniteFinalizeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    uint finite = 1u;
    for (uint i = 0u; i < uint(p.partial_count); ++i) {
        if (r.partial_nonfinite[i] != 0u) finite = 0u;
    }
    r.finite[0] = finite;
}
