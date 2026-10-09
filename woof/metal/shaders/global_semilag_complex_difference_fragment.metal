// Magnitude of a complex spectral difference for Global SL diagnostics.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_complex_difference_fragment_bindings.metalh"
using namespace metal;

kernel void global_semilag_complex_difference_fragment(
    constant GlobalSemilagComplexDifferenceFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagComplexDifferenceFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const uint component = 2u * tid;
    const ulong real = soft64_sub(r.after[component], r.before[component]);
    const ulong imag = soft64_sub(r.after[component + 1u], r.before[component + 1u]);
    const ulong magnitude = soft64_sqrt(soft64_add(
        soft64_mul(real, real), soft64_mul(imag, imag)));
    r.magnitudes[tid] = magnitude;
}
