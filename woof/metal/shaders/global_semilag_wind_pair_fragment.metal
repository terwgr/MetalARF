// Stack two resident Global wind components for the vector Fourier analysis.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_wind_pair_fragment_bindings.metalh"
using namespace metal;

kernel void global_semilag_wind_pair_fragment(
    constant GlobalSemilagWindPairFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagWindPairFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    r.pair[tid] = r.u[tid];
    r.pair[uint(p.count) + tid] = r.v[tid];
}
