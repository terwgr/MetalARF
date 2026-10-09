#include <metal_stdlib>
#include "global_grid_mean_fragment_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;
#pragma clang fp contract(off)

// Area mean on the Gaussian grid. Each row keeps longitude's source order;
// the final thread combines weighted latitude rows and applies the 1/2 norm.
kernel void global_grid_mean_rows_fragment(
    constant GlobalGridMeanRowsFragmentResources &r [[buffer(0)]],
    constant GlobalGridMeanRowsFragmentParams &p [[buffer(1)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= uint(p.nlat)) return;
    ulong sum = 0ul;
    const uint base = row * uint(p.nlon);
    for (uint col = 0; col < uint(p.nlon); ++col)
        sum = soft64_add(sum, r.values[base + col]);
    const ulong mean = soft64_div(sum, soft64_from_i32(p.nlon));
    r.weighted_rows[row] = soft64_mul(mean, r.quadrature_weights[row]);
}

kernel void global_grid_mean_finalize_fragment(
    constant GlobalGridMeanFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalGridMeanFinalizeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    ulong sum = 0ul;
    for (uint row = 0; row < uint(p.nlat); ++row)
        sum = soft64_add(sum, r.weighted_rows[row]);
    r.mean[0] = soft64_mul(sum, 0x3fe0000000000000ul);
}
