// Global in-situ spectral kinetic energy from resident complex FP64 state.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_insitu_spectra_fragment_bindings.metalh"
using namespace metal;

kernel void global_insitu_spectra_modes_fragment(
    constant GlobalInsituSpectraModesFragmentResources &r [[buffer(0)]],
    constant GlobalInsituSpectraModesFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint nterm = uint(p.nterm), nlevel = uint(p.nlevel);
    const uint total = 2u * nlevel * nterm;
    if (tid >= total) return;
    const uint field = tid / (nlevel * nterm);
    const uint rem = tid - field * nlevel * nterm;
    const uint level = rem / nterm, degree = rem % nterm;
    device const ulong *coeff = field == 0u ? r.vorticity : r.divergence;
    ulong sum = 0ul;
    for (uint order = 0; order <= degree; ++order) {
        const uint index = level * nterm * nterm + degree * nterm + order;
        const ulong real_bits = coeff[2u * index];
        const ulong imag_bits = coeff[2u * index + 1u];
        const ulong magnitude = soft64_add(
            soft64_mul(real_bits, real_bits), soft64_mul(imag_bits, imag_bits));
        sum = soft64_add(sum, soft64_mul(magnitude, r.order_weight[order]));
    }
    r.per_degree[tid] = soft64_mul(sum, r.factor[degree]);
}

kernel void global_insitu_spectra_pack_fragment(
    constant GlobalInsituSpectraPackFragmentResources &r [[buffer(0)]],
    constant GlobalInsituSpectraPackFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint nterm = uint(p.nterm), nlevel = uint(p.nlevel);
    const uint band_width = 2u * nlevel;
    const uint row_width = 2u * nterm + 4u * nlevel;
    if (tid >= row_width) return;
    if (tid < 2u * nterm) {
        const uint field = tid / nterm, degree = tid % nterm;
        ulong sum = 0ul;
        for (uint level = 0; level < nlevel; ++level)
            sum = soft64_add(sum, r.per_degree[field * nlevel * nterm
                + level * nterm + degree]);
        r.row[tid] = sum;
        return;
    }
    const uint top_start = 2u * nterm;
    if (tid < top_start + band_width) {
        const uint offset = tid - top_start;
        const uint field = offset / nlevel, level = offset % nlevel;
        ulong sum = 0ul;
        for (uint degree = uint(p.band_start); degree < nterm; ++degree)
            sum = soft64_add(sum, r.per_degree[field * nlevel * nterm
                + level * nterm + degree]);
        r.row[tid] = sum;
        return;
    }
    const uint offset = tid - top_start - band_width;
    const uint field = offset / nlevel, level = offset % nlevel;
    ulong sum = 0ul;
    for (uint degree = 0; degree < nterm; ++degree)
        sum = soft64_add(sum, r.per_degree[field * nlevel * nterm
            + level * nterm + degree]);
    r.row[tid] = sum;
}
