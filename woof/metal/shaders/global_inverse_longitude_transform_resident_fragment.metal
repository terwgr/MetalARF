// FP64 inverse longitude transform reading an interleaved resident waist.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_inverse_longitude_transform_resident_fragment_bindings.metalh"
using namespace metal;

kernel void global_inverse_longitude_transform_resident_fragment(
    constant GlobalInverseLongitudeTransformResidentFragmentResources &r [[buffer(0)]],
    constant GlobalInverseLongitudeTransformResidentFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint band_rows = uint(p.nlat), nlon = uint(p.nlon);
    const uint nfreq = uint(p.nfreq), batch = uint(p.batch);
    const uint count = batch * band_rows * nlon;
    if (tid >= count) return;
    const uint longitude = tid % nlon;
    const uint row = (tid / nlon) % band_rows;
    const uint field = tid / (band_rows * nlon);
    const uint global_row = uint(p.start_row) + row;
    const ulong base = (ulong(field) * uint(p.full_nlat) + global_row) * nfreq;
    const ulong nlon_bits = soft64_from_i32(p.nlon);
    const uint zero_twiddle = 2u * longitude;
    ulong sum = soft64_mul(
        soft64_mul(r.spectrum[2ul * base], nlon_bits),
        r.twiddles[zero_twiddle]);
    for (uint mode = 1; mode < nfreq; ++mode) {
        const uint twiddle = 2u * (mode * nlon + longitude);
        const ulong real = soft64_mul(r.spectrum[2ul * (base + mode)], nlon_bits);
        const ulong imag = soft64_mul(r.spectrum[2ul * (base + mode) + 1ul], nlon_bits);
        const ulong product = soft64_add(
            soft64_mul(real, r.twiddles[twiddle]),
            soft64_negate_bits(soft64_mul(imag, r.twiddles[twiddle + 1u])));
        const bool nyquist = ((nlon & 1u) == 0u && mode == nlon / 2u);
        sum = soft64_add(sum, nyquist ? product : soft64_add(product, product));
    }
    r.grid[tid] = sum;
}
