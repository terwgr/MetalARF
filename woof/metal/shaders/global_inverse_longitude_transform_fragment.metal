#include <metal_stdlib>
#include "global_inverse_longitude_transform_fragment_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;

#pragma clang fp contract(off)

// Drain the positive-frequency waist used by NumPy irfft. The caller
// multiplies the retained spectrum by nlon before this stage, matching the
// inverse FFT input; the twiddle table carries irfft's final 1/nlon scale.
kernel void global_inverse_longitude_transform_fragment(
    constant GlobalInverseLongitudeTransformFragmentResources &r [[buffer(0)]],
    constant GlobalInverseLongitudeTransformFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint nlat = uint(p.nlat), nlon = uint(p.nlon);
    const uint nfreq = uint(p.nfreq), batch = uint(p.batch);
    const uint count = batch * nlat * nlon;
    if (tid >= count) return;
    const uint longitude = tid % nlon;
    const uint row = (tid / nlon) % nlat;
    const uint field = tid / (nlat * nlon);
    const ulong spectrum_base = (ulong(field) * nlat + row) * nfreq;
    const uint zero_twiddle = 2u * longitude;
    ulong sum = soft64_mul(r.real_spectrum[spectrum_base],
                           r.twiddles[zero_twiddle]);
    for (uint mode = 1u; mode < nfreq; ++mode) {
        const uint twiddle = 2u * (mode * nlon + longitude);
        const ulong re = r.real_spectrum[spectrum_base + mode];
        const ulong im = r.imag_spectrum[spectrum_base + mode];
        const ulong cosine = r.twiddles[twiddle];
        const ulong sine = r.twiddles[twiddle + 1u];
        const ulong product = soft64_add(
            soft64_mul(re, cosine),
            soft64_negate_bits(soft64_mul(im, sine)));
        const bool nyquist = ((nlon & 1u) == 0u && mode == nlon / 2u);
        sum = soft64_add(sum, nyquist ? product : soft64_add(product, product));
    }
    r.grid[tid] = sum;
}
