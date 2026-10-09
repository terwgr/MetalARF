// needs integration: FP64 real-grid to longitude-Fourier transform.
//
// The ABI owner must register this fragment before the Global transform can
// dispatch it. `twiddles` is a host-built (nlon/2+1, nlon) row-major table of
// IEEE-754 binary64 cosine and negative-sine bit pairs. `scale` is the
// binary64 reciprocal of nlon. The result is an unnormalized NumPy-compatible
// rFFT scaled by 1/nlon, laid out [..., nlat, nlon/2+1, real/imag].
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_longitude_transform_fragment_bindings.metalh"

using namespace metal;

kernel void global_longitude_transform_fragment(
    constant GlobalLongitudeTransformFragmentResources &r [[buffer(0)]],
    constant GlobalLongitudeTransformFragmentParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint nlat = uint(q.nlat), nlon = uint(q.nlon);
    const uint nfreq = uint(q.nfreq), batch = uint(q.batch);
    const uint output_count = batch * nlat * nfreq;
    if (tid >= output_count) return;
    const uint mode = tid % nfreq;
    const uint row = (tid / nfreq) % nlat;
    const uint field = tid / (nlat * nfreq);
    ulong real_sum = 0ul;
    ulong imag_sum = 0ul;
    for (uint longitude = 0u; longitude < nlon; ++longitude) {
        const ulong value = r.grid[(field * nlat + row) * nlon + longitude];
        const uint twiddle = 2u * (mode * nlon + longitude);
        real_sum = soft64_add(real_sum,
            soft64_mul(value, r.twiddles[twiddle]));
        imag_sum = soft64_add(imag_sum,
            soft64_mul(value, r.twiddles[twiddle + 1u]));
    }
    r.spectrum[2u * tid] = soft64_mul(real_sum, r.scale[0]);
    r.spectrum[2u * tid + 1u] = soft64_mul(imag_sum, r.scale[0]);
}
