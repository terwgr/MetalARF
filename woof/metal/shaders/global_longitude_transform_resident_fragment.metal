// FP64 longitude transform that writes directly into a resident whole-field waist.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_longitude_transform_resident_fragment_bindings.metalh"
using namespace metal;

kernel void global_longitude_transform_resident_fragment(
    constant GlobalLongitudeTransformResidentFragmentResources &r [[buffer(0)]],
    constant GlobalLongitudeTransformResidentFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint band_rows = uint(p.nlat), nlon = uint(p.nlon);
    const uint nfreq = uint(p.nfreq), batch = uint(p.batch);
    const uint count = batch * band_rows * nfreq;
    if (tid >= count) return;
    const uint mode = tid % nfreq;
    const uint row = (tid / nfreq) % band_rows;
    const uint field = tid / (band_rows * nfreq);
    const uint global_row = uint(p.start_row) + row;
    const ulong source_base = (ulong(field) * uint(p.full_nlat) + global_row) * nlon;
    const ulong output_base = (ulong(field) * uint(p.full_nlat) + global_row) * nfreq + mode;
    ulong real_sum = 0ul, imag_sum = 0ul;
    for (uint longitude = 0; longitude < nlon; ++longitude) {
        const ulong value = r.grid[source_base + longitude];
        const uint twiddle = 2u * (mode * nlon + longitude);
        real_sum = soft64_add(real_sum, soft64_mul(value, r.twiddles[twiddle]));
        imag_sum = soft64_add(imag_sum, soft64_mul(value, r.twiddles[twiddle + 1u]));
    }
    r.spectrum[2ul * output_base] = soft64_mul(real_sum, r.scale[0]);
    r.spectrum[2ul * output_base + 1ul] = soft64_mul(imag_sum, r.scale[0]);
}
