// FP64 latitude contraction over an interleaved complex MetalArray waist.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_latitude_transform_resident_fragment_bindings.metalh"
using namespace metal;

kernel void global_latitude_transform_resident_fragment(
    constant GlobalLatitudeTransformResidentFragmentResources &r [[buffer(0)]],
    constant GlobalLatitudeTransformResidentFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.norder) * uint(p.batch) * uint(p.noutput);
    if (tid >= count) return;
    const uint output_lat = tid % uint(p.noutput);
    const uint batch = (tid / uint(p.noutput)) % uint(p.batch);
    const uint local_order = tid / (uint(p.noutput) * uint(p.batch));
    const uint mode = uint(p.order_start) + local_order;
    const ulong input_base = ulong(batch) * uint(p.ninput) * uint(p.nmodes);
    const ulong table_base = ulong(local_order) * uint(p.ninput) * uint(p.noutput)
        + output_lat;
    ulong real_sum = 0ul, imag_sum = 0ul;
    for (uint input_lat = 0; input_lat < uint(p.ninput); ++input_lat) {
        const ulong input_index = 2ul * (input_base
            + ulong(input_lat) * uint(p.nmodes) + mode);
        const ulong weight = r.legendre[table_base
            + ulong(input_lat) * uint(p.noutput)];
        real_sum = soft64_add(real_sum,
            soft64_mul(r.values[input_index], weight));
        imag_sum = soft64_add(imag_sum,
            soft64_mul(r.values[input_index + 1ul], weight));
    }
    const ulong output_index = 2ul * (ulong(batch) * uint(p.noutput) * uint(p.nmodes)
        + ulong(output_lat) * uint(p.nmodes) + mode);
    r.spectral[output_index] = real_sum;
    r.spectral[output_index + 1ul] = mode == 0u ? 0ul : imag_sum;
}
