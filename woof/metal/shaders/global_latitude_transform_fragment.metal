#include <metal_stdlib>
#include "global_latitude_transform_fragment_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;

#pragma clang fp contract(off)

// One output owns (order, batch, output latitude); input order is
// [order, batch, input latitude], table order is [order, input, output].
kernel void global_latitude_transform_fragment(
    constant GlobalLatitudeTransformFragmentResources &r [[buffer(0)]],
    constant GlobalLatitudeTransformFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = p.norder * p.batch * p.noutput;
    if (tid >= count) return;
    const uint output = tid % p.noutput;
    const uint batch = (tid / p.noutput) % p.batch;
    const uint order = tid / (p.noutput * p.batch);
    const uint input_base = (order * p.batch + batch) * p.ninput;
    const uint table_base = order * p.ninput * p.noutput + output;
    ulong real_sum = 0ul;
    ulong imag_sum = 0ul;
    for (uint input = 0; input < p.ninput; ++input) {
        const ulong weight = r.legendre[table_base + input * p.noutput];
        real_sum = soft64_add(real_sum,
            soft64_mul(r.real_values[input_base + input], weight));
        imag_sum = soft64_add(imag_sum,
            soft64_mul(r.imag_values[input_base + input], weight));
    }
    r.real_spectral[tid] = real_sum;
    r.imag_spectral[tid] = imag_sum;
}
