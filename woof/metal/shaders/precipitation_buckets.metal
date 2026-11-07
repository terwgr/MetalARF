// Port of woof/core/kernels/output_diagnostics.cu::precipitation_buckets.
// The CUDA expressions promote through binary64 ((reset?0.:x)+amount), so the
// sums are formed in soft binary64 and rounded once to binary32.
#include <metal_stdlib>
#include "precipitation_buckets_bindings.metalh"
#include "soft_binary64.metalh"

using namespace metal;

static inline ulong pb_wide(float value) { return soft64_from_f32_bits(as_type<uint>(value)); }

// fmaxf(0., sum): NaN and negative values (including -0) give +0.
static inline float pb_floor(ulong sum) {
    const uint bits = soft64_to_f32_bits(sum);
    if ((bits & 0x7fffffffu) > 0x7f800000u || (bits & 0x80000000u) != 0u) return 0.0f;
    return as_type<float>(bits);
}

kernel void precipitation_buckets(constant PrecipitationBucketsResources &r [[buffer(0)]],
                                  constant PrecipitationBucketsParams &p [[buffer(1)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.count)) return;
    const float c = r.rainc[i];
    const bool reset = p.reset != 0;
    const ulong zero = 0ul;
    r.pc[i] = pb_floor(soft64_add(reset ? zero : pb_wide(r.pc[i]), pb_wide(c)));
    r.pn[i] = pb_floor(soft64_add(reset ? zero : pb_wide(r.pn[i]), pb_wide(r.rain[i])));
    const float frozen = r.t2[i] < 273.15f ? c : 0.0f;
    ulong sum = soft64_add(reset ? zero : pb_wide(r.sn[i]), pb_wide(r.snow[i]));
    r.sn[i] = pb_floor(soft64_add(sum, pb_wide(frozen)));
    r.gn[i] = pb_floor(soft64_add(reset ? zero : pb_wide(r.gn[i]), pb_wide(r.graup[i])));
}
