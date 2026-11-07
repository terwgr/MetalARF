#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "thompson_vapour_fill_bindings.metalh"
using namespace metal;

kernel void thompson_negative_vapour_fill(
    constant ThompsonNegativeVapourFillResources &r [[buffer(0)]],
    constant ThompsonNegativeVapourFillParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    ulong ncol = ulong(p.ny) * ulong(p.nx);
    if (gid >= ncol) return;
    device uint *qv = reinterpret_cast<device uint *>(r.qv);
    const uint floor = as_type<uint>(1.0e-7f);
    for (int k = 0; k < p.nz; ++k) {
        ulong index = ulong(k) * ncol + gid;
        uint value = qv[index];
        if (value > 0x80000000u && value <= 0xff800000u) {
            if (k > 1 && k < p.nz - 3) {
                uint mean = soft32_mul(0x3f000000u, soft32_add(
                    qv[index - ncol], qv[index + ncol]));
                // CUDA fmaxf selects the finite floor when the mean is NaN.
                qv[index] = soft32_is_nan(mean) || soft32_compare(mean, floor) < 0
                    ? floor : mean;
            } else {
                qv[index] = floor;
            }
        }
    }
}
