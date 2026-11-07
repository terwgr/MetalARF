#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "wrf_exact_glue_add_positive_bindings.metalh"
using namespace metal;

// CuPy maximum(src, +0) then an independent FP32 add, with strict FTZ off.
kernel void wrf_exact_glue_add_positive(
    constant WrfExactGlueAddPositiveResources &r [[buffer(0)]],
    constant WrfExactGlueAddPositiveParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    device const uint *src = reinterpret_cast<device const uint *>(r.src);
    device uint *dst = reinterpret_cast<device uint *>(r.dst);
    const uint x = src[gid];
    const uint positive = soft32_greater(x, 0u) ? x : 0u;
    dst[gid] = soft32_add(dst[gid], positive);
}
