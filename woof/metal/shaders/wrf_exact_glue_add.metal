#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "wrf_exact_glue_add_bindings.metalh"
using namespace metal;

// bandwidth_glue.cu: each element uses __fadd_rn in the selected CUDA kernel.
kernel void wrf_exact_glue_add(
    constant WrfExactGlueAddResources &r [[buffer(0)]],
    constant WrfExactGlueAddParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (ulong(tid) >= p.n) return;
    device const uint *src = reinterpret_cast<device const uint *>(r.src);
    device uint *dst = reinterpret_cast<device uint *>(r.dst);
    dst[tid] = soft32_add(dst[tid], src[tid]);
}
