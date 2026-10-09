#include <metal_stdlib>
#include "emit_g_bindings.metalh"
#include "saxpy_bindings.metalh"

using namespace metal;

kernel void emit_g(constant EmitGResources &r [[buffer(0)]],
                   constant EmitGParams &q [[buffer(1)]],
                   uint tid [[thread_position_in_grid]]) {
    if (tid == 0) r.out[0] = q.gravity;
}

kernel void saxpy(constant SaxpyResources &r [[buffer(0)]],
                  constant SaxpyParams &q [[buffer(1)]],
                  uint gid [[thread_position_in_grid]]) {
    if (q.n <= 0) return;
    const uint block_size = uint(q.block_size);
    const uint block = gid / block_size;
    const uint lane = gid - block * block_size;
    const ulong n = ulong(q.n);

    if (q.vector_aligned) {
        const ulong first = ulong(block) * ulong(block_size) * 4ul;
        if (first >= n) return;
        const ulong i = first + ulong(lane) * 4ul;
        if (i + 3ul < n) {
            // Match CUDA's aligned float4 load, lane-wise expression, store.
            const device float4 *x4 = reinterpret_cast<device const float4 *>(r.x + i);
            const device float4 *y4 = reinterpret_cast<device const float4 *>(r.y + i);
            device float4 *out4 = reinterpret_cast<device float4 *>(r.out + i);
            const float4 xv = *x4;
            const float4 yv = *y4;
            float4 value;
            value.x = q.a * xv.x + yv.x;
            value.y = q.a * xv.y + yv.y;
            value.z = q.a * xv.z + yv.z;
            value.w = q.a * xv.w + yv.w;
            *out4 = value;
        } else {
            for (uint j = 0; j < 4; ++j) {
                const ulong at = i + ulong(j);
                if (at < n) r.out[at] = q.a * r.x[at] + r.y[at];
            }
        }
    } else {
        // Preserve CUDA's four block-strided scalar accesses per thread.
        const ulong i = ulong(block) * ulong(block_size) * 4ul + ulong(lane);
        for (uint j = 0; j < 4; ++j) {
            const ulong at = i + ulong(j) * ulong(block_size);
            if (at < n) r.out[at] = q.a * r.x[at] + r.y[at];
        }
    }
}
