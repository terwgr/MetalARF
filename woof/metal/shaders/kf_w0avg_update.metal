#include <metal_stdlib>
#include "kf_w0avg_update_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

kernel void kf_w0avg_update(
    constant KfW0AvgUpdateResources &r [[buffer(0)]],
    constant KfW0AvgUpdateParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    const uint nz = uint(p.nz), ny = uint(p.ny), nx = uint(p.nx);
    const ulong ncol = ulong(ny) * ulong(nx);
    const ulong count = ulong(nz) * ncol;
    if (ulong(index) >= count) return;

    const ulong idx = ulong(index);
    const float old = r.w0avg[idx];
    const float instantaneous = 0.5f * (r.w[idx] + r.w[idx + ncol]);
    const float weighted = old * (p.tst - 1.0f);
    r.w0avg[idx] = (weighted + instantaneous) / p.tst;
}
