// Rectangle movement for Metal tile/nest boundary tables. The shape
// convention is contiguous [k,j,i], with the packed destination beginning at
// (j0,i0) in the source rectangle.

#include <metal_stdlib>
#include "nest_rect_gather_f32_bindings.metalh"
#include "nest_rect_scatter_f32_bindings.metalh"
using namespace metal;

static inline ulong nest_rect_i3(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx) + ulong(i);
}

kernel void nest_rect_gather_f32(
    constant NestRectGatherF32Resources &r [[buffer(0)]],
    constant NestRectGatherF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong count = ulong(p.nz) * ulong(p.ny) * ulong(p.nx);
    if (ulong(tid) >= count) return;
    const int i = int(tid % uint(p.nx));
    const int j = int((tid / uint(p.nx)) % uint(p.ny));
    const int k = int(tid / (ulong(p.nx) * ulong(p.ny)));
    r.packed_a[tid] = r.source_a[nest_rect_i3(
        k, p.j0 + j, p.i0 + i, p.source_ny, p.source_nx)];
}

kernel void nest_rect_scatter_f32(
    constant NestRectScatterF32Resources &r [[buffer(0)]],
    constant NestRectScatterF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong count = ulong(p.nz) * ulong(p.ny) * ulong(p.nx);
    if (ulong(tid) >= count) return;
    const int i = int(tid % uint(p.nx));
    const int j = int((tid / uint(p.nx)) % uint(p.ny));
    const int k = int(tid / (ulong(p.nx) * ulong(p.ny)));
    r.target_a[nest_rect_i3(
        k, p.j0 + j, p.i0 + i, p.target_ny, p.target_nx)] = r.packed_a[tid];
}
