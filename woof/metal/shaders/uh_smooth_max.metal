#include <metal_stdlib>
#include "uh_smooth_max_bindings.metalh"

using namespace metal;

// Port of woof/core/kernels/uh_diag.cu::uh_smooth_max.  Each thread owns one
// mass-grid column.  The interior 3x3 smoother only reads uh; the running
// maximum is updated in place when usecol is set and the strict comparison
// succeeds. Boundary slots are deliberately left untouched for the caller's
// subsequent boundary-copy stage.
kernel void uh_smooth_max(constant UhSmoothMaxResources &r [[buffer(0)]],
                          constant UhSmoothMaxParams &p [[buffer(1)]],
                          uint col [[thread_position_in_grid]]) {
    const uint count = uint(p.ny) * uint(p.nx);
    if (col >= count) return;

    const int cy = int(col) / p.nx;
    const int cx = int(col) - cy * p.nx;
    if (cx < 1 || cx > p.nx - 2 || cy < 1 || cy > p.ny - 2) return;

    const ulong center = ulong(cy) * ulong(p.nx) + ulong(cx);
    const float t1 = 0.25f * r.uh[center];
    float edge = r.uh[center + 1] + r.uh[center - 1];
    edge = edge + r.uh[center + ulong(p.nx)];
    edge = edge + r.uh[center - ulong(p.nx)];
    float corner = r.uh[center + ulong(p.nx) + 1]
                 + r.uh[center - ulong(p.nx) + 1];
    corner = corner + r.uh[center + ulong(p.nx) - 1];
    corner = corner + r.uh[center - ulong(p.nx) - 1];
    const float uh_smth = (t1 + 0.125f * edge) + 0.0625f * corner;

    if (r.usecol[center] != 0.0f && uh_smth > r.up_heli_max[center])
        r.up_heli_max[center] = uh_smth;
}
