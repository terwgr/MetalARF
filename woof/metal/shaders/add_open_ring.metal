// target += plane on the width-1 outer ring that _zero_open_strips clears at
// radiative open boundaries, each ring word once (dycore._add_open_ring).
#include <metal_stdlib>
#include "add_open_ring_bindings.metalh"

using namespace metal;

kernel void add_open_ring(constant AddOpenRingResources &r [[buffer(0)]],
                          constant AddOpenRingParams &p [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ny) * uint(p.nx)) return;
    const int j = int(tid) / p.nx, i = int(tid) - j * p.nx;
    const bool column = p.open_x != 0 && (i == 0 || i == p.nx - 1);
    const bool row = p.open_y != 0 && (j == 0 || j == p.ny - 1)
                  && (p.open_x == 0 || (i >= 1 && i <= p.nx - 2));
    if (column || row) r.target[tid] = r.target[tid] + r.plane[tid];
}
