#include <metal_stdlib>
using namespace metal;
#include "hexrrfs_w_boundary_bindings.metalh"

kernel void hexrrfs_w_boundary(
    constant HexrrfsWBoundaryResources &r [[buffer(0)]],
    constant HexrrfsWBoundaryParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.end)) return;
    int c = int(gid);
    if (c < p.first || r.mask[c] <= p.relaxation) return;
    // Nearest donors are relaxation cells and are never written by this kernel.
    device uint *w = reinterpret_cast<device uint *>(r.w);
    for (int k = 1; k < p.nlev; ++k)
        w[k * p.ncells + c] = p.zero ? 0u : w[k * p.ncells + r.nearest[c]];
}
