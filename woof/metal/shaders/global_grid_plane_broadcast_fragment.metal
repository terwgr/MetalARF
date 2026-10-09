// Broadcast a resident FP64 grid plane through the vertical levels.
#include <metal_stdlib>
#include "global_grid_plane_broadcast_fragment_bindings.metalh"

using namespace metal;

kernel void global_grid_plane_broadcast_fragment(
    constant GlobalGridPlaneBroadcastFragmentResources &r [[buffer(0)]],
    constant GlobalGridPlaneBroadcastFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const uint plane_index = tid % uint(p.nlat * p.nlon);
    r.volume[tid] = r.plane[plane_index];
}
