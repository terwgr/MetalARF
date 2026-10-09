// Add a per-column precipitation plane to the resident surface reservoir.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_surface_water_add_fragment_bindings.metalh"
using namespace metal;

kernel void global_surface_water_add_fragment(
    constant GlobalSurfaceWaterAddFragmentResources &r [[buffer(0)]],
    constant GlobalSurfaceWaterAddFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    r.water_out[tid] = soft64_add(r.water[tid], r.precip[tid]);
}
