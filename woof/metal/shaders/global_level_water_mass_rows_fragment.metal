// FP64 longitude-row means of tracer mass: sum(q * dp) / nlon.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_level_water_mass_rows_fragment_bindings.metalh"
using namespace metal;

kernel void global_level_water_mass_rows_fragment(
    constant GlobalLevelWaterMassRowsFragmentResources &r [[buffer(0)]],
    constant GlobalLevelWaterMassRowsFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint row_count = uint(p.nspecies) * uint(p.nlev) * uint(p.nlat);
    if (tid >= row_count) return;
    const uint plane = uint(p.nlat) * uint(p.nlon);
    const uint row = tid * uint(p.nlon);
    const uint level = (tid / uint(p.nlat)) % uint(p.nlev);
    const uint dp_row = level * plane + (tid % uint(p.nlat)) * uint(p.nlon);
    ulong sum = 0ul;
    for (uint lon = 0; lon < uint(p.nlon); ++lon) {
        sum = soft64_add(sum, soft64_mul(
            r.stacked[row + lon], r.dp[dp_row + lon]));
    }
    r.rows[tid] = soft64_div(sum, soft64_from_i32(p.nlon));
}
