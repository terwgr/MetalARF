// Compute the local atmosphere + surface water budget for resident physics.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_column_water_before_fragment_bindings.metalh"
using namespace metal;

kernel void global_column_water_before_fragment(
    constant GlobalColumnWaterBeforeFragmentResources &r [[buffer(0)]],
    constant GlobalColumnWaterBeforeFragmentParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= uint(p.ncolumn)) return;
    ulong total = r.surface_water[column];
    for (uint level = 0u; level < uint(p.nlevel); ++level) {
        const uint index = level * uint(p.ncolumn) + column;
        ulong mixing_ratio = soft64_add(r.qv[index], r.qc[index]);
        mixing_ratio = soft64_add(mixing_ratio, r.qr[index]);
        mixing_ratio = soft64_add(mixing_ratio, r.qi[index]);
        mixing_ratio = soft64_add(mixing_ratio, r.qs[index]);
        mixing_ratio = soft64_add(mixing_ratio, r.qg[index]);
        const ulong layer_water = soft64_div(
            soft64_mul(mixing_ratio, r.dp[index]), r.gravity[0]);
        total = soft64_add(total, layer_water);
    }
    r.before_water[column] = total;
}
