// Resident Global total atmospheric-water column before the SL water gate.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_water_column_fragment_bindings.metalh"
using namespace metal;

kernel void global_semilag_water_column_fragment(
    constant GlobalSemilagWaterColumnFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagWaterColumnFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncolumn = uint(p.nlat) * uint(p.nlon);
    if (tid >= ncolumn) return;
    ulong column = 0ul;
    for (uint level = 0u; level < uint(p.nlevel); ++level) {
        const uint index = level * ncolumn + tid;
        ulong mixing = soft64_add(r.qv[index], r.qc[index]);
        mixing = soft64_add(mixing, r.qr[index]);
        mixing = soft64_add(mixing, r.qi[index]);
        mixing = soft64_add(mixing, r.qs[index]);
        mixing = soft64_add(mixing, r.qg[index]);
        column = soft64_add(column, soft64_mul(mixing, r.dp[index]));
    }
    r.water_column[tid] = soft64_div(column, r.gravity[0]);
}
