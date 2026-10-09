// Reduce negative and positive vapor mass along longitude for each level/row.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_enforce_vapor_rows_fragment_bindings.metalh"
using namespace metal;

kernel void global_enforce_vapor_rows_fragment(
    constant GlobalEnforceVaporRowsFragmentResources &r [[buffer(0)]],
    constant GlobalEnforceVaporRowsFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint rows = uint(p.nlev) * uint(p.band_nlat);
    if (tid >= rows) return;
    const uint level = tid / uint(p.band_nlat);
    const uint latitude = tid % uint(p.band_nlat);
    const ulong source = (ulong(level) * uint(p.full_nlat)
                          + uint(p.start_latitude) + latitude)
                         * uint(p.nlon);
    ulong negative = 0ul;
    ulong positive = 0ul;
    for (uint longitude = 0u; longitude < uint(p.nlon); ++longitude) {
        const ulong mass = soft64_mul(
            r.qv[source + longitude], r.dp[source + longitude]);
        negative = soft64_add(
            negative, soft64_less(mass, 0ul) ? mass : 0ul);
        positive = soft64_add(
            positive, soft64_greater(mass, 0ul) ? mass : 0ul);
    }
    const ulong output = ulong(level) * uint(p.full_nlat)
                         + uint(p.start_latitude) + latitude;
    r.negative_rows[output] = negative;
    r.positive_rows[output] = positive;
}
