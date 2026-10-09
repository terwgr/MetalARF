#include <metal_stdlib>
#include "urban_bep_bem_hi_profile_gather_fragment_bindings.metalh"
using namespace metal;

// Reproduce bb_column's zero-initialized hi_urb1d followed by the copy of
// hi_urb2d(:,ix,iy) for iz=1..num_urban_hi.
kernel void urban_bep_bem_hi_profile_gather_fragment(
    constant UrbanBepBemHiProfileGatherFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemHiProfileGatherFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    device const float *hi_domain = r.hi_domain;
    device float *hi_column = r.hi_column;
    device int *error = r.error;
    error[col] = 0;
    if (p.num_urban_hi < 0 || p.num_urban_hi > 18) {
        error[col] = 5013;
        return;
    }
    for (uint iz = 0; iz < 18u; ++iz) {
        hi_column[ulong(col) * 18ul + iz] =
            iz < uint(p.num_urban_hi)
                ? hi_domain[ulong(iz) * ulong(p.ncol) + col]
                : 0.0f;
    }
}
