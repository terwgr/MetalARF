#include <metal_stdlib>
#include "urban_bep_bem_profile_select_fragment_bindings.metalh"
using namespace metal;

// Select `ss_urb(:,iurb)` from the BEM class profile for each active domain
// column. WRF urban type ids are one-based; class profile rows are zero-based.
kernel void urban_bep_bem_profile_select_fragment(
    constant UrbanBepBemProfileSelectFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemProfileSelectFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    device const int *active = r.active_cols;
    device const int *utype = r.utype;
    device const float *ss_u = r.ss_u;
    device float *ss_urb = r.ss_urb;
    device int *error = r.error;
    error[col] = 0;
    if (active[col] == 0) return;
    const int row = utype[col] - 1;
    if (row < 0 || row >= p.nurbm) {
        error[col] = 5012;
        return;
    }
    const ulong src = ulong(row) * 19ul;
    const ulong dst = ulong(col) * 18ul;
    for (uint iz = 0; iz < 18u; ++iz)
        ss_urb[dst + iz] = ss_u[src + iz];
}
