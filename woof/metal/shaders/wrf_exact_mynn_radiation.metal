// Strict radiation MYNN cloud merge, matching the caller's binary32 order.
#include <metal_stdlib>
#include "rrtmgp_mynn_phase.metalh"
#include "wrf_exact_mynn_radiation_bindings.metalh"
using namespace metal;
kernel void wrf_exact_rrtmgp_mynn_cloud_merge(
    constant WrfExactRrtmgpMynnCloudMergeResources &r [[buffer(0)]],
    constant WrfExactRrtmgpMynnCloudMergeParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.ncell)) return;
    rrtmgp_merge_mynn_legacy(r, p, i);
}

kernel void wrf_exact_rrtmgp_mynn_cloud_merge_gsd41(
    constant WrfExactRrtmgpMynnCloudMergeGsd41Resources &r [[buffer(0)]],
    constant WrfExactRrtmgpMynnCloudMergeGsd41Params &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.ncell)) return;
    rrtmgp_merge_mynn_gsd41(r, p, i);
}
