#include <metal_stdlib>
#include "urban_bem_building_level_scan_fragment_bindings.metalh"
using namespace metal;

constant uint BEM_SCAN_NZ = 18u;
constant uint BEM_SCAN_MAX = 15u;

// WRF module_sf_bep_bem.F:739-755. Compact positive ss_urb levels in
// ascending vertical order; the nlev and d_urb vectors retain their native
// 18-entry local extents while building_nlev is the 15-entry state-gather
// view used by later column packing.
kernel void urban_bem_building_level_scan_fragment(
    constant UrbanBemBuildingLevelScanFragmentResources &r [[buffer(0)]],
    constant UrbanBemBuildingLevelScanFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    device const int *active = r.active_cols;
    device const float *ss_urb = r.ss_urb;
    device int *nlev = r.nlev;
    device float *d_urb = r.d_urb;
    device int *nbui = r.nbui;
    device int *building_nlev = r.building_nlev;
    device int *error = r.error;

    if (col >= uint(p.ncol)) return;
    error[col] = 0;
    nbui[col] = 0;
    const ulong level_base = ulong(col) * BEM_SCAN_NZ;
    const ulong building_base = ulong(col) * BEM_SCAN_MAX;
    for (uint level = 0; level < BEM_SCAN_NZ; ++level) {
        nlev[level_base + level] = 0;
        d_urb[level_base + level] = 0.0f;
    }
    for (uint building = 0; building < BEM_SCAN_MAX; ++building)
        building_nlev[building_base + building] = 0;
    if (active[col] == 0) return;
    if (p.ncol <= 0) { error[col] = 6510; return; }

    uint building = 0u;
    for (uint iz = 1u; iz <= BEM_SCAN_NZ; ++iz) {
        const float spacing = ss_urb[ulong(col) * (BEM_SCAN_NZ + 1u) + iz];
        if (spacing > 0.0f) {
            if (building >= BEM_SCAN_MAX) {
                error[col] = 20;
                nbui[col] = int(building + 1u);
                return;
            }
            nlev[level_base + building] = int(iz - 1u);
            d_urb[level_base + building] = spacing;
            building_nlev[building_base + building] = int(iz - 1u);
            ++building;
        }
    }
    nbui[col] = int(building);
}
