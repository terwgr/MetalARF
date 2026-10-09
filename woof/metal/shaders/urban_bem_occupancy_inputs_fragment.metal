#include <metal_stdlib>
#include "urban_bem_occupancy_inputs_fragment_bindings.metalh"
using namespace metal;

constant uint BEM_NBUIMAX = 15u;
constant uint BEM_HSEQUIP_OFFSET = 1239u;
constant uint BEM_PERFLO_OFFSET = 1195u;
constant uint BEM_HSESF_OFFSET = 1228u;

// Select the immutable per-class values used by bem_fluxeqocc. One item is
// one (column, possible building slot); downstream BEM loops consume only
// slots marked by building_nlev. hsequip is repeated into row-major item
// records because the occupancy kernel's ABI uses one 24-hour profile/item.
kernel void urban_bem_occupancy_inputs_fragment(
    constant UrbanBemOccupancyInputsFragmentResources &r [[buffer(0)]],
    constant UrbanBemOccupancyInputsFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= uint(p.nitem)) return;
    device const int *active = r.active_cols;
    device const int *utype = r.utype;
    device const float *class_float = r.class_float;
    device const float *bs = r.bs;
    device float *hsequip = r.hsequip;
    device float *building_width = r.building_width;
    device float *building_length = r.building_length;
    device float *floor_fraction = r.floor_fraction;
    device float *equipment_scale = r.equipment_scale;
    device int *error = r.error;
    const uint col = item / BEM_NBUIMAX;
    const int type = utype[col];
    error[item] = 0;
    if (active[col] == 0) {
        building_width[item] = 0.0f; building_length[item] = 0.0f;
        floor_fraction[item] = 0.0f; equipment_scale[item] = 0.0f;
        for (uint hour = 0; hour < 24u; ++hour)
            hsequip[ulong(item) * 24ul + hour] = 0.0f;
        return;
    }
    if (type < 1 || type > 11) { error[item] = 7510; return; }
    const uint cls = uint(type - 1);
    building_width[item] = bs[ulong(col) * 2ul];
    building_length[item] = bs[ulong(col) * 2ul + 1ul];
    floor_fraction[item] = class_float[BEM_PERFLO_OFFSET + cls];
    equipment_scale[item] = class_float[BEM_HSESF_OFFSET + cls];
    for (uint hour = 0; hour < 24u; ++hour)
        hsequip[ulong(item) * 24ul + hour] = class_float[BEM_HSEQUIP_OFFSET + hour];
}
