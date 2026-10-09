// needs integration: gather per-column BEM canyon geometry without host staging.
#include <metal_stdlib>
#include "urban_bem_view_inputs_fragment_bindings.metalh"
using namespace metal;

kernel void urban_bem_view_inputs_fragment(
    constant UrbanBemViewInputsFragmentResources &r [[buffer(0)]],
    constant UrbanBemViewInputsFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    const uint count = uint(p.ncol) * uint(p.nd);
    if (item >= count) return;
    const uint col = item / uint(p.nd);
    const uint direction = item % uint(p.nd);
    const int category = r.utype[col];
    if (category < 0 || category > p.nclasses) {
        r.error[item] = 4875;
        r.dxy[item] = 0.0f;
        r.street_width[item] = 0.0f;
        return;
    }
    const uint class_index = category > 0 ? uint(category - 1) : 0u;
    r.dxy[item] = r.street_direction_table[
        direction * uint(p.nclasses) + class_index];
    r.street_width[item] = r.street_width_table[
        direction * uint(p.nclasses) + class_index];
    r.error[item] = 0;
    if (direction == 0u) {
        const ulong zbase = ulong(col) * 19ul;
        for (uint level = 0; level < 19u; ++level)
            r.z[zbase + level] = 5.0f * float(level);
    }
}
