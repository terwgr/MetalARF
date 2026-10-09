#include <metal_stdlib>
#include "urban_bep_bem_class_profile_pack_fragment_bindings.metalh"
using namespace metal;

// Pack the 19-level bb_icbep scratch into WRF's 18-level class descriptors.
kernel void urban_bep_bem_class_profile_pack_fragment(
    constant UrbanBepBemClassProfilePackFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemClassProfilePackFragmentParams &p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.nurbm)) return;
    device float *f = r.class_float;
    device int *i = r.class_int;
    const uint src = c * 19;
    const uint dst = c * 18;
    for (uint iz = 0; iz < 18; ++iz) {
        f[638 + dst + iz] = r.ss_profile[src + iz];
        f[836 + dst + iz] = r.pb_profile[src + iz];
        if (c == 0) f[1034 + iz] = r.z_u[iz];
    }
    i[12 + c] = r.nz_profile[c];
    if (r.profile_error[c] != 0) i[0] = r.profile_error[c];
}
