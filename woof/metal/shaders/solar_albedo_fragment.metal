// NOAA-EMC/HRRR v4.1.21 module_radiation_driver.F:781-789,1038-1063.
// Validation is separate so invalid MODIS classes leave persistent outputs
// untouched, matching the host and CUDA preflight behavior.
#include <metal_stdlib>
#include "solar_albedo_fragment_bindings.metalh"

using namespace metal;

#pragma clang fp contract(off)

kernel void solar_albedo_validate(
    constant SolarAlbedoValidateResources &r [[buffer(0)]],
    constant SolarAlbedoValidateParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const bool active = r.coszen[tid] > 0.0f && r.xland[tid] < 1.5f
                     && r.snow[tid] == 0.0f && r.xice[tid] == 0.0f;
    const int category = r.ivgtyp[tid];
    r.status[tid] = uint(active && (category < 1 || category > 21));
}

kernel void solar_albedo_update(
    constant SolarAlbedoUpdateResources &r [[buffer(0)]],
    constant SolarAlbedoUpdateParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    if (p.initialize != 0) {
        r.albsol[tid] = r.albedo[tid];
        r.albbcksol[tid] = r.albbck[tid];
    }
    if (r.coszen[tid] > 0.0f && r.xland[tid] < 1.5f
        && r.snow[tid] == 0.0f && r.xice[tid] == 0.0f) {
        const int category = r.ivgtyp[tid];
        const bool class_one = (category >= 1 && category <= 5)
                            || category == 8 || category == 9
                            || category == 15 || category == 18;
        const float d = class_one ? 0.1f : 0.25f;
        const float twice_d = 2.0f * d;
        const float numerator = 1.0f + twice_d;
        const float cosine_term = twice_d * r.coszen[tid];
        const float denominator = 1.0f + cosine_term;
        const float dm = numerator / denominator;
        const float normalized = r.albbck[tid] * dm;
        r.albsol[tid] = normalized;
        r.albbcksol[tid] = normalized;
    }
    r.albsol[tid] = fmin(r.albsol[tid], 0.9f);
}
