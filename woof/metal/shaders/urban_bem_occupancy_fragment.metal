#include <metal_stdlib>
#include "urban_bem_occupancy_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmo_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmo_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmo_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmo_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }

// WRF module_sf_bem.F bem_phirat/phiequ/fluxeqocc (1888-1976). Source
// phirat is a constant 1.0 occupancy ratio; equipment profile is indexed by
// int(nhourday)+1 in one-based Fortran, so the flattened slot is int(hour).
kernel void urban_bem_occupancy_fragment(
    constant UrbanBemOccupancyFragmentResources &r [[buffer(0)]],
    constant UrbanBemOccupancyFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const float *hour_angle = r.hour_angle;
    device const float *hsequip = r.hsequip;
    device const float *building_width = r.building_width;
    device const float *building_length = r.building_length;
    device const float *floor_fraction = r.floor_fraction;
    device const float *equipment_scale = r.equipment_scale;
    device float *sensible_gain = r.sensible_gain;
    device float *latent_gain = r.latent_gain;
    device int *error = r.error;
    const int nitem = p.nitem;
    if (item >= uint(nitem)) return;
    // module_sf_bep_bem.F:1490-1492, per column from the solar hour angle.
    float nhourday = ubmo_add(ubmo_div(ubmo_mul(ubmo_div(hour_angle[item / 15u],
        3.141592653f), 180.0f), 15.0f), 12.0f);
    if (nhourday >= 24.0f) nhourday = ubmo_sub(nhourday, 24.0f);
    if (nhourday < 0.0f) nhourday = ubmo_add(nhourday, 24.0f);
    const int hour = int(nhourday);
    if (hour < 0 || hour >= 24) { error[item] = 1915; return; }
    const float area = ubmo_mul(building_width[item], building_length[item]);
    const float equipment = ubmo_mul(hsequip[ulong(item) * 24ul + uint(hour)], equipment_scale[item]);
    const float occupied_sensible = 160.0f;
    const float occupied_latent = ubmo_div(1.96e6f, 86400.0f);
    sensible_gain[item] = ubmo_add(
        ubmo_mul(ubmo_mul(ubmo_mul(area, floor_fraction[item]), occupied_sensible), 1.0f),
        ubmo_mul(area, equipment));
    latent_gain[item] = ubmo_mul(
        ubmo_mul(ubmo_mul(area, floor_fraction[item]), occupied_latent), 1.0f);
    error[item] = 0;
}
