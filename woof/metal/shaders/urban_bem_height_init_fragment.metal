#include <metal_stdlib>
#include "urban_bem_height_init_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmhi_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmhi_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }

// WRF module_sf_bep_bem.F bb_icbephi_xy (5556-5633). One item owns one
// urban class's profile weights. Invalid/nonpositive profiles take the
// source L20 path and leave nzu/ss/pb zero. Class extents follow nz_um=18.
kernel void urban_bem_height_init_fragment(
    constant UrbanBemHeightInitFragmentResources &r [[buffer(0)]],
    constant UrbanBemHeightInitFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const float *z_u = r.z_u;
    device float *hi_urb1d = r.hi_urb1d;
    device float *hb_u = r.hb_u;
    device float *ss_u = r.ss_u;
    device float *pb_u = r.pb_u;
    device int *nzu_out = r.nzu_out;
    device int *error = r.error;
    const int nitem = p.nitem;
    const float dz_u = p.dz_u;
    if (item >= uint(nitem)) return;
    if (nitem < 1) { error[item] = 5556; nzu_out[item] = 0; return; }
    const ulong base = ulong(item) * 18ul;
    const ulong profile = ulong(item) * 19ul;
    for (int iz = 0; iz < 19; ++iz) { ss_u[profile + uint(iz)] = 0.0f; pb_u[profile + uint(iz)] = 0.0f; }
    nzu_out[item] = 0;
    float total = 0.0f;
    for (int iz = 0; iz < 18; ++iz) {
        hb_u[base + uint(iz)] = 0.0f;
        total = ubmhi_add(total, hi_urb1d[base + uint(iz)]);
        if (hi_urb1d[base + uint(iz)] < 0.0f) { error[item] = 0; return; }
    }
    if (!(total > 0.0f)) { error[item] = 0; return; }
    for (int iz = 0; iz < 18; ++iz)
        hi_urb1d[base + uint(iz)] = ubmhi_div(hi_urb1d[base + uint(iz)], total);
    float hbmax = 0.0f;
    float building_height = dz_u;
    for (int iz = 0; iz < 18; ++iz) {
        if (iz > 0) building_height = ubmhi_add(dz_u, building_height);
        hb_u[base + uint(iz)] = building_height;
        const float fraction = hi_urb1d[base + uint(iz)];
        if (fraction > 0.0f && fraction <= 1.0f) hbmax = hb_u[base + uint(iz)];
    }
    int nz = 19;
    for (int iz = 0; iz < 17; ++iz) {
        if (z_u[uint(iz + 1)] > hbmax) { nz = iz + 2; break; }
    }
    if (nz >= 18) { error[item] = 12; return; }
    nzu_out[item] = nz;
    for (int iz = 0; iz < nz; ++iz) {
        float fraction = 0.0f;
        for (int bld = 0; bld < 18; ++bld) {
            if (z_u[uint(iz)] <= hb_u[base + uint(bld)] &&
                z_u[uint(iz + 1)] > hb_u[base + uint(bld)])
                fraction = ubmhi_add(fraction, hi_urb1d[base + uint(bld)]);
        }
        ss_u[profile + uint(iz)] = fraction;
    }
    pb_u[profile] = 1.0f;
    for (int iz = 0; iz < nz; ++iz)
        pb_u[profile + uint(iz + 1)] = max(0.0f, pb_u[profile + uint(iz)] - ss_u[profile + uint(iz)]);
    error[item] = 0;
}
