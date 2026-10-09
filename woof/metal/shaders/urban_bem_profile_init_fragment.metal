#include <metal_stdlib>
#include "urban_bem_profile_init_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmpi_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmpi_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmpi_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }

// WRF module_sf_bep_bem.F bb_icbep (4757-4819). One item owns one urban
// class. It normalizes building-height counts, computes its active urban
// layer count, layer fractions, and remaining cumulative profile. This is a
// source-only setup stage; its fixed source limits are nz_um=18 and nurbmax=11.
kernel void urban_bem_profile_init_fragment(
    constant UrbanBemProfileInitFragmentResources &r [[buffer(0)]],
    constant UrbanBemProfileInitFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const float *z_u = r.z_u;
    device const float *h_b = r.h_b;
    device float *d_b = r.d_b;
    device float *ss_u = r.ss_u;
    device float *pb_u = r.pb_u;
    device int *nz_u = r.nz_u;
    device int *error = r.error;
    const int nurbm = p.nurbm;
    if (item >= uint(nurbm)) return;
    if (nurbm < 1 || nurbm > 11) { error[item] = 4757; return; }
    const ulong base = ulong(item) * 18ul;
    const ulong profile = ulong(item) * 19ul;
    float total = 0.0f;
    for (int iz = 0; iz < 18; ++iz)
        total = ubmpi_add(total, d_b[base + uint(iz)]);
    if (!(total > 0.0f)) { nz_u[item] = 0; error[item] = 4773; return; }
    for (int iz = 0; iz < 18; ++iz)
        d_b[base + uint(iz)] = ubmpi_div(d_b[base + uint(iz)], total);

    float hbmax = 0.0f;
    for (int iz = 0; iz < 18; ++iz) hbmax = max(hbmax, h_b[base + uint(iz)]);
    int nz = 18;
    for (int iz = 0; iz < 17; ++iz) {
        if (z_u[uint(iz + 1)] > hbmax) { nz = iz + 2; break; }
    }
    nz_u[item] = nz;
    for (int iz = 0; iz < 19; ++iz) { ss_u[profile + uint(iz)] = 0.0f; pb_u[profile + uint(iz)] = 0.0f; }
    for (int iz = 0; iz < nz; ++iz) {
        float fraction = 0.0f;
        for (int bld = 0; bld < 18; ++bld) {
            if (z_u[uint(iz)] <= h_b[base + uint(bld)] &&
                z_u[uint(iz + 1)] > h_b[base + uint(bld)])
                fraction = ubmpi_add(fraction, d_b[base + uint(bld)]);
        }
        ss_u[profile + uint(iz)] = fraction;
    }
    pb_u[profile] = 1.0f;
    for (int iz = 0; iz < nz; ++iz)
        pb_u[profile + uint(iz + 1)] = max(0.0f,
            ubmpi_sub(pb_u[profile + uint(iz)], ss_u[profile + uint(iz)]));
    error[item] = 0;
}
