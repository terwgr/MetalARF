#include <metal_stdlib>
#include "urban_bep_bem_wall_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubm_wall_add(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubm_wall_sub(float a, float b) {
    return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubm_wall_mul(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubm_wall_div(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}

// WRF module_sf_bem.F bem_wall/bem_wall_gr/wall_coeff/tridia
// (984-1032, 1039-1150, 1156-1205, 2396-2409). A thread advances one
// material profile and keeps the SAVE coefficients in resident k1/k2/kc.
kernel void urban_bep_bem_wall_fragment(
    constant UrbanBepBemWallFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemWallFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= uint(p.nitem)) return;
    const int nz = p.nz;
    if (nz < 2 || nz > 20) {
        r.error[item] = r.green_roof[item] != 0 ? 1 : 2;
        return;
    }
    const ulong base = ulong(item) * 20ul;
    if (r.swwall[item] == 1) {
        float kf[20];
        for (int iz = 0; iz < nz; ++iz) {
            r.kc[base + uint(iz)] = ubm_wall_div(p.dt,
                ubm_wall_mul(r.dz[base + uint(iz)], r.cs[base + uint(iz)]));
            if (iz == 0) {
                kf[0] = ubm_wall_div(ubm_wall_mul(2.0f, r.k[base]),
                                     r.dz[base]);
            } else {
                kf[uint(iz)] = ubm_wall_div(ubm_wall_mul(2.0f,
                    r.k[base + uint(iz)]),
                    ubm_wall_add(r.dz[base + uint(iz)], r.dz[base + uint(iz - 1)]));
            }
        }
        for (int iz = 0; iz < nz; ++iz)
            r.k1[base + uint(iz)] = ubm_wall_mul(r.kc[base + uint(iz)], kf[uint(iz)]);
        for (int iz = 0; iz < nz - 1; ++iz)
            r.k2[base + uint(iz)] = ubm_wall_div(ubm_wall_mul(
                ubm_wall_mul(r.kc[base + uint(iz)], kf[uint(iz + 1)]),
                r.cs[base + uint(iz)]), r.cs[base + uint(iz + 1)]);
        r.k2[base + uint(nz - 1)] = 0.0f;
        r.swwall[item] = 0;
    }

    float a[60];
    float b[20];
    for (int j = 0; j < 60; ++j) a[j] = 0.0f;
    for (int j = 0; j < nz; ++j) b[uint(j)] = r.temp[base + uint(j)];
    const bool green = r.green_roof[item] != 0;
    const float frac = r.green_fraction[item];
    // Three packed diagonals are ordered lower, main, upper.
    a[1] = ubm_wall_add(1.0f, r.k2[base]);
    a[2] = -r.k2[base];
    b[0] = ubm_wall_add(b[0], ubm_wall_mul(r.flux[ulong(item) * 2ul],
                                           r.kc[base]));
    for (int iz = 1; iz < nz - 1; ++iz) {
        const ulong at = ulong(iz) * 3ul;
        const float k1 = r.k1[base + uint(iz)];
        const float k2 = r.k2[base + uint(iz)];
        a[at] = -k1;
        if (green && iz == 4) {
            a[at + 1ul] = ubm_wall_add(ubm_wall_add(1.0f, k1),
                ubm_wall_mul(ubm_wall_sub(1.0f, frac), k2));
            b[uint(iz)] = ubm_wall_add(b[uint(iz)], ubm_wall_div(
                ubm_wall_mul(ubm_wall_mul(frac, r.hfgr[item]), p.dt),
                r.dz[base + uint(iz)]));
            a[at + 2ul] = -ubm_wall_mul(k2, ubm_wall_sub(1.0f, frac));
        } else {
            a[at + 1ul] = ubm_wall_add(ubm_wall_add(1.0f, k1), k2);
            a[at + 2ul] = -k2;
        }
    }
    const ulong bottom = ulong(nz - 1) * 3ul;
    a[bottom] = -r.k1[base + uint(nz - 1)];
    a[bottom + 1ul] = ubm_wall_add(1.0f, r.k1[base + uint(nz - 1)]);
    b[uint(nz - 1)] = ubm_wall_add(b[uint(nz - 1)],
        ubm_wall_mul(r.flux[ulong(item) * 2ul + 1ul],
                     r.kc[base + uint(nz - 1)]));

    // Source tridia elimination order, using one-based source indices.
    for (int i = nz - 2; i >= 0; --i) {
        const ulong at = ulong(i) * 3ul;
        const ulong next = ulong(i + 1) * 3ul;
        b[uint(i)] = ubm_wall_sub(b[uint(i)], ubm_wall_div(
            ubm_wall_mul(a[at + 2ul], b[uint(i + 1)]), a[next + 1ul]));
        a[at + 1ul] = ubm_wall_sub(a[at + 1ul], ubm_wall_div(
            ubm_wall_mul(a[at + 2ul], a[next]), a[next + 1ul]));
    }
    for (int i = 1; i < nz; ++i) {
        const ulong at = ulong(i) * 3ul;
        const ulong prev = ulong(i - 1) * 3ul;
        b[uint(i)] = ubm_wall_sub(b[uint(i)], ubm_wall_div(
            ubm_wall_mul(a[at], b[uint(i - 1)]), a[prev + 1ul]));
    }
    for (int i = 0; i < nz; ++i)
        r.temp[base + uint(i)] = ubm_wall_div(b[uint(i)],
                                                   a[ulong(i) * 3ul + 1ul]);
    r.error[item] = 0;
}
