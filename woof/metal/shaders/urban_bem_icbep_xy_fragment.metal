#include <metal_stdlib>
#include "urban_bem_icbep_xy_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubix_add(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubix_sub(float a, float b) {
    return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubix_mul(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubix_div(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubix_sqrt(float x) { return precise::sqrt(x); }
static inline float ubix_log(float x) { return precise::log(x); }
static inline float ubix_atan(float x) { return precise::atan(x); }

// The CUDA bb_icbep_xy wrapper clears all factor arrays and calls
// bb_view_factors once per direction through ndu. One thread handles a
// (domain column, direction); fww is stored with source's iz-fast order in
// the [column, direction, jz, iz] scratch layout. Its log/power/atan/sqrt
// intrinsics use Metal precise math here; soft32 keeps every surrounding
// FP32 operation source ordered and non-contracted.
static inline float ubix_fprls(float a, float b, float c) {
    const float x = ubix_div(a, c);
    const float y = ubix_div(b, c);
    if (a == 0.0f || b == 0.0f) return 0.0f;
    const float x2 = ubix_mul(x, x);
    const float y2 = ubix_mul(y, y);
    const float numerator = ubix_mul(ubix_add(1.0f, x2),
                                     ubix_add(1.0f, y2));
    const float denominator = ubix_add(ubix_add(1.0f, x2), y2);
    const float t1 = ubix_log(ubix_sqrt(ubix_div(numerator, denominator)));
    const float sx = ubix_sqrt(ubix_add(1.0f, x2));
    const float sy = ubix_sqrt(ubix_add(1.0f, y2));
    const float t2 = ubix_mul(ubix_mul(y, sx), ubix_atan(ubix_div(y, sx)));
    const float t3 = ubix_mul(ubix_mul(x, sy), ubix_atan(ubix_div(x, sy)));
    const float v = ubix_sub(
        ubix_sub(ubix_add(ubix_add(t1, t2), t3), ubix_mul(y, ubix_atan(y))),
        ubix_mul(x, ubix_atan(x)));
    return ubix_div(ubix_mul(v, 2.0f),
                    ubix_mul(ubix_mul(3.141592653f, x), y));
}

static inline float ubix_fnrm(float a, float b, float c) {
    const float x = ubix_div(a, b);
    const float y = ubix_div(c, b);
    const float x2 = ubix_mul(x, x);
    const float y2 = ubix_mul(y, y);
    const float z = ubix_add(x2, y2);
    if (y == 0.0f || x == 0.0f) return 0.0f;
    const float a1 = ubix_log(ubix_div(
        ubix_mul(ubix_add(1.0f, x2), ubix_add(1.0f, y2)),
        ubix_add(1.0f, z)));
    const float a2 = ubix_mul(y2, ubix_log(ubix_div(
        ubix_div(ubix_mul(y2, ubix_add(1.0f, z)), z),
        ubix_add(1.0f, y2))));
    const float a3 = ubix_mul(x2, ubix_log(ubix_div(
        ubix_div(ubix_mul(x2, ubix_add(1.0f, z)), z),
        ubix_add(1.0f, x2))));
    const float a4 = ubix_mul(y, ubix_atan(ubix_div(1.0f, y)));
    const float a5 = ubix_mul(x, ubix_atan(ubix_div(1.0f, x)));
    const float rootz = ubix_sqrt(z);
    const float a6 = ubix_mul(rootz, ubix_atan(ubix_div(1.0f, rootz)));
    const float v = ubix_sub(
        ubix_add(ubix_add(ubix_mul(0.25f,
            ubix_add(ubix_add(a1, a2), a3)), a4), a5), a6);
    return ubix_div(v, ubix_mul(3.141592653f, y));
}

kernel void urban_bem_icbep_xy_fragment(
    constant UrbanBemIcbepXyFragmentResources &r [[buffer(0)]],
    constant UrbanBemIcbepXyFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const int *active = r.active_cols;
    device const int *utype = r.utype;
    device const int *class_int = r.class_int;
    device const int *nzurban = r.nzurban;
    device const float *z = r.z_u;
    device const float *strd = r.strd;
    device const float *ws = r.ws;
    device float *fww = r.fww;
    device float *fwg = r.fwg;
    device float *fgw = r.fgw;
    device float *fsw = r.fsw;
    device float *fws = r.fws;
    device float *fsg = r.fsg;
    device int *error = r.error;

    const uint ncol = uint(p.ncol);
    const uint direction_count = 2u;
    if (item >= ncol * direction_count) return;
    error[item] = 0;
    const uint col = item / direction_count;
    const uint direction = item % direction_count;
    if (active[col] == 0) return;
    if (p.ncol <= 0 || p.nurbm < 1 || p.nurbm > 11) {
        error[item] = 6501;
        return;
    }
    const int row = utype[col] - 1;
    if (row < 0 || row >= p.nurbm) {
        error[item] = 6502;
        return;
    }
    const int ndu = class_int[1u + uint(row)];
    const int nz_u = nzurban[col];
    if (ndu < 1 || ndu > 2 || nz_u < 0 || nz_u > 18) {
        error[item] = 6503;
        return;
    }
    if (direction >= uint(ndu)) return;

    const ulong dbase = ulong(col) * 2ul + direction;
    const float dxy = strd[dbase];
    const float width = ws[dbase];
    const float hut = z[uint(nz_u)];
    if (!(dxy > 0.0f) || !(width > 0.0f)) {
        error[item] = 6504;
        return;
    }
    const ulong vector_base = ulong(item) * 18ul;
    const ulong matrix_base = ulong(item) * 324ul;

    for (int jz = 1; jz <= nz_u; ++jz) {
        for (int iz = 1; iz <= nz_u; ++iz) {
            const float f123 = ubix_fprls(
                dxy, fabs(ubix_sub(z[uint(jz)], z[uint(iz - 1)])), width);
            const float f23 = ubix_fprls(
                dxy, fabs(ubix_sub(z[uint(jz)], z[uint(iz)])), width);
            const float f12 = ubix_fprls(
                dxy, fabs(ubix_sub(z[uint(jz - 1)], z[uint(iz - 1)])), width);
            const float f2 = ubix_fprls(
                dxy, fabs(ubix_sub(z[uint(jz - 1)], z[uint(iz)])), width);
            const float a123 = ubix_mul(
                dxy, fabs(ubix_sub(z[uint(jz)], z[uint(iz - 1)])));
            const float a12 = ubix_mul(
                dxy, fabs(ubix_sub(z[uint(jz - 1)], z[uint(iz - 1)])));
            const float a23 = ubix_mul(
                dxy, fabs(ubix_sub(z[uint(jz)], z[uint(iz)])));
            const float a1 = ubix_mul(
                dxy, fabs(ubix_sub(z[uint(iz)], z[uint(iz - 1)])));
            const float a2 = ubix_mul(
                dxy, fabs(ubix_sub(z[uint(jz - 1)], z[uint(iz)])));
            const float a3 = ubix_mul(
                dxy, fabs(ubix_sub(z[uint(jz)], z[uint(jz - 1)])));
            float numerator = ubix_sub(ubix_mul(a123, f123),
                                       ubix_mul(a23, f23));
            numerator = ubix_sub(numerator, ubix_mul(a12, f12));
            numerator = ubix_add(numerator, ubix_mul(a2, f2));
            const float ftot = ubix_div(ubix_mul(0.5f, numerator), a1);
            const ulong out = matrix_base + uint(jz - 1) * 18ul + uint(iz - 1);
            fww[out] = ubix_div(ubix_mul(ftot, a1), a3);
        }

        const float f12 = ubix_fnrm(z[uint(jz)], dxy, width);
        const float f1 = ubix_fnrm(z[uint(jz - 1)], dxy, width);
        const float a1 = ubix_mul(width, dxy);
        const float a12 = ubix_mul(width, dxy);
        const float a4 = ubix_mul(ubix_sub(z[uint(jz)], z[uint(jz - 1)]), dxy);
        const float ftot = ubix_div(ubix_sub(ubix_mul(a12, f12),
                                             ubix_mul(a12, f1)), a1);
        fgw[vector_base + uint(jz - 1)] = ubix_div(ubix_mul(ftot, a1), a4);

        const float f12s = ubix_fnrm(ubix_sub(hut, z[uint(jz - 1)]), dxy, width);
        const float f1s = ubix_fnrm(ubix_sub(hut, z[uint(jz)]), dxy, width);
        const float a1s = ubix_mul(width, dxy);
        const float a12s = ubix_mul(width, dxy);
        const float a4s = ubix_mul(ubix_sub(z[uint(jz)], z[uint(jz - 1)]), dxy);
        const float fts = ubix_div(ubix_sub(ubix_mul(a12s, f12s),
                                            ubix_mul(a12s, f1s)), a1s);
        fsw[vector_base + uint(jz - 1)] = ubix_div(ubix_mul(fts, a1s), a4s);
    }

    for (int iz = 1; iz <= nz_u; ++iz) {
        const float f12 = ubix_fnrm(width, dxy, z[uint(iz)]);
        const float f1 = ubix_fnrm(width, dxy, z[uint(iz - 1)]);
        const float a1 = ubix_mul(ubix_sub(z[uint(iz)], z[uint(iz - 1)]), dxy);
        const float a2 = ubix_mul(z[uint(iz - 1)], dxy);
        const float a12 = ubix_mul(z[uint(iz)], dxy);
        const float a4 = ubix_mul(width, dxy);
        const float ftot = ubix_div(ubix_sub(ubix_mul(a12, f12),
                                             ubix_mul(a2, f1)), a1);
        fwg[vector_base + uint(iz - 1)] = ubix_div(ubix_mul(ftot, a1), a4);

        const float f12w = ubix_fnrm(width, dxy,
                                     ubix_sub(hut, z[uint(iz - 1)]));
        const float f1w = ubix_fnrm(width, dxy,
                                    ubix_sub(hut, z[uint(iz)]));
        const float a1w = ubix_mul(ubix_sub(z[uint(iz)], z[uint(iz - 1)]), dxy);
        const float a2w = ubix_mul(ubix_sub(hut, z[uint(iz)]), dxy);
        const float a12w = ubix_mul(ubix_sub(hut, z[uint(iz - 1)]), dxy);
        const float a4w = ubix_mul(width, dxy);
        const float ftotw = ubix_div(ubix_sub(ubix_mul(a12w, f12w),
                                              ubix_mul(a2w, f1w)), a1w);
        fws[vector_base + uint(iz - 1)] = ubix_div(ubix_mul(ftotw, a1w), a4w);
    }

    fsg[item] = ubix_fprls(dxy, width, hut);
    error[item] = 0;
}
