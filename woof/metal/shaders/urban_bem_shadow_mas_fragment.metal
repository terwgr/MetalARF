#include <metal_stdlib>
#include "urban_bem_shadow_mas_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmsm_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmsm_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmsm_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmsm_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }

static inline float ubmsm_shade_wall(float z1, float z2, float hu,
                                     float elevation, float azimuth,
                                     float width) {
    const float sine = precise::sin(azimuth);
    const float tangent = precise::tan(elevation);
    const float x1 = min(ubmsm_mul(ubmsm_sub(hu, z1), tangent),
                         max(0.0f, ubmsm_div(width, sine)));
    const float x2 = max(ubmsm_mul(ubmsm_sub(hu, z2), tangent), 0.0f);
    return max(0.0f, ubmsm_div(
        ubmsm_mul(sine, max(0.0f, ubmsm_sub(x1, x2))),
        ubmsm_sub(z2, z1)));
}

// WRF bb_shadow_mas/bb_shade_wall (urban_bep_bem.cu:2272-2452). One thread
// owns one column, preserves direction/level/roof accumulation order, and
// emits direct-beam rsg[col,2] and rsw[col,4,18].
kernel void urban_bem_shadow_mas_fragment(
    constant UrbanBemShadowMasFragmentResources &r [[buffer(0)]],
    constant UrbanBemShadowMasFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    device const int *active_cols = r.active_cols;
    device const int *utype = r.utype;
    device const int *class_int = r.class_int;
    device const int *nzu_class = r.nzu_class;
    device const float *z_u = r.z_u;
    device const float *ws = r.ws;
    device const float *drst = r.drst;
    device const float *ss = r.ss;
    device const float *pb = r.pb;
    device const float *swddir = r.swddir;
    device const float *cosz = r.cosz;
    device const float *hour_angle = r.hour_angle;
    device const float *latitude = r.latitude;
    device float *rsw = r.rsw;
    device float *rsg = r.rsg;
    device int *error = r.error;
    const int ncol = p.ncol;
    const int nurbm = p.nurbm;
    if (col >= uint(ncol)) return;
    const ulong wall_base = ulong(col) * 72ul;
    const ulong ground_base = ulong(col) * 2ul;
    const ulong profile = ulong(col) * 19ul;
    for (int q = 0; q < 72; ++q) rsw[wall_base + uint(q)] = 0.0f;
    for (int dir = 0; dir < 2; ++dir) rsg[ground_base + uint(dir)] = 0.0f;
    error[col] = 0;
    if (active_cols[col] == 0) return;
    const int type = utype[col];
    if (nurbm < 1 || nurbm > 11 || type < 1 || type > nurbm) {
        error[col] = 5531;
        return;
    }
    const int nzu = nzu_class[uint(type - 1)];
    const int nd = class_int[uint(type)];
    if (nzu < 1 || nzu > 18 || nd < 1 || nd > 2) {
        error[col] = 5532;
        return;
    }
    const float direct = swddir[col];
    const float zr = precise::acos(cosz[col]);
    const float sun_sine = precise::sin(zr);
    if (direct == 0.0f || sun_sine == 1.0f) return;
    const float xlat_r = ubmsm_div(
        ubmsm_mul(latitude[col], 3.14159265358979323846f), 180.0f);
    float bbb = 0.0f;
    if (fabs(sun_sine) > 1.0e-10f) {
        const float ratio = ubmsm_div(
            ubmsm_mul(precise::cos(p.declin), precise::sin(hour_angle[col])),
            sun_sine);
        if (ratio >= 1.0f) bbb = 1.57079632679489661923f;
        else if (ratio <= -1.0f) bbb = -1.57079632679489661923f;
        else {
            bbb = precise::asin(ratio);
            if (precise::sin(p.declin) <
                ubmsm_mul(precise::cos(zr), precise::sin(xlat_r)))
                bbb = ubmsm_sub(3.14159265358979323846f, bbb);
        }
    } else {
        const float product = ubmsm_mul(
            precise::cos(p.declin), precise::sin(hour_angle[col]));
        if (product >= 0.0f) bbb = 1.57079632679489661923f;
        else if (product < 0.0f) bbb = -1.57079632679489661923f;
    }
    const float tangent = precise::tan(zr);
    for (int dir = 0; dir < nd; ++dir) {
        const ulong direction = ulong(col) * 2ul + uint(dir);
        const float width = ws[direction];
        const float aae = ubmsm_sub(bbb, drst[direction]);
        const float aaw = ubmsm_add(aae, 3.14159265358979323846f);
        const float sine_a = precise::sin(aae);
        const float sine_b = precise::sin(aaw);
        for (int iz = 0; iz < nzu; ++iz) {
            const int face_a = dir * 2;
            const int face_b = face_a + 1;
            const ulong out_a = wall_base + uint(face_a * 18 + iz);
            const ulong out_b = wall_base + uint(face_b * 18 + iz);
            if (pb[profile + uint(iz + 1)] > 0.0f) {
                for (int jz = 0; jz < nzu; ++jz) {
                    const float roof = z_u[uint(jz + 1)];
                    const float spacing = ss[profile + uint(jz + 1)];
                    if (fabs(sine_a) > 1.0e-10f) {
                        const float rd = ubmsm_shade_wall(
                            z_u[uint(iz)], z_u[uint(iz + 1)], roof, zr,
                            aae, width);
                        const float beam = ubmsm_mul(
                            ubmsm_mul(direct, rd), spacing);
                        rsw[out_a] = ubmsm_add(rsw[out_a],
                            ubmsm_div(beam, pb[profile + uint(iz + 1)]));
                    }
                    if (fabs(sine_b) > 1.0e-10f) {
                        const float rd = ubmsm_shade_wall(
                            z_u[uint(iz)], z_u[uint(iz + 1)], roof, zr,
                            aaw, width);
                        const float beam = ubmsm_mul(
                            ubmsm_mul(direct, rd), spacing);
                        rsw[out_b] = ubmsm_add(rsw[out_b],
                            ubmsm_div(beam, pb[profile + uint(iz + 1)]));
                    }
                }
            }
        }
        if (fabs(sine_a) > 1.0e-10f) {
            const float wsd = fabs(ubmsm_div(width, sine_a));
            float ground = 0.0f;
            for (int jz = 0; jz < nzu; ++jz) {
                const float roof = z_u[uint(jz + 1)];
                const float rd = max(0.0f, ubmsm_sub(
                    wsd, ubmsm_mul(roof, tangent)));
                const float beam = ubmsm_mul(
                    ubmsm_mul(direct, rd), ss[profile + uint(jz + 1)]);
                ground = ubmsm_add(ground, ubmsm_div(beam, wsd));
            }
            rsg[ground_base + uint(dir)] = ground;
        } else {
            rsg[ground_base + uint(dir)] = direct;
        }
    }
}
