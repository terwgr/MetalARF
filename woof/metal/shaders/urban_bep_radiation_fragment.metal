#include <metal_stdlib>
#include "urban_bep_radiation_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubr_add(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubr_sub(float a, float b) {
    return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubr_mul(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubr_div(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}

static inline float ubr_shade_wall(float z1, float z2, float hu,
                                   float elevation, float azimuth,
                                   float width) {
    const float sine = precise::sin(azimuth);
    const float tangent = precise::tan(elevation);
    const float x1 = min(ubr_mul(ubr_sub(hu, z1), tangent),
                         max(0.0f, ubr_div(width, sine)));
    const float x2 = max(ubr_mul(ubr_sub(hu, z2), tangent), 0.0f);
    return max(0.0f, ubr_div(ubr_mul(sine, max(0.0f, ubr_sub(x1, x2))),
                            ubr_sub(z2, z1)));
}

// WRF module_sf_bep.F shadow_mas/shade_wall (1815-1890, 1896-1946).
// Calculates direct solar input to ground and both walls in source order.
// The following radiosity solve consumes rsg/rsw.
kernel void urban_bep_radiation_fragment(
    constant UrbanBepRadiationFragmentResources &r [[buffer(0)]],
    constant UrbanBepRadiationFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong param = ulong(col) * 140ul;
    const ulong pix = ulong(col) * 3ul;
    const int nz = r.column_param_ix[pix];
    const int nd = r.column_param_ix[pix + 1ul];
    if (r.column_param_ix[pix + 2ul] != 0) {
        r.error[col] = r.column_param_ix[pix + 2ul];
        return;
    }
    if (r.active_cols[col] == 0) {
        for (int k = 0; k < 72; ++k) r.rsw[ulong(col) * 72ul + uint(k)] = 0.0f;
        r.rsg[ulong(col) * 2ul] = 0.0f;
        r.rsg[ulong(col) * 2ul + 1ul] = 0.0f;
        r.error[col] = 0;
        return;
    }
    const ulong slots = ulong(nz + 1);
    const ulong xin = ulong(col) * 15ul * slots;
    const ulong cold = ulong(col) * 2ul;
    const ulong colw = ulong(col) * 72ul;
    const float rs = r.swdown[col];
    const float zr = precise::acos(r.cosz[col]);
    const float sun = precise::sin(zr);
    for (int id = 0; id < 2; ++id) {
        r.rsg[cold + uint(id)] = 0.0f;
        for (int iz = 0; iz < 18; ++iz) {
            r.rsw[colw + uint(id * 36 + iz)] = 0.0f;
            r.rsw[colw + uint((id * 2 + 1) * 18 + iz)] = 0.0f;
        }
    }
    if (nz < 1 || nz > 18 || nd < 1 || nd > 2) {
        r.error[col] = 1946;
        return;
    }
    if (rs == 0.0f || sun == 1.0f) {
        r.error[col] = 0;
        return;
    }

    const float cosdecl = precise::cos(p.declin);
    const float sinhour = precise::sin(r.hour_angle[col]);
    const float product = ubr_mul(cosdecl, sinhour);
    float ratio = 0.0f;
    if (fabs(sun) > 1.0e-10f) ratio = ubr_div(product, sun);
    const float bbb = fabs(sun) > 1.0e-10f
        ? precise::asin(clamp(ratio, -1.0f, 1.0f))
        : copysign(1.5707963267948966f, product);
    const float tangent = precise::tan(zr);
    for (int id = 0; id < nd; ++id) {
        const ulong direction = cold + uint(id);
        const float width = r.column_param[param + 98ul + uint(id)];
        const float street_direction = r.column_param[param + 100ul + uint(id)];
        const float angle_a = ubr_sub(bbb, street_direction);
        const float angle_b = ubr_add(angle_a, 3.141592653589793f);
        for (int iz = 0; iz < nz; ++iz) {
            const float denom = r.column_param[param + 122ul + uint(iz + 1)];
            if (denom <= 0.0f) continue;
            const float zlo = r.column_x[xin + uint(iz)];
            const float zhi = r.column_x[xin + uint(iz + 1)];
            for (int jz = 0; jz < nz; ++jz) {
                const float roof = r.column_x[xin + uint(jz + 1)];
                const float ss = r.column_param[param + 104ul + uint(jz + 1)];
                if (fabs(precise::sin(angle_a)) > 1.0e-10f) {
                    const float rd = ubr_shade_wall(zlo, zhi, roof, zr,
                                                    angle_a, width);
                    const ulong at = colw + uint(id * 36 + iz);
                    r.rsw[at] = ubr_add(r.rsw[at],
                        ubr_div(ubr_mul(ubr_mul(rs, rd), ss), denom));
                }
                if (fabs(precise::sin(angle_b)) > 1.0e-10f) {
                    const float rd = ubr_shade_wall(zlo, zhi, roof, zr,
                                                    angle_b, width);
                    const ulong at = colw + uint((id * 2 + 1) * 18 + iz);
                    r.rsw[at] = ubr_add(r.rsw[at],
                        ubr_div(ubr_mul(ubr_mul(rs, rd), ss), denom));
                }
            }
        }
        const float sine_a = precise::sin(angle_a);
        if (fabs(sine_a) > 1.0e-10f) {
            const float wsd = fabs(ubr_div(width, sine_a));
            float ground = 0.0f;
            for (int jz = 0; jz < nz; ++jz) {
                const float roof = r.column_x[xin + uint(jz + 1)];
                const float rd = max(0.0f, ubr_sub(wsd, ubr_mul(roof, tangent)));
                ground = ubr_add(ground,
                    ubr_div(ubr_mul(ubr_mul(rs, rd),
                                    r.column_param[param + 104ul + uint(jz + 1)]), wsd));
            }
            r.rsg[direction] = ground;
        } else {
            r.rsg[direction] = rs;
        }
    }
    r.error[col] = 0;
}
