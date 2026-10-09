#include <metal_stdlib>
#include "urban_bep_surface_temperature_fragment_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_pow.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubep_add(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubep_sub(float a, float b) {
    return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubep_mul(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubep_div(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubep_pow(float a, float b) {
    return as_type<float>(soft_gfk_pow_bits(as_type<uint>(a), as_type<uint>(b)));
}

// One WRF soil-temperature column. temp is updated in place and pt/rt/gf
// retain the outputs used by the following BEP buildings and flux stages.
static inline void ubep_soil_temperature(
        thread float *temp, thread const float *dz,
        thread const float *ala, thread const float *cs,
        float rs, float rl, float pressure, float dt, float emissivity,
        float albedo, float surface_flux,
        thread float &pt, thread float &rt, thread float &gf) {
    const int nz = 10;
    float a[30];
    float c[10];
    float cddz[12];
    for (int i = 0; i < 30; ++i) a[i] = 0.0f;
    for (int i = 0; i < 10; ++i) c[i] = 0.0f;
    for (int i = 0; i < 12; ++i) cddz[i] = 0.0f;
    const float tsig = temp[9];
    const float tsig2 = ubep_mul(tsig, tsig);
    const float tsig4 = ubep_mul(tsig2, tsig2);
    const float alpha = ubep_add(ubep_sub(ubep_add(
        ubep_mul(ubep_sub(1.0f, albedo), rs), ubep_mul(emissivity, rl)),
        ubep_mul(ubep_mul(emissivity, 5.67e-8f), tsig4)), surface_flux);
    cddz[0] = ubep_div(ala[0], dz[0]);
    for (int iz = 1; iz < nz; ++iz)
        cddz[uint(iz)] = ubep_div(ubep_mul(2.0f, ala[uint(iz)]),
            ubep_add(dz[uint(iz)], dz[uint(iz - 1)]));
    a[1] = 1.0f;
    c[0] = temp[0];
    for (int iz = 1; iz < nz - 1; ++iz) {
        const ulong at = ulong(iz) * 3ul;
        a[at] = ubep_div(ubep_mul(-cddz[uint(iz)], dt), dz[uint(iz)]);
        a[at + 1ul] = ubep_add(1.0f, ubep_div(
            ubep_mul(dt, ubep_add(cddz[uint(iz)], cddz[uint(iz + 1)])),
            dz[uint(iz)]));
        a[at + 2ul] = ubep_div(ubep_mul(-cddz[uint(iz + 1)], dt),
            dz[uint(iz)]);
        c[uint(iz)] = temp[uint(iz)];
    }
    const ulong bottom = ulong(nz - 1) * 3ul;
    a[bottom] = ubep_div(ubep_mul(-dt, cddz[9]), dz[9]);
    a[bottom + 1ul] = ubep_add(1.0f,
        ubep_div(ubep_mul(dt, cddz[9]), dz[9]));
    c[9] = ubep_add(temp[9], ubep_div(
        ubep_div(ubep_mul(dt, alpha), cs[9]), dz[9]));
    for (int i = nz - 2; i >= 0; --i) {
        const ulong at = ulong(i) * 3ul, next = ulong(i + 1) * 3ul;
        c[uint(i)] = ubep_sub(c[uint(i)], ubep_div(
            ubep_mul(a[at + 2ul], c[uint(i + 1)]), a[next + 1ul]));
        a[at + 1ul] = ubep_sub(a[at + 1ul], ubep_div(
            ubep_mul(a[at + 2ul], a[next]), a[next + 1ul]));
    }
    for (int i = 1; i < nz; ++i) {
        const ulong at = ulong(i) * 3ul, prev = ulong(i - 1) * 3ul;
        c[uint(i)] = ubep_sub(c[uint(i)], ubep_div(
            ubep_mul(a[at], c[uint(i - 1)]), a[prev + 1ul]));
    }
    for (int i = 0; i < nz; ++i)
        temp[uint(i)] = ubep_div(c[uint(i)], a[ulong(i) * 3ul + 1ul]);
    const float pressure_scale = ubep_pow(
        ubep_div(pressure, 100000.0f), -0.28585657f);
    const float emitted = ubep_mul(ubep_mul(emissivity, 5.67e-8f), tsig4);
    const float net = ubep_sub(ubep_add(
        ubep_mul(ubep_sub(1.0f, albedo), rs), ubep_mul(emissivity, rl)), emitted);
    pt = ubep_mul(temp[9], pressure_scale);
    rt = net;
    gf = ubep_add(net, surface_flux);
}

// One item owns one of the 108 padded ground/roof/wall soil columns in a BEP
// atmospheric column. State and parameter records remain in their canonical
// layouts; no host-side channel gather is needed.
kernel void urban_bep_surface_temperature_fragment(
    constant UrbanBepSurfaceTemperatureFragmentResources &r [[buffer(0)]],
    constant UrbanBepSurfaceTemperatureFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    const uint slots = 108u;
    const uint col = item / slots, slot = item % slots;
    if (col >= uint(p.ncol)) return;
    const ulong pix = ulong(col) * 3ul;
    const int nzu = r.column_param_ix[pix];
    const int nd = r.column_param_ix[pix + 1ul];
    const int type = r.utype_urb[col] - 1;
    const ulong param = ulong(col) * 140ul;
    const ulong x = ulong(col) * 330ul;
    const ulong state = ulong(col) * 1210ul;
    if (!r.active_cols[col] || type < 0 || type >= 11
            || r.column_param_ix[pix + 2ul] != 0 || nzu < 1 || nzu > 18
            || nd < 1 || nd > 2) {
        if (slot == 0u) r.error[col] = r.column_param_ix[pix + 2ul];
        return;
    }
    if (slot == 0u) r.error[col] = 0;

    uint kind, dir, side, level;
    if (slot < 2u) {
        kind = 0u; dir = slot; side = 0u; level = 0u;
    } else if (slot < 36u) {
        const uint roof = slot - 2u;
        kind = 1u; dir = roof / 17u; side = 0u; level = roof % 17u + 1u;
    } else {
        const uint wall = slot - 36u;
        kind = 2u; side = wall / 18u; dir = side / 2u; level = wall % 18u;
    }
    if (dir >= uint(nd) || (kind != 0u && level >= uint(nzu))) return;
    if (kind == 1u && level == 0u) return;

    float dz[10], ala[10], cs[10], temp[10];
    if (kind == 0u) {
        const float widths[10] = {0.2f,0.12f,0.08f,0.05f,0.03f,0.02f,0.02f,0.01f,0.005f,0.0025f};
        for (int k = 0; k < 10; ++k) {
            dz[k] = widths[k];
            ala[k] = r.class_x[ulong(type)];
            cs[k] = r.class_x[33ul + ulong(type)];
            temp[k] = r.column_state[state + 720ul + ulong(dir) + 2ul * ulong(k)];
        }
        float pt, rt, gf;
        ubep_soil_temperature(temp, dz, ala, cs,
            r.rsg[ulong(col) * 2ul + ulong(dir)],
            r.rlg[ulong(col) * 2ul + ulong(dir)],
            r.column_interp_x[ulong(col) * 108ul + 4ul * 18ul], p.dt,
            r.class_x[132ul + ulong(type)], r.class_x[99ul + ulong(type)],
            r.column_state[state + 1172ul + ulong(dir)], pt, rt, gf);
        for (int k = 0; k < 10; ++k)
            r.column_state[state + 720ul + ulong(dir) + 2ul * ulong(k)] = temp[k];
        r.surface_x[x + ulong(dir)] = pt;
        r.surface_x[x + 110ul + ulong(dir)] = rt;
        r.surface_x[x + 220ul + ulong(dir)] = gf;
    } else if (kind == 1u) {
        const float widths[10] = {0.02f,0.02f,0.02f,0.02f,0.02f,0.02f,0.02f,0.01f,0.005f,0.0025f};
        for (int k = 0; k < 10; ++k) {
            dz[k] = widths[k];
            ala[k] = r.class_x[22ul + ulong(type)];
            cs[k] = r.class_x[55ul + ulong(type)];
            temp[k] = r.column_state[state + 740ul + ulong(dir) + 2ul * ulong(level) + 36ul * ulong(k)];
        }
        const ulong interp = ulong(col) * 108ul + 4ul * 18ul + ulong(level);
        const ulong param_ss = param + 104ul + ulong(level);
        if (r.column_param[param_ss] <= 0.0f) return;
        float pt, rt, gf;
        ubep_soil_temperature(temp, dz, ala, cs, r.swdown[col], r.glw[col],
            r.column_interp_x[interp], p.dt,
            r.class_x[154ul + ulong(type)], r.class_x[121ul + ulong(type)],
            r.column_state[state + 1174ul + ulong(dir) + 2ul * ulong(level)],
            pt, rt, gf);
        for (int k = 0; k < 10; ++k)
            r.column_state[state + 740ul + ulong(dir) + 2ul * ulong(level) + 36ul * ulong(k)] = temp[k];
        r.surface_x[x + 2ul + ulong(dir) * 18ul + ulong(level)] = pt;
        r.surface_x[x + 112ul + ulong(dir) * 18ul + ulong(level)] = rt;
        r.surface_x[x + 222ul + ulong(dir) * 18ul + ulong(level)] = gf;
    } else {
        const float widths[10] = {0.02f,0.02f,0.02f,0.02f,0.02f,0.02f,0.02f,0.01f,0.005f,0.0025f};
        for (int k = 0; k < 10; ++k) {
            dz[k] = widths[k];
            ala[k] = r.class_x[11ul + ulong(type)];
            cs[k] = r.class_x[44ul + ulong(type)];
            temp[k] = r.column_state[state + ulong(side) + 4ul * ulong(level) + 72ul * ulong(k)];
        }
        const ulong interp = ulong(col) * 108ul + 4ul * 18ul + ulong(level);
        float pt, rt, gf;
        ubep_soil_temperature(temp, dz, ala, cs,
            r.rsw[ulong(col) * 72ul + ulong(side) * 18ul + ulong(level)],
            r.rlw[(ulong(col) * 2ul + ulong(dir)) * 36ul + ulong(side % 2u) * 18ul + ulong(level)],
            r.column_interp_x[interp], p.dt,
            r.class_x[143ul + ulong(type)], r.class_x[110ul + ulong(type)],
            r.column_state[state + 1100ul + ulong(side) + 4ul * ulong(level)], pt, rt, gf);
        for (int k = 0; k < 10; ++k)
            r.column_state[state + ulong(side) + 4ul * ulong(level) + 72ul * ulong(k)] = temp[k];
        r.surface_x[x + 38ul + ulong(side) * 18ul + ulong(level)] = pt;
        r.surface_x[x + 148ul + ulong(side) * 18ul + ulong(level)] = rt;
        r.surface_x[x + 258ul + ulong(side) * 18ul + ulong(level)] = gf;
    }
}
