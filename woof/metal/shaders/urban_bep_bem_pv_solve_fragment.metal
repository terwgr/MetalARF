#include <metal_stdlib>
#include "urban_bep_bem_pv_solve_fragment_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_pow.metalh"
using namespace metal;
#pragma clang fp contract(off)

constant uint BEMPV_NBUI = 15u;
constant uint BEMPV_NZUM = 18u;
constant uint BEMPV_NDM = 2u;
constant uint BEMPV_INTERP_FIELDS = 7u;
constant uint BEMPV_GR_FRAC = 1206u;
constant uint BEMPV_EMR = 121u;
constant uint BEMPV_BS = 220u;
constant uint BEMPV_PV_FRAC = 1217u;

static inline float ubpv_add(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubpv_sub(float a, float b) {
    return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubpv_mul(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubpv_div(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubpv_sqrt(float a) {
    return as_type<float>(soft32_sqrt(as_type<uint>(a)));
}
static inline float ubpv_pow(float a, float b) {
    return as_type<float>(soft_gfk_pow_bits(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubpv_abs(float a) {
    return as_type<float>(as_type<uint>(a) & 0x7fffffffu);
}
static inline float ubpv_min(float a, float b) { return a < b ? a : b; }
static inline float ubpv_max(float a, float b) { return a > b ? a : b; }

// WRF bem_hsfluxpv (urban_bem.cuh:1010-1084), followed by the PV contribution
// in bb_buildings. One item owns one column and walks its building slots in
// source order so it can update each panel and the shared level tendencies.
kernel void urban_bep_bem_pv_solve_fragment(
    constant UrbanBepBemPvSolveFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemPvSolveFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= uint(p.ncol)) return;
    const uint col = item;
    if (r.active_cols[col] == 0) return;

    const int utype = r.utype[col] - 1;
    if (utype < 0 || utype >= 11) return;
    const ulong c = ulong(col);
    const ulong class_row = ulong(utype);
    const float pv_fraction = r.class_float[BEMPV_PV_FRAC + class_row];
    const float green_fraction = r.class_float[BEMPV_GR_FRAC + class_row];

    const ulong interp_base = c * BEMPV_INTERP_FIELDS * BEMPV_NZUM;
    constexpr float albpv = 0.11f;
    constexpr float empv_down = 0.95f;
    constexpr float empv_up = 0.79f;
    constexpr float t_amb = 25.0f;
    constexpr float r1 = 2330.0f, r2 = 1200.0f, r3 = 3000.0f;
    constexpr float c1 = 677.0f, c2 = 1250.0f, c3 = 500.0f;
    constexpr float d1 = 0.0003f, d2 = 0.0005f, d3 = 0.003f;
    const float cm = ubpv_add(ubpv_add(
        ubpv_mul(ubpv_mul(r1, c1), d1),
        ubpv_mul(ubpv_mul(r2, c2), d2)),
        ubpv_mul(ubpv_mul(r3, c3), d3));
    const float emissivity = r.class_float[ BEMPV_EMR + class_row ];
    const float hrad = ubpv_div(5.67e-8f, ubpv_add(ubpv_add(
        ubpv_div(ubpv_sub(1.0f, empv_down), empv_down), 1.0f),
        ubpv_div(ubpv_sub(1.0f, emissivity), emissivity)));
    const ulong class_bs = BEMPV_BS + class_row * BEMPV_NDM;
    const float rs = r.swdown[col];
    const float rl = r.glw[col];
    for (uint building = 0; building < BEMPV_NBUI; ++building) {
        const ulong profile = c * BEMPV_NBUI + building;
        const uint out = col * BEMPV_NBUI + building;
        r.pv_energy[out] = 0.0f;
        r.pv_hspv[out] = 0.0f;
        const int level = r.building_nlev[profile];
        if (level <= 0 || level >= int(BEMPV_NZUM) || !(pv_fraction > 0.0f))
            continue;
        const uint z = uint(level);
        const float ua = r.interp[interp_base + z];
        const float va = r.interp[interp_base + BEMPV_NZUM + z];
        const float pt = r.interp[interp_base + 2ul * BEMPV_NZUM + z];
        const float pressure = r.interp[interp_base + 4ul * BEMPV_NZUM + z];
        const float tair = ubpv_mul(pt, ubpv_pow(
            ubpv_div(pressure, 100000.0f), 287.0f / 1004.0f));
        const float tr = r.tr_av[(c * BEMPV_NZUM + z) * BEMPV_NDM];
        const ulong tpv_index = (c * BEMPV_NZUM + z) * BEMPV_NDM;
        float tpv = r.tpv[tpv_index];
        const float uroof = ubpv_sqrt(ubpv_add(
            ubpv_mul(ua, ua), ubpv_mul(va, va)));
        const float deltat = ubpv_sub(tpv, tair);
        const float hf = ubpv_mul(2.5f, ubpv_pow(
            ubpv_mul(0.4f, uroof), 0.5f));
        float hc = ubpv_div(ubpv_mul(9.482f,
            ubpv_pow(ubpv_abs(deltat), 1.0f / 3.0f)), 6.238f);
        const float hup = ubpv_sqrt(ubpv_add(ubpv_mul(hc, hc), ubpv_mul(hf, hf)));
        hc = ubpv_div(ubpv_mul(1.810f,
            ubpv_pow(ubpv_abs(deltat), 1.0f / 3.0f)), 2.382f);
        const float hdown = ubpv_sqrt(ubpv_add(ubpv_mul(hc, hc), ubpv_mul(hf, hf)));
        const float enerpv = ubpv_mul(ubpv_mul(0.19f, rs), ubpv_min(
            1.0f, ubpv_sub(1.0f, ubpv_mul(0.005f,
                 ubpv_sub(tpv, ubpv_add(t_amb, 273.15f))))));
        const float sw_d = ubpv_mul(ubpv_sub(1.0f, albpv), rs);
        const float lw_d = ubpv_mul(empv_up, rl);
        const float lwpv_out = ubpv_mul(ubpv_mul(empv_up, 5.67e-8f),
            ubpv_pow(tpv, 4.0f));
        const float lwupr = ubpv_mul(hrad,
            ubpv_sub(ubpv_pow(tr, 4.0f), ubpv_pow(tpv, 4.0f)));
        const float hspv = ubpv_mul(ubpv_add(hup, hdown),
            ubpv_sub(tpv, tair));
        const float tendency = ubpv_div(ubpv_sub(ubpv_sub(ubpv_add(
            ubpv_sub(ubpv_add(sw_d, lw_d), lwpv_out), lwupr), hspv), enerpv), cm);
        tpv = ubpv_add(tpv, ubpv_mul(tendency, p.dt));
        r.tpv[tpv_index] = tpv;
        r.tpv[tpv_index + 1ul] = tpv;
        const float eppv = ubpv_mul(ubpv_mul(ubpv_mul(
            enerpv, pv_fraction), r.class_float[class_bs]),
            r.class_float[class_bs + 1ul]);
        r.pv_energy[out] = eppv;
        r.pv_hspv[out] = hspv;

        const float da = r.interp[interp_base + 5ul * BEMPV_NZUM + z];
        const float pv_tendency = ubpv_div(
            ubpv_mul(ubpv_mul(ubpv_sub(1.0f, green_fraction),
                              pv_fraction), hspv),
            ubpv_mul(da, p.cp_u));
        for (uint d = 0; d < BEMPV_NDM; ++d) {
            const ulong flat = (c * BEMPV_NDM + d) * BEMPV_NZUM + z;
            r.flat_thb[flat] = ubpv_add(r.flat_thb[flat], pv_tendency);
        }
    }
}
