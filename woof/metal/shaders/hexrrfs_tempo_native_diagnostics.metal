#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "hexrrfs_tempo_native_radar_bindings.metalh"
#include "hexrrfs_tempo_native_prate_bindings.metalh"
using namespace ntc;
#pragma clang fp contract(off)
inline F get(device const float *p, ulong i) { return f32_word(as_type<uint>(p[i])); }
inline void put(device float *p, ulong i, F x) { p[i] = as_type<float>(x.bits); }

// The two source polynomials have identical structure but distinct coefficients.
inline F saturation(F p, F t, bool ice) {
    F x = tc_max(F(-80.f), tc_sub(t, F(273.16f)));
    F c0 = F(ice ? .609868993E03f : .611583699E03f), c1 = F(ice ? .499320233E02f : .444606896E02f);
    F c2 = F(ice ? .184672631E01f : .143177157e01f), c3 = F(ice ? .402737184E-1f : .264224321e-1f);
    F c4 = F(ice ? .565392987E-3f : .299291081e-3f), c5 = F(ice ? .521693933E-5f : .203154182e-5f);
    F c6 = F(ice ? .307839583E-7f : .702620698e-8f), c7 = F(ice ? .105785160E-9f : .379534310e-11f);
    F c8 = F(ice ? .161444444E-12f : -.321582393e-13f);
    F e = tc_add(c7, tc_mul(x, c8));
    e = tc_add(c6, tc_mul(x, e));
    e = tc_add(c5, tc_mul(x, e));
    e = tc_add(c4, tc_mul(x, e));
    e = tc_add(c3, tc_mul(x, e));
    e = tc_add(c2, tc_mul(x, e));
    e = tc_add(c1, tc_mul(x, e));
    e = tc_add(c0, tc_mul(x, e));
    e = tc_min(e, tc_mul(p, F(.15f)));
    return tc_div(tc_mul(F(.622f), e), tc_sub(p, e));
}
// status: 0 valid/skipped, 1 lacks kp+1, 2 selected spacing is zero or nonfinite.
// Invalid full diagnostics leave every profile and radar output untouched.
kernel void hexrrfs_tempo_native_radar(constant HexrrfsTempoNativeRadarResources &r [[buffer(0)]],
                                       constant HexrrfsTempoNativeRadarParams &p [[buffer(1)]],
                                       uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.solve))
        return;
    r.status[c] = 0;
    bool hourly = p.has_hourly_word ? r.hourly_word[0] == 1 : p.hourly != 0;
    if (p.full) {
        if (p.nz < 2) {
            r.status[c] = 1;
            return;
        }
        ulong n = ulong(uint(p.nz)) * uint(p.nc);
        int kp = 0;
        F zbase = get(r.prepared, 19 * n + c);
        for (int k = 0; k < p.nz; ++k) {
            ulong x = ulong(k) * uint(p.nc) + c;
            F zp = tc_add(tc_sub(get(r.prepared, 19 * n + x), zbase),
                          tc_mul(F(.5f), get(r.prepared, 3 * n + x)));
            if (zp < F(1000.f))
                kp = k;
        }
        if (kp + 1 >= p.nz) {
            r.status[c] = 1;
            return;
        }
        ulong x1 = ulong(kp) * uint(p.nc) + c, x2 = ulong(kp + 1) * uint(p.nc) + c;
        F z1 = tc_add(tc_sub(get(r.prepared, 19 * n + x1), zbase),
                      tc_mul(F(.5f), get(r.prepared, 3 * n + x1)));
        F z2 = tc_add(tc_sub(get(r.prepared, 19 * n + x2), zbase),
                      tc_mul(F(.5f), get(r.prepared, 3 * n + x2)));
        F spacing = tc_sub(z2, z1);
        uint magnitude = spacing.bits & 0x7fffffffu;
        if (magnitude == 0u || magnitude >= 0x7f800000u) {
            r.status[c] = 2;
            return;
        }
        if (p.validate_only)
            return;
        F maxz = F(-35.f);
        for (int k = 0; k < p.nz; ++k) {
            ulong x = ulong(k) * uint(p.nc) + c;
            maxz = tc_max(maxz, tc_max(F(-35.f), get(r.refl10cm, x)));
            F temp = tc_mul(get(r.prepared, x), get(r.prepared, 17 * n + x));
            F qs = saturation(get(r.prepared, n + x), temp, tc_sub(temp, F(273.16f)) <= F(0.f));
            put(r.relhum, x, tc_mul(tc_div(get(r.prepared, 4 * n + x), qs), F(100.f)));
        }
        F w1 = tc_div(tc_sub(z2, F(1000.f)), spacing), w2 = tc_sub(F(1.f), w1);
        put(r.refl10cm_max, c, maxz);
        put(r.refl10cm_1km, c,
            tc_add(tc_mul(w1, tc_max(F(-35.f), get(r.refl10cm, x1))),
                   tc_mul(w2, tc_max(F(-35.f), get(r.refl10cm, x2)))));
    }
    if (hourly)
        put(r.refl10cm_1km_max, c, tc_max(get(r.refl10cm_1km_max, c), get(r.refl10cm_1km, c)));
}
kernel void hexrrfs_tempo_native_prate(constant HexrrfsTempoNativePrateResources &r [[buffer(0)]],
                                       constant HexrrfsTempoNativePrateParams &p [[buffer(1)]],
                                       uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.solve))
        return;
    put(r.peak, c, tc_max(tc_div(get(r.step, c), F(p.dt)), get(r.peak, c)));
    if (!p.has_rolling)
        return;
    for (int k = p.longest - 1; k > 0; --k)
        r.rolling[ulong(k) * uint(p.nc) + c] = r.rolling[ulong(k - 1) * uint(p.nc) + c];
    r.rolling[c] = r.total[c];
    put(r.ave1, c,
        tc_max(tc_div(tc_sub(get(r.rolling, c), get(r.rolling, ulong(p.n1 - 1) * uint(p.nc) + c)),
                      F(p.time1)),
               get(r.ave1, c)));
    put(r.ave5, c,
        tc_max(tc_div(tc_sub(get(r.rolling, c), get(r.rolling, ulong(p.n5 - 1) * uint(p.nc) + c)),
                      F(p.time5)),
               get(r.ave5, c)));
    put(r.ave10, c,
        tc_max(tc_div(tc_sub(get(r.rolling, c), get(r.rolling, ulong(p.n10 - 1) * uint(p.nc) + c)),
                      F(p.time10)),
               get(r.ave10, c)));
}
