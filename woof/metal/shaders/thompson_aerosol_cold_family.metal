#include <metal_stdlib>
#include "thompson_aerosol_cold_family_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_gfk_atan.metalh"
#include "soft_libm64.metalh"

using namespace metal;
#pragma clang fp contract(off)

// Shared arithmetic adapters for the source translation. All binary64 values
// remain IEEE words; no native Metal double arithmetic is used. The generic
// soft-libm routines are present in the tree but remain unreviewed for this
// production kernel, so these adapters do not by themselves admit dispatch.

inline uint cold_f32(float x) { return as_type<uint>(x); }
inline float cold_f32_value(uint x) { return as_type<float>(x); }
inline float cold_add(float a, float b) {
    return cold_f32_value(soft32_add(cold_f32(a), cold_f32(b)));
}
inline float cold_sub(float a, float b) {
    return cold_f32_value(soft32_sub(cold_f32(a), cold_f32(b)));
}
inline float cold_mul(float a, float b) {
    return cold_f32_value(soft32_mul(cold_f32(a), cold_f32(b)));
}
inline float cold_div(float a, float b) {
    return cold_f32_value(soft32_div(cold_f32(a), cold_f32(b)));
}
inline float cold_fmax(float a, float b) {
    const uint ab = cold_f32(a), bb = cold_f32(b);
    if (soft32_is_nan(ab)) return b;
    if (soft32_is_nan(bb)) return a;
    if (soft32_is_zero(ab) && soft32_is_zero(bb))
        return (ab & 0x80000000u) != 0u ? b : a;
    return soft32_greater(ab, bb) ? a : b;
}
inline float cold_fmin(float a, float b) {
    const uint ab = cold_f32(a), bb = cold_f32(b);
    if (soft32_is_nan(ab)) return b;
    if (soft32_is_nan(bb)) return a;
    if (soft32_is_zero(ab) && soft32_is_zero(bb))
        return (ab & 0x80000000u) != 0u ? a : b;
    return soft32_less(ab, bb) ? a : b;
}
inline float cold_powf(float a, float b) {
    return cold_f32_value(soft_gfk_pow_bits(cold_f32(a), cold_f32(b)));
}
inline float cold_logf(float a) {
    return cold_f32_value(soft_gfk_log(cold_f32(a)));
}
inline float cold_expf(float a) {
    return cold_f32_value(soft_gfk_exp(cold_f32(a)));
}
inline float cold_atanf(float a) {
    return cold_f32_value(soft_gfk_atanf(cold_f32(a)));
}
inline ulong cold_widen(float a) { return soft64_from_f32_bits(cold_f32(a)); }
inline float cold_narrow(ulong a) {
    return cold_f32_value(soft64_to_f32_bits(a));
}
inline ulong cold_dadd(ulong a, ulong b) { return soft64_add(a, b); }
inline ulong cold_dsub(ulong a, ulong b) { return soft64_sub(a, b); }
inline ulong cold_dmul(ulong a, ulong b) { return soft64_mul(a, b); }
inline ulong cold_ddiv(ulong a, ulong b) { return soft64_div(a, b); }
inline ulong cold_dlog(ulong a) { return soft64_plm_log(a); }
inline ulong cold_dexp(ulong a) { return soft64_plm_exp(a); }
inline ulong cold_dpow(ulong a, ulong b) { return soft64_plm_pow(a, b); }
// CUDA's *_cr helpers evaluate the binary64 libm overload, then round once
// to float32. Keep that route separate from cold_powf/logf/expf, which model
// a direct float32 libm call. The WRF39 fork takes these helpers on all of
// its cold DeMott/Koop, snow-moment, and aerosol-efficiency paths.
inline float cold_powf_cr(float a, float b) {
    return cold_narrow(cold_dpow(cold_widen(a), cold_widen(b)));
}
inline float cold_logf_cr(float a) {
    return cold_narrow(cold_dlog(cold_widen(a)));
}
inline float cold_expf_cr(float a) {
    return cold_narrow(cold_dexp(cold_widen(a)));
}
inline float cold_powf_version(float a, float b, bool wrf39) {
    return wrf39 ? cold_powf_cr(a, b) : cold_powf(a, b);
}
inline float cold_logf_version(float a, bool wrf39) {
    return wrf39 ? cold_logf_cr(a) : cold_logf(a);
}
inline float cold_expf_version(float a, bool wrf39) {
    return wrf39 ? cold_expf_cr(a) : cold_expf(a);
}
inline ulong cold_dmin(ulong a, ulong b) {
    return soft64_less(a, b) ? a : b;
}
inline ulong cold_dmax(ulong a, ulong b) {
    return soft64_greater(a, b) ? a : b;
}
inline ulong cold_datan(ulong x) {
    const bool negative = soft64_sign(x);
    ulong a = soft64_abs_bits(x);
    const ulong one = 0x3ff0000000000000ul;
    const ulong half_pi = 0x3ff921fb54442d18ul;
    const ulong quarter_pi = 0x3fe921fb54442d18ul;
    const bool reciprocal = soft64_greater(a, one);
    if (reciprocal) a = cold_ddiv(one, a);
    const bool reduce = soft64_greater(a, 0x3fe0000000000000ul);
    if (reduce) a = cold_ddiv(cold_dsub(a, one), cold_dadd(a, one));
    const ulong z = cold_dmul(a, a);
    ulong term = a;
    ulong sum = a;
    for (int k = 1; k < 32; ++k) {
        term = cold_dmul(term, z) ^ 0x8000000000000000ul;
        sum = cold_dadd(sum,
                        cold_ddiv(term, soft64_from_i32(2 * k + 1)));
    }
    if (reduce) sum = cold_dadd(quarter_pi, sum);
    if (reciprocal) sum = cold_dsub(half_pi, sum);
    return negative ? (sum ^ 0x8000000000000000ul) : sum;
}

inline int cold_nint64(ulong value) {
    const bool negative = soft64_sign(value);
    const ulong magnitude = soft64_abs_bits(value);
    const ulong rounded = soft64_add(magnitude, 0x3fe0000000000000ul);
    const Soft64I32Result result = soft64_to_i32_rz(rounded);
    if (!result.valid) return negative ? (-2147483647 - 1) : 2147483647;
    return negative ? -result.value : result.value;
}

inline float cold_clamp_nwfa(float x) {
    return cold_fmax(11.1e6f, cold_fmin(9999.0e6f, x));
}
inline float cold_clamp_nifa(float x) {
    return cold_fmax(5.0e3f, cold_fmin(9999.0e6f, x));
}
inline float cold_clamp_nc(float x) {
    return cold_fmax(2.0f, cold_fmin(1999.0e6f, x));
}
inline float cold_rslf(float pressure, float temp) {
    const float x = cold_fmax(-80.0f, cold_sub(temp, 273.16f));
    float es = -0.0000000000000321582393f;
    es = cold_add(0.379534310e-11f, cold_mul(x, es));
    es = cold_add(0.00000000702620698f, cold_mul(x, es));
    es = cold_add(0.00000203154182f, cold_mul(x, es));
    es = cold_add(0.000299291081f, cold_mul(x, es));
    es = cold_add(0.0264224321f, cold_mul(x, es));
    es = cold_add(1.43177157f, cold_mul(x, es));
    es = cold_add(44.4606896f, cold_mul(x, es));
    es = cold_add(611.583699f, cold_mul(x, es));
    es = cold_fmin(es, cold_mul(pressure, 0.15f));
    return cold_div(cold_mul(0.622f, es), cold_sub(pressure, es));
}
inline float cold_rsif(float pressure, float temp) {
    const float x = cold_fmax(-80.0f, cold_sub(temp, 273.16f));
    float es = 0.000000000000161444444f;
    es = cold_add(0.000000000105785160f, cold_mul(x, es));
    es = cold_add(0.0000000307839583f, cold_mul(x, es));
    es = cold_add(0.00000521693933f, cold_mul(x, es));
    es = cold_add(0.000565392987f, cold_mul(x, es));
    es = cold_add(0.0402737184f, cold_mul(x, es));
    es = cold_add(1.84672631f, cold_mul(x, es));
    es = cold_add(49.9320233f, cold_mul(x, es));
    es = cold_add(609.868993f, cold_mul(x, es));
    es = cold_fmin(es, cold_mul(pressure, 0.15f));
    return cold_div(cold_mul(0.622f, es),
                    cold_fmax(1.0e-4f, cold_sub(pressure, es)));
}
inline int cold_nu_c(float nc_m3) {
    const float estimate = cold_add(cold_div(1000.0e6f, nc_m3), 0.5f);
    return min(15, int(floor(estimate)) + 2);
}
inline int cold_nu_c_working(float nc_m3) { return cold_nu_c(nc_m3); }
inline int cold_droplet_bin(float nc_m3) {
    const ulong nc = cold_widen(nc_m3);
    const ulong log_ratio = cold_dlog(cold_ddiv(
        nc, 0x412fc397d2e29693ul));
    const ulong raw = cold_dadd(0x3ff0000000000000ul,
        cold_ddiv(cold_dmul(0x4059000000000000ul, log_ratio),
                  0x401c000000000000ul));
    return max(1, min(cold_nint64(raw), 100)) - 1;
}
inline int cold_decade_index(float value, int first_exponent, int table_size) {
    const float log10_value = cold_div(cold_logf(value), 2.302585093f);
    // Non-finite input saturates int(round(inf)) to INT_MAX and the
    // candidate loop's center + 1 overflows, which the GPU ran forever.
    // Every finite float has |log10| < 46, so the clamp never moves one.
    const int center = int(round(clamp(log10_value, -60.0f, 60.0f)));
    int exponent = center;
    for (int candidate = center - 1; candidate <= center + 1; ++candidate) {
        const float scale = cold_powf(10.0f, float(candidate));
        const float mantissa = cold_div(value, scale);
        if (mantissa >= 1.0f && mantissa < 10.0f) {
            exponent = candidate;
            break;
        }
    }
    const float scale = cold_powf(10.0f, float(exponent));
    const int digit = int(cold_div(value, scale));
    return max(0, min(digit + 9 * (exponent - first_exponent) - 1,
                      table_size - 1));
}
inline int cold_decade_index64(ulong value, int first_exponent,
                               int table_size) {
    const ulong log10_value = cold_ddiv(cold_dlog(value),
                                        0x40026bb1bbb55516ul);
    // Same overflow guard as cold_decade_index: finite doubles stay well
    // inside +/-400, non-finite ones would wrap center + 1.
    const int center = max(-400, min(cold_nint64(log10_value), 400));
    int exponent = center;
    for (int candidate = center - 1; candidate <= center + 1; ++candidate) {
        const float scale = cold_powf(10.0f, float(candidate));
        const ulong mantissa = cold_ddiv(value, cold_widen(scale));
        if (!soft64_less(mantissa, 0x3ff0000000000000ul)
                && soft64_less(mantissa, 0x4024000000000000ul)) {
            exponent = candidate;
            break;
        }
    }
    const float scale = cold_powf(10.0f, float(exponent));
    const Soft64I32Result digit = soft64_to_i32_rz(
        cold_ddiv(value, cold_widen(scale)));
    return max(0, min(digit.value + 9 * (exponent - first_exponent) - 1,
                      table_size - 1));
}
inline int cold_in_bin(float xni) {
    return xni > 1.0f ? cold_decade_index(xni, 0, 55) : 0;
}

constant float cold_ccg1[16] = {
    0.0f, 1.0f, 2.0f, 6.0f, 24.0f, 120.0000076f, 720.0000610f,
    5040.001953f, 40319.99609f, 362879.9688f, 3628801.750f,
    39916800.0f, 479001856.0f, 6227022336.0f, 87178297344.0f,
    1307673886720.0f};
constant float cold_ccg2[16] = {
    0.0f, 24.0f, 120.0000076f, 720.0000610f, 5040.001953f,
    40319.99609f, 362879.9688f, 3628801.750f, 39916800.0f,
    479001856.0f, 6227022336.0f, 87178297344.0f, 1307673886720.0f,
    20922782191616.0f, 355687448240128.0f, 6402383730776064.0f};
constant float cold_ccg3[16] = {
    0.0f, 5040.001953f, 40319.99609f, 362879.9688f, 3628801.75f,
    39916800.0f, 479001856.0f, 6227022336.0f, 87178297344.0f,
    1307673886720.0f, 20922782191616.0f, 355687448240128.0f,
    6402383730776064.0f, 121645285000000000.0f,
    2432903398000000000.0f, 51090914450000000000.0f};
constant float cold_ocg1[16] = {
    0.0f, 1.0f, 0.5f, 0.1666666716f, 0.04166666791f, 0.008333332837f,
    0.001388888806f, 0.0001984126284f, 0.00002480158946f,
    0.000002755732112f, 0.0000002755730577f, 0.00000002505210794f,
    0.000000002087674478f, 0.0000000001605904021f,
    0.00000000001147074449f, 0.0000000000007647166320f};
constant float cold_ocg2[16] = {
    0.0f, 0.04166666791f, 0.008333332837f, 0.001388888806f,
    0.0001984126284f, 0.00002480158946f, 0.000002755732112f,
    0.0000002755730577f, 0.00000002505210794f, 0.000000002087674478f,
    0.0000000001605904021f, 0.00000000001147074449f,
    0.0000000000007647166320f, 0.00000000000004779478950f,
    0.000000000000002811457147f, 0.0000000000000001561918299f};
constant float cold_cce2[16] = {
    0.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
    12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};

constant float cold_field_sa[10] = {
    5.065339f, -0.062659f, -3.032362f, 0.029469f, -0.000285f,
    0.31255f, 0.000204f, 0.003199f, 0.0f, -0.015952f};
constant float cold_field_sb[10] = {
    0.476221f, -0.015896f, 0.165977f, 0.007468f, -0.000141f,
    0.060366f, 0.000079f, 0.000594f, 0.0f, -0.003577f};

inline float cold_field_loga(float tc, float moment) {
    float v = cold_field_sa[0];
    v = cold_add(v, cold_mul(cold_field_sa[1], tc));
    v = cold_add(v, cold_mul(cold_field_sa[2], moment));
    v = cold_add(v, cold_mul(cold_mul(cold_field_sa[3], tc), moment));
    v = cold_add(v, cold_mul(cold_mul(cold_field_sa[4], tc), tc));
    v = cold_add(v, cold_mul(cold_mul(cold_field_sa[5], moment), moment));
    v = cold_add(v, cold_mul(cold_mul(cold_mul(cold_field_sa[6], tc), tc), moment));
    v = cold_add(v, cold_mul(cold_mul(cold_mul(cold_field_sa[7], tc), moment), moment));
    v = cold_add(v, cold_mul(cold_mul(cold_mul(cold_field_sa[8], tc), tc), tc));
    v = cold_add(v, cold_mul(cold_mul(cold_mul(cold_field_sa[9], moment), moment), moment));
    return v;
}
inline float cold_field_a(float tc, float moment, bool wrf39) {
    return cold_powf_version(10.0f, cold_field_loga(tc, moment), wrf39);
}
inline float cold_field_b(float tc, float moment) {
    float v = cold_field_sb[0];
    v = cold_add(v, cold_mul(cold_field_sb[1], tc));
    v = cold_add(v, cold_mul(cold_field_sb[2], moment));
    v = cold_add(v, cold_mul(cold_mul(cold_field_sb[3], tc), moment));
    v = cold_add(v, cold_mul(cold_mul(cold_field_sb[4], tc), tc));
    v = cold_add(v, cold_mul(cold_mul(cold_field_sb[5], moment), moment));
    v = cold_add(v, cold_mul(cold_mul(cold_mul(cold_field_sb[6], tc), tc), moment));
    v = cold_add(v, cold_mul(cold_mul(cold_mul(cold_field_sb[7], tc), moment), moment));
    v = cold_add(v, cold_mul(cold_mul(cold_mul(cold_field_sb[8], tc), tc), tc));
    v = cold_add(v, cold_mul(cold_mul(cold_mul(cold_field_sb[9], moment), moment), moment));
    return v;
}
inline float cold_snow_number(float smob, float smoc, bool wrf39) {
    const float m0 = cold_div(smob, smoc);
    const float mrat = cold_mul(cold_mul(cold_mul(smob, m0), m0), m0);
    const float slam1 = cold_mul(m0, 20.78f);
    const float slam2 = cold_mul(m0, 3.29f);
    const float first = cold_div(cold_mul(mrat, 490.6f), slam1);
    const float second = cold_div(
        cold_mul(cold_mul(cold_mul(mrat, 17.46f),
                          cold_powf_version(m0, 0.6357f, wrf39)),
                 0.8980315328f),
        cold_powf_version(slam2, 1.635699987f, wrf39));
    return cold_add(first, second);
}

inline float cold_cloud_dist(float rc, float nc_per_kg, float rho,
                             thread int &nu_c_entry,
                             thread ulong &lambda_entry, bool wrf39) {
    float nc = cold_clamp_nc(cold_mul(nc_per_kg, rho));
    nu_c_entry = cold_nu_c(nc);
    const int nu = nu_c_entry;
    const float arg = cold_div(
        cold_mul(cold_mul(cold_mul(nc, 5.235988159e2f),
                          cold_ccg2[nu]), cold_ocg1[nu]), rc);
    ulong lambda = cold_widen(cold_powf_version(arg, 0.3333333433f, wrf39));
    const float diameter = cold_narrow(cold_ddiv(
        cold_widen(cold_add(cold_add(3.0f, float(nu)), 1.0f)), lambda));
    if (diameter < 1.0e-6f) {
        lambda = cold_widen(cold_div(cold_cce2[nu], 1.0e-6f));
    } else if (diameter > 100.0e-6f) {
        lambda = cold_widen(cold_div(cold_cce2[nu], 100.0e-6f));
    }
    const ulong lam3 = cold_dpow(lambda, 0x4008000000000000ul);
    const float prefactor = cold_div(
        cold_mul(cold_mul(cold_ccg1[nu], cold_ocg2[nu]), rc),
        5.235988159e2f);
    const ulong diagnosed = cold_dmul(cold_widen(prefactor), lam3);
    nc = cold_narrow(diagnosed);
    if (nc > 1999.0e6f) nc = 1999.0e6f;
    lambda_entry = lambda;
    return nc;
}

inline void cold_bound_rain_number(float rain_mass, float density,
                                   thread float &rain_number_per_kg) {
    if (rain_mass <= 1.0e-12f) {
        rain_number_per_kg = 0.0f;
        return;
    }
    float number = cold_fmax(1.0e-6f,
                             cold_mul(rain_number_per_kg, density));
    float lambda = cold_powf(cold_div(
        cold_mul(cold_mul(5.235988159e2f, 6.0f), number), rain_mass),
        0.3333333433f);
    float mvd = cold_div(3.672f, lambda);
    if (mvd > 2.5e-3f) mvd = 2.5e-3f;
    else if (mvd < 37.5e-6f) mvd = 37.5e-6f;
    else return;
    lambda = cold_div(3.672f, mvd);
    number = cold_mul(cold_div(cold_mul(1.0f / 6.0f, rain_mass),
                               5.235988159e2f),
                      cold_mul(cold_mul(lambda, lambda), lambda));
    rain_number_per_kg = cold_div(number, density);
}

inline bool cold_entry_rain_distribution(
        float rain_per_kg, float rain_number_per_kg, float density,
        thread float &rain_number, thread ulong &rain_lambda,
        thread float &rain_mvd, thread ulong &rain_intercept_n0,
        bool wrf39) {
    constexpr float am_r = 5.235988159e2f;
    constexpr float obmr = 0.3333333433f;
    constexpr float crg2_org3 = 1.0f / 6.0f;
    float rr;
    float nr;
    const bool active = rain_per_kg > 1.0e-12f;
    if (active) {
        rr = cold_mul(rain_per_kg, density);
        nr = cold_fmax(1.0e-6f,
                       cold_mul(rain_number_per_kg, density));
        if (nr <= 1.0e-6f) {
            const ulong initial_lambda = cold_widen(cold_div(3.672f, 1.0e-3f));
            const float initial_prefactor = cold_div(
                cold_mul(crg2_org3, rr), am_r);
            const float initial_number = cold_narrow(cold_dmul(
                cold_widen(initial_prefactor),
                cold_dpow(initial_lambda, 0x4008000000000000ul)));
            nr = initial_number;
        }
        const float argument = cold_div(cold_mul(cold_mul(am_r, 6.0f), nr), rr);
        ulong lambda = cold_widen(cold_powf_version(argument, obmr, wrf39));
        float mvd = cold_narrow(cold_ddiv(0x400d604189374bc7ul, lambda));
        if (mvd > 2.5e-3f) {
            mvd = 2.5e-3f;
            lambda = cold_widen(cold_div(3.672f, mvd));
            const float clamp_prefactor = cold_div(
                cold_mul(crg2_org3, rr), am_r);
            nr = cold_narrow(cold_dmul(
                cold_widen(clamp_prefactor),
                cold_dpow(lambda, 0x4008000000000000ul)));
        } else if (mvd < 37.5e-6f) {
            mvd = 37.5e-6f;
            lambda = cold_widen(cold_div(3.672f, mvd));
            const float clamp_prefactor = cold_div(
                cold_mul(crg2_org3, rr), am_r);
            nr = cold_narrow(cold_dmul(
                cold_widen(clamp_prefactor),
                cold_dpow(lambda, 0x4008000000000000ul)));
        }
    } else {
        rr = 1.0e-12f;
        nr = 1.0e-6f;
    }
    const float arg = cold_div(cold_mul(cold_mul(am_r, 6.0f), nr), rr);
    const ulong lambda = cold_widen(cold_powf_version(arg, obmr, wrf39));
    rain_number = nr;
    rain_lambda = lambda;
    rain_mvd = cold_narrow(cold_ddiv(0x400d604189374bc7ul, lambda));
    rain_intercept_n0 = cold_dmul(cold_widen(nr), lambda);
    return active;
}

inline float cold_sqrtf(float x) {
    return cold_f32_value(soft32_sqrt(cold_f32(x)));
}
inline float cold_log10f(float x) {
    return cold_div(cold_logf(x), 2.302585093f);
}
inline float cold_ice_demott(float tempc, float rho, float nifa_m3,
                             bool wrf39) {
    float nifa_cc = cold_div(
        cold_mul(cold_mul(nifa_m3, 1.292283773f), 1.0e-6f), rho);
    if (!wrf39) nifa_cc = cold_fmax(0.5f, nifa_cc);
    const float exponent = cold_add(cold_mul(-0.0264f, tempc), 0.0033f);
    float xni = cold_mul(
        cold_mul(5.94e-5f, cold_powf_version(-tempc, 3.33f, wrf39)),
        cold_powf_version(nifa_cc, exponent, wrf39));
    xni = cold_mul(cold_div(cold_mul(xni, rho), 1.292283773f), 1000.0f);
    return cold_fmax(0.0f, xni);
}
inline float cold_ice_koop(float temp, float qv, float qvs,
                           float naero, float dt, bool wrf39) {
    const float satw = cold_div(qv, qvs);
    float mu_diff = cold_add(210368.0f, cold_mul(131.438f, temp));
    mu_diff = cold_sub(mu_diff, cold_div(3.32373e6f, temp));
    mu_diff = cold_sub(mu_diff, cold_mul(41729.1f,
                                         cold_logf_version(temp, wrf39)));
    const float a_w_i = cold_expf_version(cold_div(mu_diff,
        cold_mul(8.314f, temp)), wrf39);
    const float d = cold_sub(satw, a_w_i);
    float log_j = cold_add(-906.7f, cold_mul(8502.0f, d));
    log_j = cold_sub(log_j, cold_mul(cold_mul(26924.0f, d), d));
    log_j = cold_add(log_j,
        cold_mul(cold_mul(cold_mul(29180.0f, d), d), d));
    log_j = cold_fmin(20.0f, log_j);
    const float j_rate = cold_powf_version(10.0f, log_j, wrf39);
    const float prob_h = cold_fmin(
        cold_sub(1.0f, cold_expf_version(cold_mul(
            cold_mul(-j_rate, 6.544983879e-17f), dt), wrf39)), 1.0f);
    const float xni = prob_h > 0.0f
        ? cold_fmin(cold_mul(prob_h, naero), 1000.0e3f) : 0.0f;
    return cold_fmax(0.0f, xni);
}

struct ColdEntryState {
    float temp0, qv0, qvsi, qvs, ssati, ssatw;
    float rho, orho, inverse_dt, dt, tempc, inverse_temp;
    float diffusivity, viscosity, conductivity, inverse_cp;
    float saturation_slope, saturation_curvature, gamma, alpha;
    float vapor_geometry, vapor_limit, vapor_limit_sum, density_factor;
    float nwfa_work, nifa_work, cloud_mass, nc_work, cloud_mvd;
    bool has_cloud, nucleation_active;
    int nu_c_entry, nu_c;
    ulong inverse_dt_d, cloud_lambda;
};

inline void cold_bound_ice_number(float ice_mass, float density,
                                  thread float &ice_number_per_kg,
                                  bool wrf39);
inline float cold_eff_aero(float diameter, float aerosol_diameter,
                           float viscosity, float rho, float temperature,
                           int species, bool wrf39);

// Source-order translation of the cold call's environmental and entry-state
// setup (:422-542). The production kernel will consume this record before
// evaluating the process groups; table-dependent rates remain below this
// helper boundary.
inline ColdEntryState cold_prepare_entry(
        float temperature, float pressure, float qv, float dt,
        float qc, float nc_entry, float nwfa_entry, float nifa_entry,
        bool wrf39) {
    ColdEntryState e;
    e.temp0 = temperature;
    e.qv0 = cold_fmax(1.0e-10f, qv);
    e.qvsi = cold_rsif(pressure, e.temp0);
    e.qvs = cold_rslf(pressure, e.temp0);
    e.ssati = cold_sub(cold_div(e.qv0, e.qvsi), 1.0f);
    e.ssatw = cold_sub(cold_div(e.qv0, e.qvs), 1.0f);
    if (fabs(e.ssati) < 1.0e-15f) e.ssati = 0.0f;
    if (fabs(e.ssatw) < 1.0e-15f) e.ssatw = 0.0f;
    e.nucleation_active = e.ssati >= 0.25f
        || (e.ssatw > 1.0e-15f && e.temp0 < 253.15f);
    e.rho = cold_div(cold_mul(0.622f, pressure),
        cold_mul(cold_mul(287.04f, e.temp0), cold_add(e.qv0, 0.622f)));
    e.orho = cold_div(1.0f, e.rho);
    e.inverse_dt = cold_div(1.0f, dt);
    e.dt = dt;
    e.inverse_dt_d = cold_widen(e.inverse_dt);
    e.tempc = cold_sub(e.temp0, 273.15f);
    e.inverse_temp = cold_div(1.0f, e.temp0);
    e.diffusivity = cold_mul(cold_mul(2.11e-5f,
        cold_powf(cold_div(e.temp0, 273.15f), 1.94f)),
        cold_div(101325.0f, pressure));
    e.viscosity = cold_mul(cold_sub(
        cold_add(1.718f, cold_mul(0.0049f, e.tempc)),
        cold_mul(cold_mul(1.2e-5f, e.tempc), e.tempc)), 1.0e-5f);
    e.conductivity = cold_mul(
        cold_mul(cold_add(5.69f, cold_mul(0.0168f, e.tempc)), 1.0e-5f),
        418.936f);
    e.inverse_cp = cold_div(1.0f,
        cold_mul(1004.0f, cold_add(1.0f, cold_mul(0.887f, e.qv0))));
    const float saturated_density = cold_mul(e.rho, e.qvsi);
    const float slope_term = cold_mul(e.inverse_temp,
        cold_sub(cold_mul(cold_mul(2.834e6f, e.inverse_temp),
                          1.0f / 461.5f), 1.0f));
    e.saturation_slope = cold_mul(saturated_density, slope_term);
    float curvature_term = cold_mul(slope_term, slope_term);
    float curvature_subtract = cold_mul(2.0f, 2.834e6f);
    curvature_subtract = cold_mul(curvature_subtract, e.inverse_temp);
    curvature_subtract = cold_mul(curvature_subtract, e.inverse_temp);
    curvature_subtract = cold_mul(curvature_subtract, e.inverse_temp);
    curvature_subtract = cold_mul(curvature_subtract, 1.0f / 461.5f);
    curvature_term = cold_sub(curvature_term, curvature_subtract);
    curvature_term = cold_add(curvature_term,
        cold_mul(e.inverse_temp, e.inverse_temp));
    e.saturation_curvature = cold_mul(saturated_density, curvature_term);
    e.gamma = cold_mul(cold_div(cold_mul(2.834e6f, e.diffusivity),
                                e.conductivity), e.saturation_slope);
    const float gamma_ratio = cold_div(e.gamma, cold_add(1.0f, e.gamma));
    float alpha = cold_mul(0.5f, gamma_ratio);
    alpha = cold_mul(alpha, gamma_ratio);
    alpha = cold_mul(alpha, e.saturation_curvature);
    alpha = cold_div(alpha, e.saturation_slope);
    alpha = cold_mul(alpha, saturated_density);
    alpha = cold_div(alpha, e.saturation_slope);
    e.alpha = cold_fmax(1.0e-9f, alpha);
    float xsat = e.ssati;
    if (fabs(xsat) < 1.0e-9f) xsat = 0.0f;
    const float xsat2 = cold_mul(xsat, xsat);
    const float alpha2 = cold_mul(e.alpha, e.alpha);
    float geometry = cold_sub(1.0f, cold_mul(e.alpha, xsat));
    geometry = cold_add(geometry, cold_mul(cold_mul(2.0f, alpha2), xsat2));
    geometry = cold_sub(geometry, cold_mul(
        cold_mul(cold_mul(5.0f, alpha2), e.alpha),
        cold_mul(xsat2, xsat)));
    e.vapor_geometry = cold_div(cold_mul(
        cold_mul(4.0f, 3.1415926536f), geometry), cold_add(1.0f, e.gamma));
    e.vapor_limit = cold_mul(cold_sub(e.qv0, e.qvsi), e.rho);
    e.vapor_limit = cold_mul(e.vapor_limit, e.inverse_dt);
    e.vapor_limit = cold_mul(e.vapor_limit, 0.999f);
    e.vapor_limit_sum = wrf39
        ? cold_mul(cold_mul(cold_sub(e.qv0, e.qvsi), e.inverse_dt), 0.999f)
        : e.vapor_limit;
    const float reference_density = cold_div(101325.0f,
        cold_mul(287.05f, 298.0f));
    e.density_factor = cold_sqrtf(cold_div(reference_density, e.rho));
    e.nwfa_work = cold_clamp_nwfa(cold_mul(nwfa_entry, e.rho));
    e.nifa_work = cold_clamp_nifa(cold_mul(nifa_entry, e.rho));
    e.has_cloud = qc > 1.0e-12f;
    e.cloud_mass = e.has_cloud ? cold_mul(qc, e.rho) : 1.0e-12f;
    e.nu_c_entry = 0;
    e.nu_c = 0;
    e.cloud_lambda = 0ul;
    e.nc_work = 2.0f;
    e.cloud_mvd = 1.0e-6f;
    if (e.has_cloud) {
        e.nc_work = cold_cloud_dist(e.cloud_mass, nc_entry, e.rho,
                                    e.nu_c_entry, e.cloud_lambda, wrf39);
        e.nu_c = cold_nu_c_working(e.nc_work);
        const float lambda_arg = cold_div(
            cold_mul(cold_mul(cold_mul(e.nc_work, 5.235988159e2f),
                              cold_ccg2[e.nu_c]), cold_ocg1[e.nu_c]),
            e.cloud_mass);
        e.cloud_lambda = cold_widen(cold_powf_version(
            lambda_arg, 0.3333333433f, wrf39));
        const float mvd_numerator = cold_add(
            cold_add(3.0f, float(e.nu_c)), 0.672f);
        const float mvd = cold_narrow(cold_ddiv(
            cold_widen(mvd_numerator), e.cloud_lambda));
        e.cloud_mvd = wrf39 ? mvd
            : cold_fmax(1.0e-6f, cold_fmin(mvd, 50.0e-6f));
    }
    return e;
}

struct ColdSnowDistribution {
    float mass, tc0, smob, smoc, smo0, smoe, diameter, number_ns;
    bool active;
};

// WRF snow moments used by riming, deposition, aerosol scavenging, and the
// Koop population gate (:2027-2101).
inline ColdSnowDistribution cold_prepare_snow(
        float qs, float rho, float tempc, bool wrf39) {
    ColdSnowDistribution s;
    s.mass = cold_mul(qs, rho);
    s.active = qs > 1.0e-12f;
    s.tc0 = cold_fmin(-0.1f, tempc);
    s.smob = s.smoc = s.smo0 = s.smoe = 0.0f;
    s.diameter = s.number_ns = 0.0f;
    if (s.active) {
        s.smob = cold_mul(s.mass, 1.0f / 0.069f);
        s.smoc = cold_mul(cold_field_a(s.tc0, 3.0f, wrf39),
            cold_powf_version(s.smob, cold_field_b(s.tc0, 3.0f), wrf39));
        s.smo0 = cold_mul(cold_field_a(s.tc0, 0.0f, wrf39),
            cold_powf_version(s.smob, cold_field_b(s.tc0, 0.0f), wrf39));
        s.smoe = cold_mul(cold_field_a(s.tc0, 2.55f, wrf39),
            cold_powf_version(s.smob, cold_field_b(s.tc0, 2.55f), wrf39));
        s.diameter = cold_div(s.smoc, s.smob);
        s.number_ns = cold_snow_number(s.smob, s.smoc, wrf39);
    }
    return s;
}

struct ColdGraupelDistribution {
    float mass, number, diameter;
    ulong lambda, inverse_lambda, intercept;
};

// Production entry graupel diagnosis (:623-653). The number is a mixing
// volume concentration; the prognostic shadow is handled during state apply.
inline ColdGraupelDistribution cold_prepare_graupel(
        float qg, float rho, float intercept_entry, bool wrf39) {
    ColdGraupelDistribution g;
    g.mass = cold_mul(qg, rho);
    g.number = 0.0f;
    g.diameter = 0.0f;
    g.lambda = cold_widen(1.0f);
    g.inverse_lambda = cold_widen(1.0f);
    g.intercept = 0ul;
    if (wrf39) {
        if (qg > 1.0e-12f) {
            const float am_g = cold_div(cold_mul(3.1415926536f, 500.0f), 6.0f);
            const ulong n0 = cold_widen(intercept_entry);
            ulong slope_base = cold_dmul(n0, cold_widen(am_g));
            slope_base = cold_dmul(slope_base, cold_widen(6.0f));
            slope_base = cold_ddiv(slope_base, cold_widen(g.mass));
            g.lambda = cold_dpow(slope_base, 0x3fd0000000000000ul);
            g.inverse_lambda = cold_ddiv(cold_widen(1.0f), g.lambda);
            g.diameter = cold_narrow(cold_dmul(
                cold_widen(4.0f), g.inverse_lambda));
            g.intercept = cold_dmul(
                cold_ddiv(n0, g.lambda), g.lambda);
        }
        return g;
    }
    if (g.mass >= 1.0e-6f) {
        const float am_g = cold_div(cold_mul(3.1415926536f, 400.0f), 6.0f);
        float power = cold_add(3.0f, cold_mul(2.0f / 7.0f,
            cold_add(cold_log10f(cold_fmax(1.0e-9f, g.mass)), 8.0f)));
        power = cold_fmax(2.0f, cold_fmin(6.0f, power));
        const float guess = cold_powf(10.0f, power);
        float lambda_f = cold_powf(cold_div(cold_mul(cold_mul(
            guess, am_g), 6.0f), g.mass), 0.25f);
        ulong lambda = cold_widen(lambda_f);
        const float mvd = cold_narrow(cold_ddiv(
            0x400d604189374bc7ul, lambda));
        if (mvd > 25.4e-3f) {
            lambda = cold_ddiv(0x400d604189374bc7ul,
                               0x3f9a027525460aa6ul);
        } else if (mvd < 50.0e-6f) {
            lambda = cold_ddiv(0x400d604189374bc7ul,
                               0x3f0a36e2eb1c432dul);
        }
        const ulong lambda2 = cold_dmul(lambda, lambda);
        const ulong lambda3 = cold_dmul(lambda2, lambda);
        const float lambda3_f = cold_narrow(lambda3);
        float number = cold_div(cold_mul(1.0f / 6.0f, g.mass), am_g);
        number = cold_mul(number, lambda3_f);
        const ulong intercept = cold_dmul(cold_widen(number), lambda);
        const ulong inv_lambda = cold_ddiv(cold_widen(1.0f), lambda);
        g.number = number;
        g.diameter = cold_narrow(cold_dmul(cold_widen(4.0f), inv_lambda));
        g.lambda = lambda;
        g.inverse_lambda = inv_lambda;
        g.intercept = intercept;
    }
    return g;
}

// Production cell entry and early exits (:354-424). `snow_velocity_boost`
// is reset before either the warm return or the hydrometeor-free return.
inline bool cold_network_entry(
        float qi, thread float &ni, float qs, float qg, float qr, float qc,
        float temperature, float pressure, float qv, float dt,
        float nc_entry, float nwfa_entry, float nifa_entry,
        bool wrf39, thread float &snow_velocity_boost,
        thread ColdEntryState &entry) {
    snow_velocity_boost = 1.0f;
    if (temperature >= 273.15f) {
        if (qi > 0.0f || ni > 0.0f) {
            const float qv_entry = cold_fmax(1.0e-10f, qv);
            const float rho_entry = cold_div(cold_mul(0.622f, pressure),
                cold_mul(cold_mul(287.04f, temperature),
                         cold_add(qv_entry, 0.622f)));
            cold_bound_ice_number(cold_mul(qi, rho_entry), rho_entry, ni,
                                  wrf39);
        }
        return false;
    }
    const float qv0 = cold_fmax(1.0e-10f, qv);
    float ssati = cold_sub(cold_div(qv0, cold_rsif(pressure, temperature)), 1.0f);
    float ssatw = cold_sub(cold_div(qv0, cold_rslf(pressure, temperature)), 1.0f);
    if (fabs(ssati) < 1.0e-15f) ssati = 0.0f;
    if (fabs(ssatw) < 1.0e-15f) ssatw = 0.0f;
    const bool nucleation_active = ssati >= 0.25f
        || (ssatw > 1.0e-15f && temperature < 253.15f);
    if (qi <= 1.0e-12f && qs <= 1.0e-12f && qg <= 1.0e-12f
            && qr <= 1.0e-12f && qc <= 1.0e-12f
            && !nucleation_active) return false;
    entry = cold_prepare_entry(temperature, pressure, qv, dt, qc,
                               nc_entry, nwfa_entry, nifa_entry, wrf39);
    return true;
}

struct ColdIceRates {
    float ice_particle_mass, ice_number, rain_number, rain_mvd;
    ulong rain_lambda, rain_intercept_n0;
    bool rain_active;
    ulong ice_rate, ice_to_snow_rate, ice_number_rate;
    ulong autoconversion_rate, autoconversion_number_rate;
    ulong rain_ice_ice_rate, rain_ice_rain_rate;
    ulong rain_ice_ice_number_rate, rain_ice_rain_number_rate;
    ulong rain_ice_graupel_rate, rain_self_number_rate;
};

// Translation of the entry rain distribution, ice deposition, ice
// autoconversion, and rain-collects-ice group (:677-822).
inline ColdIceRates cold_ice_processes(
        float qi, float ni, float qr, float nr,
        thread const ColdEntryState &e,
        device const ulong *ice_deposition_partition,
        device const ulong *ice_to_snow_mass,
        device const ulong *ice_to_snow_number, bool wrf39) {
    ColdIceRates r;
    const ulong zero = 0ul;
    r.ice_particle_mass = 1.0e-12f;
    r.ice_number = cold_fmax(1.0e-6f, cold_mul(ni, e.rho));
    r.rain_number = 1.0e-6f;
    r.rain_mvd = 0.0f;
    r.rain_lambda = 0x3ff0000000000000ul;
    r.rain_intercept_n0 = zero;
    r.rain_active = cold_entry_rain_distribution(
        qr, nr, e.rho, r.rain_number, r.rain_lambda, r.rain_mvd,
        r.rain_intercept_n0, wrf39);
    r.ice_rate = r.ice_to_snow_rate = r.ice_number_rate = zero;
    r.autoconversion_rate = r.autoconversion_number_rate = zero;
    r.rain_ice_ice_rate = r.rain_ice_rain_rate = zero;
    r.rain_ice_ice_number_rate = r.rain_ice_rain_number_rate = zero;
    r.rain_ice_graupel_rate = r.rain_self_number_rate = zero;

    const float rain_mass = cold_mul(qr, e.rho);
    if (r.rain_active && r.rain_mvd > 50.0e-6f) {
        const float rain_exp_arg = cold_mul(
            2300.0f, cold_sub(r.rain_mvd, 1950.0e-6f));
        const float rain_exp = wrf39 ? exp(rain_exp_arg)
                                     : cold_expf(rain_exp_arg);
        const float efficiency = cold_sub(1.0f, rain_exp);
        float self_rate = cold_mul(efficiency, 2.0f);
        self_rate = cold_mul(self_rate, r.rain_number);
        self_rate = cold_mul(self_rate, rain_mass);
        // WRF's self-collection rate has no odts factor.
        r.rain_self_number_rate = cold_widen(self_rate);
    }

    const float ice_mass = cold_mul(qi, e.rho);
    if (qi <= 1.0e-12f) return r;
    constexpr float am_i = 3.1415926536f * 890.0f / 6.0f;
    if (r.ice_number <= 1.0e-6f) {
        const ulong seed_diameter = wrf39 ? 0x3efa36e2eb1c432dul
                                          : 0x3ed4f8b588e368f1ul;
        const ulong lambda5 = cold_ddiv(0x4010000000000000ul,
                                        seed_diameter);
        const ulong lambda5_3 = cold_dpow(lambda5, 0x4008000000000000ul);
        const float prefactor = cold_div(cold_mul(1.0f / 6.0f, ice_mass), am_i);
        r.ice_number = cold_fmin(wrf39 ? 499.0e3f : 999.0e3f,
            cold_mul(prefactor, cold_narrow(lambda5_3)));
    }
    const float ice_lambda_arg = cold_div(
        cold_mul(cold_mul(am_i, 6.0f), r.ice_number), ice_mass);
    ulong lambda = cold_widen(cold_powf(ice_lambda_arg, 0.3333333433f));
    ulong inverse_lambda = cold_ddiv(0x3ff0000000000000ul, lambda);
    float mean_diameter = cold_narrow(cold_ddiv(
        0x4010000000000000ul, lambda));
    if (mean_diameter < 5.0e-6f) {
        lambda = cold_ddiv(0x4010000000000000ul, 0x3ed4f8b588e368f1ul);
        inverse_lambda = cold_ddiv(0x3ff0000000000000ul, lambda);
        r.ice_number = cold_fmin(wrf39 ? 499.0e3f : 999.0e3f,
            cold_mul(cold_div(cold_mul(1.0f / 6.0f, ice_mass), am_i),
                     cold_narrow(cold_dpow(lambda, 0x4008000000000000ul))));
        mean_diameter = 5.0e-6f;
    } else if (mean_diameter > 300.0e-6f) {
        lambda = cold_ddiv(0x4010000000000000ul, 0x3f33a92a30553261ul);
        inverse_lambda = cold_ddiv(0x3ff0000000000000ul, lambda);
        r.ice_number = cold_mul(
            cold_div(cold_mul(1.0f / 6.0f, ice_mass), am_i),
            cold_narrow(cold_dpow(lambda, 0x4008000000000000ul)));
        mean_diameter = 300.0e-6f;
    }
    const float min_diameter = cold_powf_version(
        cold_div(1.0e-12f, am_i), 0.3333333433f, wrf39);
    const float floored_ice_diameter = cold_fmax(min_diameter, mean_diameter);
    r.ice_particle_mass = cold_mul(cold_mul(cold_mul(am_i,
        floored_ice_diameter), floored_ice_diameter), floored_ice_diameter);
    const int mass_bin = ice_mass > 1.0e-10f
        ? cold_decade_index(ice_mass, -10, 64) : 0;
    const int number_bin = r.ice_number > 1.0f
        ? cold_decade_index(r.ice_number, 0, 55) : 0;
    float total_rate_f = cold_mul(0.5f, e.vapor_geometry);
    total_rate_f = cold_mul(total_rate_f, e.diffusivity);
    total_rate_f = cold_mul(total_rate_f, e.ssati);
    total_rate_f = cold_mul(total_rate_f, cold_mul(e.rho, e.qvsi));
    total_rate_f = cold_mul(total_rate_f, r.ice_number);
    ulong total_rate = cold_dmul(cold_widen(total_rate_f), inverse_lambda);
    if (soft64_greater(total_rate, zero)) {
        total_rate = cold_dmin(total_rate, cold_widen(e.vapor_limit));
        const ulong table_idx = ulong(mass_bin + 64 * number_bin);
        const ulong fraction = ice_deposition_partition[table_idx];
        r.ice_rate = cold_dmul(fraction, total_rate);
        r.ice_to_snow_rate = cold_dmul(
            cold_dsub(0x3ff0000000000000ul, fraction), total_rate);
    } else {
        const ulong lower_mass_limit = cold_widen(
            -cold_mul(ice_mass, e.inverse_dt));
        total_rate = cold_dmax(lower_mass_limit, total_rate);
        total_rate = cold_dmax(cold_widen(e.vapor_limit), total_rate);
        const float particle_diameter = cold_fmax(min_diameter, mean_diameter);
        const float particle_mass = cold_mul(cold_mul(cold_mul(am_i,
            particle_diameter), particle_diameter), particle_diameter);
        r.ice_number_rate = cold_ddiv(total_rate, cold_widen(particle_mass));
        r.ice_number_rate = cold_dmax(
            cold_widen(-cold_mul(r.ice_number, e.inverse_dt)),
            r.ice_number_rate);
        r.ice_rate = total_rate;
    }
    const ulong table_idx = ulong(mass_bin + 64 * number_bin);
    const float snow_d0 = wrf39 ? 200.0e-6f : 300.0e-6f;
    if (mass_bin == 63 || mean_diameter > cold_mul(5.0f, snow_d0)) {
        r.autoconversion_rate = cold_widen(cold_mul(ice_mass, 0.99f));
        r.autoconversion_rate = cold_dmul(r.autoconversion_rate,
                                          cold_widen(e.inverse_dt));
        r.autoconversion_number_rate = cold_widen(cold_mul(r.ice_number, 0.95f));
        r.autoconversion_number_rate = cold_dmul(r.autoconversion_number_rate,
                                                 cold_widen(e.inverse_dt));
    } else if (mean_diameter >= cold_mul(0.1f, snow_d0)) {
        const ulong mass_cap = cold_dmul(
            cold_widen(cold_mul(ice_mass, 0.99f)), cold_widen(e.inverse_dt));
        const ulong mass_table = cold_dmul(ice_to_snow_mass[table_idx],
                                           cold_widen(e.inverse_dt));
        r.autoconversion_rate = cold_dmin(mass_cap, mass_table);
        const ulong number_cap = cold_dmul(
            cold_widen(cold_mul(r.ice_number, 0.95f)),
            cold_widen(e.inverse_dt));
        const ulong number_table = cold_dmul(ice_to_snow_number[table_idx],
                                             cold_widen(e.inverse_dt));
        r.autoconversion_number_rate = cold_dmin(number_cap, number_table);
    }
    if (rain_mass >= 1.0e-6f
            && r.rain_mvd > cold_mul(4.0f, floored_ice_diameter)) {
        const ulong rain_intercept = cold_dmul(cold_widen(r.rain_number),
                                              r.rain_lambda);
        const ulong shifted_lambda = cold_dadd(r.rain_lambda,
                                               cold_widen(195.0f));
        const float mass_prefactor = cold_mul(cold_mul(cold_mul(3.1415926536f,
            0.25f), 4854.0f), 6.0f);
        const float am_r = cold_div(
            cold_mul(3.1415926536f, 1000.0f), 6.0f);
        const float rain_mass_prefactor = cold_mul(cold_mul(cold_mul(cold_mul(
            3.1415926536f, 0.25f), am_r), 4854.0f), 720.0f);
        const float ice_mass_coeff = cold_mul(cold_mul(cold_mul(
            e.density_factor, mass_prefactor), 0.95f), ice_mass);
        r.rain_ice_ice_rate = cold_dmul(cold_widen(ice_mass_coeff),
            cold_dmul(rain_intercept,
                cold_dpow(shifted_lambda, soft64_from_i32(-4))));
        const float rain_mass_coeff = cold_mul(cold_mul(cold_mul(
            e.density_factor, rain_mass_prefactor), 0.95f), r.ice_number);
        r.rain_ice_rain_rate = cold_dmul(cold_widen(rain_mass_coeff),
            cold_dmul(rain_intercept,
                cold_dpow(shifted_lambda, soft64_from_i32(-7))));
        r.rain_ice_rain_rate = cold_dmin(r.rain_ice_rain_rate,
            cold_dmul(cold_widen(rain_mass), cold_widen(e.inverse_dt)));
        r.rain_ice_ice_number_rate = cold_ddiv(
            r.rain_ice_ice_rate, cold_widen(r.ice_particle_mass));
        const float rain_number_coeff = cold_mul(cold_mul(cold_mul(
            e.density_factor, mass_prefactor), 0.95f), r.ice_number);
        r.rain_ice_rain_number_rate = cold_dmul(
            cold_widen(rain_number_coeff),
            cold_dmul(rain_intercept,
                cold_dpow(shifted_lambda, soft64_from_i32(-4))));
        if (!wrf39) {
            r.rain_ice_rain_number_rate = cold_dmin(
                r.rain_ice_rain_number_rate,
                cold_dmul(cold_widen(r.rain_number), cold_widen(e.inverse_dt)));
        }
        r.rain_ice_graupel_rate = cold_dadd(r.rain_ice_ice_rate,
                                            r.rain_ice_rain_rate);
    }
    return r;
}

struct ColdRainRates {
    ulong freeze_ice_rate, freeze_ice_number_rate;
    ulong freeze_graupel_rate, freeze_graupel_number_rate;
    ulong freeze_rain_number_rate;
    ulong rain_snow_rain_rate, rain_snow_category_rate;
    ulong rain_snow_graupel_rate, rain_snow_number_rate;
    ulong rain_graupel_rain_rate, rain_graupel_graupel_rate;
    ulong rain_graupel_number_rate;
};

inline ulong cold_wrf39_table_intercept(float graupel_mass,
                                         ulong inverse_lambda) {
    const ulong lambda = cold_ddiv(0x3ff0000000000000ul, inverse_lambda);
    const float am_g = cold_div(cold_mul(3.1415926536f, 500.0f), 6.0f);
    const float prefix = cold_div(
        cold_mul(0.16666667163372040f, graupel_mass), am_g);
    return cold_dmul(cold_widen(prefix),
                     cold_dpow(lambda, 0x4010000000000000ul));
}

// Source-order translation of the cold-rain collision and freezing group
// (:2492-2553, :2594-2603). Table storage is binary64 IEEE words in physical
// Fortran order; offsets follow the CUDA expressions directly.
inline ColdRainRates cold_rain_processes(
        float qr, float qs, float qg, int nuclei_bin,
        thread const ColdEntryState &e, thread const ColdIceRates &ice,
        device const ulong *rain_to_ice_mass,
        device const ulong *rain_to_ice_number,
        device const ulong *rain_to_graupel_mass,
        device const ulong *rain_to_graupel_number,
        device const ulong *tcs_racs1, device const ulong *tmr_racs1,
        device const ulong *tcs_racs2, device const ulong *tmr_racs2,
        device const ulong *tcr_sacr1, device const ulong *tms_sacr1,
        device const ulong *tcr_sacr2, device const ulong *tms_sacr2,
        device const ulong *tnr_racs1, device const ulong *tnr_racs2,
        device const ulong *tnr_sacr1, device const ulong *tnr_sacr2,
        device const ulong *tcg_racg, device const ulong *tmr_racg,
        device const ulong *tcr_gacr, device const ulong *tnr_racg,
        device const ulong *tnr_gacr, bool wrf39,
        ulong graupel_inverse_lambda) {
    ColdRainRates r;
    r.freeze_ice_rate = r.freeze_ice_number_rate = 0ul;
    r.freeze_graupel_rate = r.freeze_graupel_number_rate = 0ul;
    r.freeze_rain_number_rate = 0ul;
    r.rain_snow_rain_rate = r.rain_snow_category_rate = 0ul;
    r.rain_snow_graupel_rate = r.rain_snow_number_rate = 0ul;
    r.rain_graupel_rain_rate = r.rain_graupel_graupel_rate = 0ul;
    r.rain_graupel_number_rate = 0ul;
    if (!ice.rain_active) return r;

    const float rain_mass = cold_mul(qr, e.rho);
    const float snow_mass = cold_mul(qs, e.rho);
    const float graupel_mass = cold_mul(qg, e.rho);
    const int rain_mass_bin = cold_decade_index(rain_mass, -6, 37);
    const float am_r = cold_div(cold_mul(3.1415926536f, 1000.0f), 6.0f);
    const float rain_intercept_prefactor = cold_div(
        cold_mul(1.0f / 6.0f, rain_mass), am_r);
    ulong rain_intercept = cold_dmul(cold_widen(rain_intercept_prefactor),
                                     ice.rain_lambda);
    rain_intercept = cold_dmul(rain_intercept, ice.rain_lambda);
    rain_intercept = cold_dmul(rain_intercept, ice.rain_lambda);
    rain_intercept = cold_dmul(rain_intercept, ice.rain_lambda);
    const int rain_intercept_bin = cold_decade_index64(rain_intercept, 6, 37);

    if (rain_mass > 1.0e-6f) {
        const int temp_bin = max(0, min(int(round(-e.tempc)) - 1, 44));
        const ulong table_idx = ulong(rain_mass_bin + 37 *
            (rain_intercept_bin + 37 * (temp_bin + 45 * nuclei_bin)));
        r.freeze_ice_rate = cold_dmul(
            rain_to_ice_mass[table_idx], e.inverse_dt_d);
        r.freeze_ice_number_rate = cold_dmul(
            rain_to_ice_number[table_idx], e.inverse_dt_d);
        r.freeze_graupel_rate = cold_dmul(
            rain_to_graupel_mass[table_idx], e.inverse_dt_d);
        const ulong number = cold_dmin(cold_widen(ice.rain_number),
                                        rain_to_graupel_number[table_idx]);
        r.freeze_graupel_number_rate = cold_dmul(number, e.inverse_dt_d);
        r.freeze_rain_number_rate = r.freeze_graupel_number_rate;
    } else if (rain_mass > 1.0e-12f && e.temp0 < 235.16f) {
        r.freeze_ice_rate = cold_widen(cold_mul(rain_mass, e.inverse_dt));
        r.freeze_ice_number_rate = cold_widen(
            cold_mul(ice.rain_number, e.inverse_dt));
        r.freeze_rain_number_rate = r.freeze_ice_number_rate;
    }

    const int frozen_table_size = wrf39 ? 28 : 37;
    const int frozen_first_exponent = wrf39 ? -5 : -6;
    const float frozen_first_mass = wrf39 ? 1.0e-5f : 1.0e-6f;
    if (rain_mass >= 1.0e-6f && snow_mass >= frozen_first_mass) {
        const int snow_bin = cold_decade_index(
            snow_mass, frozen_first_exponent, frozen_table_size);
        const int raw_temp_bin = int(cold_div(cold_sub(e.tempc, 2.5f), 5.0f)) - 1;
        const int temp_bin = min(9, max(1, -raw_temp_bin)) - 1;
        const ulong table_idx = ulong(snow_bin + frozen_table_size * (temp_bin + 9 *
            (rain_intercept_bin + 37 * rain_mass_bin)));
        const ulong tmr2 = tmr_racs2[table_idx];
        const ulong tcr2 = tcr_sacr2[table_idx];
        const ulong tmr1 = tmr_racs1[table_idx];
        const ulong tcr1 = tcr_sacr1[table_idx];
        r.rain_snow_rain_rate = cold_dsub(0ul, cold_dadd(
            cold_dadd(tmr2, tcr2), cold_dadd(tmr1, tcr1)));
        r.rain_snow_category_rate = cold_dsub(
            cold_dadd(tmr2, tcr2),
            cold_dadd(tcs_racs1[table_idx], tms_sacr1[table_idx]));
        r.rain_snow_graupel_rate = cold_dadd(
            cold_dadd(tmr1, tcr1),
            cold_dadd(tcs_racs1[table_idx], tms_sacr1[table_idx]));
        r.rain_snow_number_rate = cold_dadd(
            cold_dadd(tnr_racs1[table_idx], tnr_racs2[table_idx]),
            cold_dadd(tnr_sacr1[table_idx], tnr_sacr2[table_idx]));
        r.rain_snow_rain_rate = cold_dmax(
            cold_widen(-cold_mul(rain_mass, e.inverse_dt)),
            r.rain_snow_rain_rate);
        r.rain_snow_category_rate = cold_dmax(
            cold_widen(-cold_mul(snow_mass, e.inverse_dt)),
            r.rain_snow_category_rate);
        r.rain_snow_graupel_rate = cold_dmin(
            cold_widen(cold_mul(cold_add(rain_mass, snow_mass), e.inverse_dt)),
            r.rain_snow_graupel_rate);
        r.rain_snow_number_rate = cold_dmin(
            cold_widen(cold_mul(ice.rain_number, e.inverse_dt)),
            r.rain_snow_number_rate);
    }

    if (rain_mass >= 1.0e-6f && graupel_mass >= frozen_first_mass) {
        const int graupel_mass_bin = cold_decade_index(
            graupel_mass, frozen_first_exponent, frozen_table_size);
        ulong graupel_intercept;
        int graupel_intercept_bin;
        ulong table_idx;
        ulong table_size;
        if (wrf39) {
            graupel_intercept = cold_wrf39_table_intercept(
                graupel_mass, graupel_inverse_lambda);
            graupel_intercept_bin = cold_decade_index64(
                graupel_intercept, 4, frozen_table_size);
            table_idx = ulong(graupel_intercept_bin + frozen_table_size *
                (graupel_mass_bin + frozen_table_size *
                (rain_intercept_bin + 37 * rain_mass_bin)));
            table_size = ulong(28 * 28 * 37 * 37);
        } else {
            float intercept_power = cold_fmax(2.0f, cold_fmin(6.0f,
                cold_add(3.0f, cold_mul(2.0f / 7.0f,
                    cold_add(cold_log10f(cold_fmax(1.0e-9f, graupel_mass)), 8.0f)))));
            graupel_intercept = cold_widen(cold_powf(10.0f,
                                                     intercept_power));
            graupel_intercept_bin = cold_decade_index64(
                graupel_intercept, 2, 37);
            const ulong nominal_idx = ulong(graupel_intercept_bin + 37 *
                (graupel_mass_bin + 37 * (rain_intercept_bin
                + 37 * rain_mass_bin)));
            table_idx = nominal_idx + ulong(4 * 37 * 37);
            table_size = ulong(37 * 37 * 37 * 37);
        }
        if (table_idx < table_size) {
            r.rain_graupel_graupel_rate = cold_dadd(
                tmr_racg[table_idx], tcr_gacr[table_idx]);
            r.rain_graupel_graupel_rate = cold_dmin(
                cold_widen(cold_mul(rain_mass, e.inverse_dt)),
                r.rain_graupel_graupel_rate);
            r.rain_graupel_rain_rate =
                r.rain_graupel_graupel_rate ^ 0x8000000000000000ul;
            r.rain_graupel_number_rate = cold_dadd(
                tnr_racg[table_idx], tnr_gacr[table_idx]);
            r.rain_graupel_number_rate = cold_dmin(
                cold_widen(cold_mul(ice.rain_number, e.inverse_dt)),
                r.rain_graupel_number_rate);
        }
    }
    (void)tcg_racg;
    return r;
}

struct ColdCloudRates {
    ulong autoconversion, autoconversion_number, rain_accretion;
    ulong freezing, freezing_number;
    ulong snow_riming, graupel_riming, hm_number, hm_mass;
    ulong snow_hm, graupel_hm;
    ulong cloud_number_autoconversion, cloud_number_rain;
    ulong cloud_number_snow, cloud_number_graupel;
    ulong nwfa_rain, nifa_rain, nwfa_snow, nifa_snow;
    ulong nwfa_graupel, nifa_graupel;
};

// Translation of the cloud-water/number and aerosol source group
// (:2157-2234, :2179-2222, :2402-2443, :2607-2617, :2444-2478).
inline ColdCloudRates cold_cloud_processes(
        thread const ColdEntryState &e, thread const ColdIceRates &ice,
        float snow_mass, float snow_diameter, float smoe, bool has_snow,
        float graupel_mass, float graupel_diameter,
        ulong graupel_ilam, ulong graupel_intercept,
        int nuclei_bin,
        device const ulong *rain_cloud_efficiency,
        device const ulong *cloud_to_ice_mass,
        device const ulong *cloud_to_ice_number, bool wrf39) {
    ColdCloudRates r;
    r.autoconversion = r.autoconversion_number = r.rain_accretion = 0ul;
    r.freezing = r.freezing_number = 0ul;
    r.snow_riming = r.graupel_riming = r.hm_number = r.hm_mass = 0ul;
    r.snow_hm = r.graupel_hm = 0ul;
    r.cloud_number_autoconversion = r.cloud_number_rain = 0ul;
    r.cloud_number_snow = r.cloud_number_graupel = 0ul;
    r.nwfa_rain = r.nifa_rain = r.nwfa_snow = r.nifa_snow = 0ul;
    r.nwfa_graupel = r.nifa_graupel = 0ul;
    const float am_r = cold_div(cold_mul(3.1415926536f, 1000.0f), 6.0f);
    const float t1_qs_qc = cold_mul(cold_mul(3.1415926536f, 0.25f), 40.0f);
    const float t1_qg_qc = cold_mul(cold_mul(cold_mul(
        3.1415926536f, 0.25f), 442.0f), 5.23476267f);
    const float cloud_mass = e.cloud_mass;

    if (cloud_mass > 0.01e-3f && e.has_cloud) {
        const float xdc = cold_fmax(1.0f, cold_mul(cold_powf_version(
            cold_div(cloud_mass, cold_mul(am_r, e.nc_work)),
            0.3333333433f, wrf39), 1.0e6f));
        const float dcg_power = cold_powf_version(
            cold_mul(cold_ccg3[e.nu_c], cold_ocg2[e.nu_c]),
            0.3333333433f, wrf39);
        const ulong dcg_double = cold_dmul(
            cold_ddiv(cold_widen(dcg_power), e.cloud_lambda),
            cold_widen(1.0e6f));
        const float dcg = cold_narrow(dcg_double);
        const float xdc3 = cold_mul(cold_mul(xdc, xdc), xdc);
        const float first = cold_mul(cold_mul(cold_mul(xdc3, dcg), dcg), dcg);
        const float second = cold_mul(cold_mul(cold_mul(xdc3, xdc), xdc), xdc);
        const float dcb_arg = cold_sub(first, second);
        const float dcb = cold_powf_version(
            cold_fmax(0.0f, dcb_arg), 0.1666666716f, wrf39);
        float zeta_term = cold_mul(6.25e-6f, xdc);
        zeta_term = cold_mul(zeta_term, dcb);
        zeta_term = cold_mul(zeta_term, dcb);
        zeta_term = cold_mul(zeta_term, dcb);
        zeta_term = cold_sub(zeta_term, 0.4f);
        const float zeta1 = cold_mul(0.5f,
            cold_add(zeta_term, fabs(zeta_term)));
        const float zeta = cold_mul(cold_mul(0.027f, cloud_mass), zeta1);
        const float tau_diameter = cold_sub(cold_mul(0.5f, dcb), 7.5f);
        const float taud = cold_add(cold_mul(0.5f,
            cold_add(tau_diameter, fabs(tau_diameter))), 1.0e-12f);
        const float tau = cold_div(3.72f, cold_mul(cloud_mass, taud));
        r.autoconversion = cold_dmin(
            cold_widen(cold_mul(cloud_mass, e.inverse_dt)),
            cold_widen(cold_div(zeta, tau)));
        float number_denom = cold_mul(am_r, float(e.nu_c));
        if (!wrf39) number_denom = cold_mul(number_denom, 10.0f);
        number_denom = cold_mul(number_denom, 50.0e-6f);
        number_denom = cold_mul(number_denom, 50.0e-6f);
        number_denom = cold_mul(number_denom, 50.0e-6f);
        r.autoconversion_number = cold_ddiv(r.autoconversion,
                                             cold_widen(number_denom));
        float mvd_cube = cold_mul(am_r, e.cloud_mvd);
        mvd_cube = cold_mul(mvd_cube, e.cloud_mvd);
        mvd_cube = cold_mul(mvd_cube, e.cloud_mvd);
        r.cloud_number_autoconversion = cold_dmin(
            cold_widen(cold_mul(e.nc_work, e.inverse_dt)),
            cold_ddiv(r.autoconversion, cold_widen(mvd_cube)));
    }

    if (ice.rain_active && ice.rain_mvd > 50.0e-6f) {
        const float coefficient = cold_mul(cold_mul(cold_mul(
            3.1415926536f, 0.25f), 4854.0f), 6.0f);
        const ulong tail = cold_dmul(ice.rain_intercept_n0,
            cold_dpow(cold_dadd(ice.rain_lambda, cold_widen(195.0f)),
                      soft64_from_i32(-4)));
        const float prefactor = cold_mul(e.density_factor, coefficient);
        if (e.cloud_mvd > 1.0e-6f) {
            const ulong dr_first = 0x3f0ad333f9742941ul;
            const ulong dr_last = 0x3f740389985ae9dbul;
            const ulong rain_ratio = cold_ddiv(cold_widen(ice.rain_mvd), dr_first);
            const ulong dr_ratio = cold_ddiv(dr_last, dr_first);
            ulong rain_bin_value = cold_ddiv(
                cold_dmul(cold_widen(100.0f), cold_dlog(rain_ratio)),
                cold_dlog(dr_ratio));
            const Soft64I32Result rain_bin_i = soft64_to_i32_rz(rain_bin_value);
            const int rain_bin = min(1 + rain_bin_i.value, 100);
            const int cloud_bin = min(100,
                int(cold_mul(e.cloud_mvd, 1.0e6f)));
            const float efficiency = cold_narrow(
                rain_cloud_efficiency[(rain_bin - 1) + 100 * (cloud_bin - 1)]);
            const float cloud_prefactor = cold_mul(cold_mul(prefactor,
                efficiency), cloud_mass);
            r.rain_accretion = cold_dmin(
                cold_widen(cold_mul(cloud_mass, e.inverse_dt)),
                cold_dmul(cold_widen(cloud_prefactor), tail));
            const float number_prefactor = cold_mul(cold_mul(prefactor,
                efficiency), e.nc_work);
            r.cloud_number_rain = cold_dmin(
                cold_dmul(cold_widen(e.nc_work), e.inverse_dt_d),
                cold_dmul(cold_widen(number_prefactor), tail));
        }
        const float ef_ccn = cold_eff_aero(ice.rain_mvd, 0.04e-6f,
            e.viscosity, e.rho, e.temp0, 0, wrf39);
        const float ccn_prefactor = cold_mul(cold_mul(cold_mul(
            prefactor, ef_ccn), e.nwfa_work), 1.0f);
        r.nwfa_rain = cold_dmin(
            cold_dmul(cold_widen(e.nwfa_work), e.inverse_dt_d),
            cold_dmul(cold_widen(ccn_prefactor), tail));
        const float ef_in = cold_eff_aero(ice.rain_mvd, 0.8e-6f,
            e.viscosity, e.rho, e.temp0, 0, wrf39);
        const float in_prefactor = cold_mul(cold_mul(prefactor, ef_in),
                                            e.nifa_work);
        r.nifa_rain = cold_dmin(
            cold_dmul(cold_widen(e.nifa_work), e.inverse_dt_d),
            cold_dmul(cold_widen(in_prefactor), tail));
    }

    const int cloud_number_bin = cold_droplet_bin(e.nc_work);
    if (cloud_mass > 1.0e-6f) {
        const int mass_bin = cold_decade_index(cloud_mass, -6, 37);
        const int temp_bin = max(0, min(int(round(-e.tempc)) - 1, 44));
        const ulong table_idx = ulong(mass_bin + 37 *
            (cloud_number_bin + 100 * (temp_bin + 45 * nuclei_bin)));
        r.freezing = cold_dmin(cold_widen(cold_mul(cloud_mass, e.inverse_dt)),
            cold_dmul(cloud_to_ice_mass[table_idx], e.inverse_dt_d));
        const ulong number_cap = cold_widen(cold_mul(e.nc_work, e.inverse_dt));
        const ulong number_from_mass = cold_ddiv(r.freezing,
                                                  0x3d819799812dea11ul);
        const ulong number_from_table = cold_dmul(
            cloud_to_ice_number[table_idx], e.inverse_dt_d);
        r.freezing_number = cold_dmin(number_cap,
            cold_dmin(number_from_mass, number_from_table));
    } else if (e.has_cloud && cloud_mass > 1.0e-12f && e.temp0 < 235.16f) {
        r.freezing = cold_widen(cold_mul(cloud_mass, e.inverse_dt));
        r.freezing_number = cold_widen(cold_mul(e.nc_work, e.inverse_dt));
    }

    const float snow_d0_f = wrf39 ? 200.0e-6f : 300.0e-6f;
    const ulong snow_d0 = cold_widen(snow_d0_f);
    if (e.has_cloud && has_snow && e.cloud_mvd > 1.0e-6f
            && snow_diameter > snow_d0_f) {
        const ulong log_ratio = cold_dlog(cold_ddiv(
            cold_widen(0.02f), snow_d0));
        const ulong first_snow_bin = cold_dmul(snow_d0,
            cold_dexp(cold_dmul(0x3f747ae147ae147bul, log_ratio)));
        const ulong last_snow_bin = cold_dmul(snow_d0,
            cold_dexp(cold_dmul(0x3fefd70a3d70a3d7ul, log_ratio)));
        const ulong bin_ratio = cold_ddiv(cold_widen(snow_diameter),
                                           first_snow_bin);
        const ulong snow_bin_raw = cold_ddiv(
            cold_dmul(cold_widen(100.0f), cold_dlog(bin_ratio)),
            cold_dlog(cold_ddiv(last_snow_bin, first_snow_bin)));
        const Soft64I32Result snow_bin_i = soft64_to_i32_rz(snow_bin_raw);
        const int snow_bin = min(1 + snow_bin_i.value, 100);
        const ulong snow_bin_center = cold_dmul(
            cold_ddiv(cold_dsub(soft64_from_i32(snow_bin),
                                0x3fe0000000000000ul),
                     soft64_from_i32(100)),
            log_ratio);
        const ulong table_snow_diameter = cold_dmul(
            snow_d0, cold_dexp(snow_bin_center));
        const int cloud_bin = min(100, int(cold_mul(e.cloud_mvd, 1.0e6f)));
        ulong table_cloud_diameter = cold_widen(1.0e-6f);
        for (int n = 2; n <= cloud_bin; ++n)
            table_cloud_diameter = cold_dadd(table_cloud_diameter,
                                               cold_widen(1.0e-6f));
        ulong cloud_velocity = cold_dmul(cold_widen(1.19e4f),
            cold_dmul(cold_dmul(cold_dmul(cold_widen(1.0e4f),
                table_cloud_diameter), table_cloud_diameter), cold_widen(0.25f)));
        ulong snow_velocity = cold_dmul(cold_widen(40.0f),
            cold_dmul(cold_dpow(table_snow_diameter, 0x3fe199999999999aul),
                cold_dexp(cold_dmul(cold_widen(-100.0f),
                                    table_snow_diameter))));
        snow_velocity = cold_dsub(snow_velocity, cloud_velocity);
        const ulong melt_numerator = cold_dmul(cold_dmul(
            cold_widen(0.069f), table_snow_diameter), table_snow_diameter);
        const ulong melt_denominator = cold_ddiv(cold_dmul(
            cold_widen(3.1415926536f), cold_widen(1000.0f)),
            cold_widen(6.0f));
        const ulong melted_snow_diameter = cold_dpow(
            cold_ddiv(melt_numerator, melt_denominator),
            0x3fd5555555555555ul);
        const ulong diameter_fraction = cold_ddiv(table_cloud_diameter,
                                                   melted_snow_diameter);
        float efficiency = 0.0f;
        if (!soft64_greater(diameter_fraction, cold_widen(0.25f))
                && !soft64_less(table_cloud_diameter, cold_widen(6.0e-6f))
                && !soft64_less(snow_velocity, cold_widen(1.0e-3f))) {
            ulong stokes = cold_dmul(table_cloud_diameter, table_cloud_diameter);
            stokes = cold_dmul(stokes, snow_velocity);
            stokes = cold_dmul(stokes, cold_widen(1000.0f));
            stokes = cold_ddiv(stokes, cold_dmul(cold_dmul(
                cold_widen(9.0f), cold_widen(1.718e-5f)),
                melted_snow_diameter));
            ulong reynolds = cold_dmul(cold_widen(9.0f), stokes);
            reynolds = cold_ddiv(reynolds, cold_dmul(cold_dmul(
                diameter_fraction, diameter_fraction), cold_widen(1000.0f)));
            const ulong log_reynolds = cold_dlog(reynolds);
            ulong k0_exp = cold_dsub(cold_widen(-0.1007f),
                cold_dmul(cold_widen(0.358f), log_reynolds));
            k0_exp = cold_dadd(k0_exp, cold_dmul(cold_widen(0.0261f),
                cold_dmul(log_reynolds, log_reynolds)));
            const ulong k0 = cold_dexp(k0_exp);
            const ulong z = cold_dlog(cold_ddiv(stokes,
                cold_dadd(k0, cold_widen(1.0e-15f))));
            ulong h = cold_dadd(cold_widen(0.1465f),
                                 cold_dmul(cold_widen(1.302f), z));
            h = cold_dsub(h, cold_dmul(cold_widen(0.607f), cold_dmul(z, z)));
            h = cold_dadd(h, cold_dmul(cold_widen(0.293f),
                                       cold_dmul(cold_dmul(z, z), z)));
            const ulong yc0 = cold_dmul(cold_ddiv(cold_widen(2.0f),
                cold_widen(3.14159265358979323846f)), cold_datan(h));
            const ulong numerator = cold_dadd(yc0, diameter_fraction);
            const ulong denominator = cold_dadd(
                0x3ff0000000000000ul, diameter_fraction);
            const ulong value = cold_ddiv(cold_dmul(numerator, numerator),
                                           cold_dmul(denominator, denominator));
            efficiency = cold_fmax(0.0f,
                cold_fmin(cold_narrow(value), 0.95f));
        }
        float rate = cold_mul(e.density_factor, t1_qs_qc);
        rate = cold_mul(rate, efficiency);
        rate = cold_mul(rate, cloud_mass);
        rate = cold_mul(rate, smoe);
        r.snow_riming = cold_dmin(cold_widen(rate),
            cold_widen(cold_mul(cloud_mass, e.inverse_dt)));
        float number_rate = cold_mul(e.density_factor, t1_qs_qc);
        number_rate = cold_mul(number_rate, efficiency);
        number_rate = cold_mul(number_rate, e.nc_work);
        number_rate = cold_mul(number_rate, smoe);
        r.cloud_number_snow = cold_dmin(cold_widen(number_rate),
            cold_widen(cold_mul(e.nc_work, e.inverse_dt)));
    }

    if (e.has_cloud
            && graupel_mass >= (wrf39 ? 1.0e-5f : 1.0e-6f)
            && e.cloud_mvd > 1.0e-6f
            && (!wrf39 || graupel_diameter > 250.0e-6f)) {
        const ulong velocity_d = cold_dmul(cold_widen(cold_mul(
            cold_mul(cold_mul(cold_mul(e.density_factor, 442.0f),
                              20.3632278f), 1.0f / 6.0f), 1.0f)),
            cold_dpow(graupel_ilam, 0x3fec7ae147ae147bul));
        const float velocity = cold_narrow(velocity_d);
        const float stokes = cold_div(cold_mul(cold_mul(cold_mul(
            cold_mul(e.cloud_mvd, e.cloud_mvd), velocity), 1000.0f), 1.0f),
            cold_mul(cold_mul(cold_mul(9.0f, e.viscosity),
                              graupel_diameter), 1.0f));
        float efficiency = 0.0f;
        if (stokes >= 0.4f && stokes <= 10.0f)
            efficiency = cold_mul(0.55f,
                cold_log10f(cold_mul(2.51f, stokes)));
        else if (stokes > 10.0f) efficiency = 0.77f;
        float mass_coeff = cold_mul(e.density_factor, t1_qg_qc);
        mass_coeff = cold_mul(mass_coeff, efficiency);
        mass_coeff = cold_mul(mass_coeff, cloud_mass);
        r.graupel_riming = cold_dmul(cold_widen(mass_coeff),
            cold_dmul(graupel_intercept,
                cold_dpow(graupel_ilam, 0x400f1eb851eb851ful)));
        float number_coeff = cold_mul(e.density_factor, t1_qg_qc);
        number_coeff = cold_mul(number_coeff, efficiency);
        number_coeff = cold_mul(number_coeff, e.nc_work);
        r.cloud_number_graupel = cold_dmin(
            cold_dmul(cold_widen(number_coeff), cold_dmul(graupel_intercept,
                cold_dpow(graupel_ilam, 0x400f1eb851eb851ful))),
            cold_dmul(cold_widen(e.nc_work), e.inverse_dt_d));
    }
    if (soft64_greater(r.graupel_riming, cold_widen(1.0e-15f))
            && e.tempc > -8.0f) {
        float factor = 0.0f;
        if (e.tempc >= -5.0f && e.tempc < -3.0f)
            factor = cold_mul(0.5f, cold_sub(-3.0f, e.tempc));
        else if (e.tempc > -8.0f && e.tempc < -5.0f)
            factor = cold_mul(0.33333333f, cold_add(8.0f, e.tempc));
        r.hm_number = cold_dmul(0x41b4dc9380000000ul,
            cold_dmul(cold_widen(factor), r.graupel_riming));
        r.hm_mass = cold_dmul(0x3d719799812dea11ul, r.hm_number);
        const ulong total_riming = cold_dadd(r.snow_riming,
                                             r.graupel_riming);
        if (soft64_greater(total_riming, 0ul)) {
            r.snow_hm = cold_ddiv(cold_dmul(r.snow_riming, r.hm_mass),
                                  total_riming);
            r.graupel_hm = cold_ddiv(cold_dmul(r.graupel_riming, r.hm_mass),
                                     total_riming);
        }
    }

    const float frozen_first = wrf39 ? 1.0e-5f : 1.0e-6f;
    if (snow_mass > frozen_first) {
        const float ef_ccn = cold_eff_aero(snow_diameter, 0.04e-6f,
            e.viscosity, e.rho, e.temp0, 1, wrf39);
        float coeff = cold_mul(e.density_factor, t1_qs_qc);
        coeff = cold_mul(coeff, ef_ccn); coeff = cold_mul(coeff, e.nwfa_work);
        coeff = cold_mul(coeff, smoe);
        r.nwfa_snow = cold_dmin(cold_widen(coeff),
            cold_dmul(cold_widen(e.nwfa_work), e.inverse_dt_d));
        const float ef_in = cold_eff_aero(snow_diameter, 0.8e-6f,
            e.viscosity, e.rho, e.temp0, 1, wrf39);
        coeff = cold_mul(e.density_factor, t1_qs_qc);
        coeff = cold_mul(coeff, ef_in); coeff = cold_mul(coeff, e.nifa_work);
        coeff = cold_mul(coeff, smoe);
        r.nifa_snow = cold_dmin(cold_widen(coeff),
            cold_dmul(cold_widen(e.nifa_work), e.inverse_dt_d));
    }
    if (graupel_mass > frozen_first) {
        const float ef_ccn = cold_eff_aero(graupel_diameter, 0.04e-6f,
            e.viscosity, e.rho, e.temp0, 2, wrf39);
        float coeff = cold_mul(e.density_factor, t1_qg_qc);
        coeff = cold_mul(coeff, ef_ccn); coeff = cold_mul(coeff, e.nwfa_work);
        r.nwfa_graupel = cold_dmin(cold_dmul(cold_widen(coeff),
            cold_dmul(graupel_intercept,
                cold_dpow(graupel_ilam, 0x400f1eb851eb851ful))),
            cold_dmul(cold_widen(e.nwfa_work), e.inverse_dt_d));
        const float ef_in = cold_eff_aero(graupel_diameter, 0.8e-6f,
            e.viscosity, e.rho, e.temp0, 2, wrf39);
        coeff = cold_mul(e.density_factor, t1_qg_qc);
        coeff = cold_mul(coeff, ef_in); coeff = cold_mul(coeff, e.nifa_work);
        r.nifa_graupel = cold_dmin(cold_dmul(cold_widen(coeff),
            cold_dmul(graupel_intercept,
                cold_dpow(graupel_ilam, 0x400f1eb851eb851ful))),
            cold_dmul(cold_widen(e.nifa_work), e.inverse_dt_d));
    }
    return r;
}

struct ColdNucleationRates {
    ulong demott, demott_number, koop, koop_number;
};

// Production deposition and homogeneous haze nucleation (:2619-2643).
// WRF subtracts rain and cloud freezing crystals before deciding how many
// new DeMott crystals are needed, then includes those crystals in the Koop
// gate. Keep the float round trips around dt explicit.
inline ColdNucleationRates cold_nucleation_processes(
        thread const ColdEntryState &e, thread const ColdIceRates &ice,
        thread const ColdRainRates &rain, thread const ColdCloudRates &cloud,
        float snow_number_ns, float snow_smo0, bool wrf39) {
    ColdNucleationRates r;
    r.demott = r.demott_number = r.koop = r.koop_number = 0ul;
    const float target = cold_ice_demott(e.tempc, e.rho, e.nifa_work,
                                         wrf39);
    if (e.nucleation_active) {
        ulong added_numbers = cold_dadd(rain.freeze_ice_number_rate,
                                         cloud.freezing_number);
        const float existing = cold_add(ice.ice_number, cold_narrow(
            cold_dmul(added_numbers, cold_widen(e.dt))));
        const float delta = cold_sub(target, existing);
        float positive = cold_add(delta, fabs(delta));
        positive = cold_mul(0.5f, positive);
        const float number_rate = cold_mul(positive, e.inverse_dt);
        r.demott_number = cold_widen(number_rate);
        r.demott = cold_dmin(cold_widen(e.vapor_limit),
            cold_dmul(0x3d719799812dea11ul, r.demott_number));
        r.demott_number = cold_ddiv(r.demott, 0x3d719799812dea11ul);
    }
    const float all_existing = cold_add(
        wrf39 ? snow_smo0 : snow_number_ns, ice.ice_number);
    const ulong added_numbers = cold_dadd(rain.freeze_ice_number_rate,
        cold_dadd(cloud.freezing_number, r.demott_number));
    const float xni_test = cold_add(all_existing,
        cold_narrow(cold_dmul(added_numbers, cold_widen(e.dt))));
    const float koop_gate = wrf39 ? 500.0e3f : 999.0e3f;
    if (xni_test <= koop_gate && e.temp0 < 238.0f && e.ssati >= 0.4f) {
        const float xnc = cold_ice_koop(e.temp0, e.qv0, e.qvs,
                                        e.nwfa_work, e.dt, wrf39);
        const float number_rate = cold_mul(xnc, e.inverse_dt);
        r.koop_number = cold_widen(number_rate);
        r.koop = cold_dmin(cold_widen(e.vapor_limit),
            cold_dmul(0x3d3c25c268497682ul, r.koop_number));
        r.koop_number = cold_ddiv(r.koop, 0x3d3c25c268497682ul);
    }
    return r;
}

struct ColdFrozenVaporRates {
    ulong snow, snow_collection, snow_collection_number;
    ulong graupel;
    ulong snow_to_graupel, snow_to_graupel_number;
    ulong snow_riming;
    float graupel_mvd, graupel_lambda, graupel_number, graupel_intercept;
};

// Deposition/sublimation and deposition-conditioned rimed-snow conversion
// (:2688-2777). Inputs smob/smoc/smoe/smo0 are the diagnosed snow moments.
inline ColdFrozenVaporRates cold_frozen_vapor_processes(
        thread const ColdEntryState &e, thread const ColdIceRates &ice,
        float snow_mass, float snow_diameter, float snow_tc0,
        float qi, float ice_mass, float smob, float smoc,
        float smoe, float smo0, bool has_snow, bool has_graupel,
        float graupel_mass, float graupel_diameter, ulong graupel_ilam,
        ulong graupel_intercept, float cloud_mvd,
        ulong snow_riming_rate, bool wrf39,
        thread float &snow_velocity_boost) {
    ColdFrozenVaporRates r;
    r.snow = r.snow_collection = r.snow_collection_number = 0ul;
    r.graupel = r.snow_to_graupel = r.snow_to_graupel_number = 0ul;
    r.snow_riming = snow_riming_rate;
    r.graupel_mvd = graupel_diameter;
    r.graupel_lambda = cold_narrow(graupel_ilam);
    r.graupel_number = cold_narrow(graupel_intercept);
    r.graupel_intercept = cold_narrow(graupel_intercept);
    const float saturated_density = cold_mul(e.rho, e.qvsi);
    if (has_snow) {
        const float rho_factor_sqrt = cold_sqrtf(e.density_factor);
        const float viscosity_factor = cold_sqrtf(cold_div(e.rho, e.viscosity));
        const float first_moment = cold_mul(cold_field_a(snow_tc0, 1.0f, wrf39),
            cold_powf_version(smob, cold_field_b(snow_tc0, 1.0f), wrf39));
        const float deposition_moment = 1.0f + (1.0f + 0.55f) * 0.5f;
        const float ventilation_moment = cold_mul(
            cold_field_a(snow_tc0, deposition_moment, wrf39),
            cold_powf_version(smob,
                cold_field_b(snow_tc0, deposition_moment), wrf39));
        const float capacitance = cold_fmax(0.15f, cold_fmin(0.5f,
            cold_add(0.15f, cold_div(cold_mul(
                cold_add(e.tempc, 1.5f), 0.35f), -28.5f))));
        const float ventilation = cold_mul(cold_mul(0.28f,
            cold_powf(0.632f, 0.3333333433f)), cold_sqrtf(40.0f));
        float moment_sum = cold_mul(0.86f, first_moment);
        moment_sum = cold_add(moment_sum, cold_mul(cold_mul(
            cold_mul(ventilation, rho_factor_sqrt), viscosity_factor),
            ventilation_moment));
        float rate = cold_mul(capacitance, e.vapor_geometry);
        rate = cold_mul(rate, e.diffusivity);
        rate = cold_mul(rate, e.ssati);
        rate = cold_mul(rate, saturated_density);
        rate = cold_mul(rate, moment_sum);
        ulong snow_rate = cold_widen(rate);
        if (rate > 0.0f) snow_rate = cold_dmin(snow_rate,
                                               cold_widen(e.vapor_limit));
        else {
            snow_rate = cold_dmax(cold_widen(cold_mul(-snow_mass,
                e.inverse_dt)), snow_rate);
            snow_rate = cold_dmax(snow_rate, cold_widen(e.vapor_limit));
        }
        r.snow = snow_rate;
        if (qi > 1.0e-12f && snow_mass >= 1.0e-6f) {
            float collection = cold_mul(cold_mul(3.1415926536f, 0.25f), 40.0f);
            collection = cold_mul(collection, e.density_factor);
            collection = cold_mul(collection, 0.05f);
            collection = cold_mul(collection, ice_mass);
            collection = cold_mul(collection, smoe);
            r.snow_collection = cold_widen(collection);
            r.snow_collection_number = cold_ddiv(r.snow_collection,
                cold_widen(ice.ice_particle_mass));
        }
    }
    if (wrf39 && has_graupel && e.ssati < -1.0e-15f) {
        const float rho_factor_sqrt = cold_sqrtf(e.density_factor);
        const float viscosity_factor = cold_sqrtf(cold_div(e.rho, e.viscosity));
        float t2_qg_sd = cold_mul(0.28f,
            cold_powf(0.632f, 0.3333333433f));
        t2_qg_sd = cold_mul(t2_qg_sd, cold_sqrtf(442.0f));
        t2_qg_sd = cold_mul(t2_qg_sd, 1.9021706581115723f);
        float rate_prefactor = cold_mul(0.5f, e.vapor_geometry);
        rate_prefactor = cold_mul(rate_prefactor, e.diffusivity);
        rate_prefactor = cold_mul(rate_prefactor, e.ssati);
        rate_prefactor = cold_mul(rate_prefactor, saturated_density);
        ulong moment = cold_dmul(cold_widen(0.86f),
            cold_dpow(graupel_ilam, 0x4000000000000000ul));
        float ventilation_prefactor = cold_mul(t2_qg_sd, viscosity_factor);
        ventilation_prefactor = cold_mul(ventilation_prefactor,
                                         rho_factor_sqrt);
        moment = cold_dadd(moment, cold_dmul(
            cold_widen(ventilation_prefactor),
            cold_dpow(graupel_ilam,
                      cold_widen(2.9449999332427978516f))));
        ulong graupel_rate = cold_dmul(cold_widen(rate_prefactor),
            cold_dmul(graupel_intercept, moment));
        if (soft64_sign(graupel_rate)) {
            graupel_rate = cold_dmax(cold_dmax(
                cold_widen(cold_mul(-graupel_mass, e.inverse_dt)),
                graupel_rate), cold_widen(e.vapor_limit));
        } else {
            graupel_rate = cold_dmin(graupel_rate,
                                      cold_widen(e.vapor_limit));
        }
        r.graupel = graupel_rate;
    } else if (!wrf39 && has_graupel && e.ssati < -1.0e-15f) {
        const float am_g = cold_div(cold_mul(3.1415926536f, 400.0f), 6.0f);
        float intercept_power = cold_add(3.0f, cold_mul(2.0f / 7.0f,
            cold_add(cold_log10f(cold_fmax(1.0e-9f, graupel_mass)), 8.0f)));
        intercept_power = cold_fmax(2.0f, cold_fmin(6.0f, intercept_power));
        const float diagnosed_intercept = cold_powf(10.0f, intercept_power);
        float lambda = cold_powf(cold_div(cold_mul(cold_mul(
            diagnosed_intercept, am_g), 6.0f), graupel_mass), 0.25f);
        float number_per_kg = cold_div(cold_mul(cold_mul(1.0f / 6.0f,
            graupel_mass), cold_powf(lambda, 3.0f)), am_g);
        number_per_kg = cold_div(number_per_kg, e.rho);
        number_per_kg = cold_fmax(1.0e-6f, number_per_kg);
        float graupel_number = cold_fmax(1.0e-6f,
            cold_mul(number_per_kg, e.rho));
        lambda = cold_powf(cold_div(cold_mul(cold_mul(am_g, 6.0f),
            graupel_number), graupel_mass), 0.3333333433f);
        float mvd = cold_div(3.672f, lambda);
        if (mvd > 25.4e-3f) {
            mvd = 25.4e-3f;
            lambda = cold_div(3.672f, mvd);
            graupel_number = cold_div(cold_mul(cold_mul(1.0f / 6.0f,
                graupel_mass), cold_powf(lambda, 3.0f)), am_g);
        } else if (mvd < 50.0e-6f) {
            mvd = 50.0e-6f;
            lambda = cold_div(3.672f, mvd);
            graupel_number = cold_div(cold_mul(cold_mul(1.0f / 6.0f,
                graupel_mass), cold_powf(lambda, 3.0f)), am_g);
        }
        const float inverse_lambda = cold_div(1.0f, lambda);
        const float intercept = cold_mul(graupel_number, lambda);
        const float ventilation = cold_mul(cold_mul(cold_mul(0.28f,
            cold_powf(0.632f, 0.3333333433f)), cold_sqrtf(442.0f)),
            1.9021706581115723f);
        float moment = cold_mul(0.86f,
            cold_powf(inverse_lambda, 2.0f));
        float vent_moment = cold_mul(cold_mul(cold_mul(ventilation,
            cold_sqrtf(cold_div(e.rho, e.viscosity))),
            cold_sqrtf(e.density_factor)),
            cold_powf(inverse_lambda, 2.945f));
        moment = cold_mul(intercept, cold_add(moment, vent_moment));
        float rate = cold_mul(0.5f, e.vapor_geometry);
        rate = cold_mul(rate, e.diffusivity);
        rate = cold_mul(rate, e.ssati);
        rate = cold_mul(rate, saturated_density);
        rate = cold_mul(rate, moment);
        ulong graupel_rate = cold_widen(rate);
        graupel_rate = cold_dmax(cold_widen(cold_mul(-graupel_mass,
            e.inverse_dt)), graupel_rate);
        r.graupel = cold_dmax(cold_widen(e.vapor_limit), graupel_rate);
        r.graupel_mvd = mvd;
        r.graupel_lambda = lambda;
        r.graupel_number = graupel_number;
        r.graupel_intercept = intercept;
    }
    if (has_snow && snow_mass > 0.0f && soft64_greater(snow_riming_rate,
                cold_dmul(cold_widen(2.0f), r.snow))
            && soft64_greater(r.snow, cold_widen(1.0e-15f))) {
        const float ratio = cold_narrow(cold_dmin(cold_widen(30.0f),
            cold_ddiv(snow_riming_rate, r.snow)));
        float fraction = cold_fmin(0.95f,
            cold_add(0.15f, cold_mul(cold_sub(ratio, 2.0f), 0.028f)));
        snow_velocity_boost = cold_fmin(1.5f,
            cold_add(1.1f, cold_mul(cold_sub(ratio, 2.0f),
                                    wrf39 ? 0.016f : 0.014f)));
        r.snow_to_graupel = cold_dmul(cold_widen(fraction), snow_riming_rate);
        r.snow_to_graupel_number = cold_ddiv(cold_dmul(
            r.snow_to_graupel, cold_widen(smo0)), cold_widen(snow_mass));
        const float snow_velocity_exp_arg = cold_mul(-100.0f, snow_diameter);
        const float snow_velocity_exp = wrf39 ? exp(snow_velocity_exp_arg)
                                              : cold_expf(snow_velocity_exp_arg);
        const float snow_velocity = cold_mul(cold_mul(40.0f,
            cold_powf(snow_diameter, 0.55f)),
            snow_velocity_exp);
        float rime_parameter = cold_div(cold_mul(
            cold_mul(-cloud_mvd, 0.5e6f), snow_velocity),
            cold_fmin(-0.1f, e.tempc));
        rime_parameter = cold_fmax(0.1f, cold_fmin(rime_parameter, 10.0f));
        if (!wrf39) {
            float rime_density = cold_add(0.051f,
                cold_mul(0.114f, rime_parameter));
            rime_density = cold_sub(rime_density,
                cold_mul(cold_mul(0.0055f, rime_parameter), rime_parameter));
            rime_density = cold_mul(rime_density, 1000.0f);
            if (rime_density < 150.0f) {
                fraction = 0.0f;
                r.snow_to_graupel = 0ul;
                r.snow_to_graupel_number = 0ul;
            }
        }
        r.snow_riming = cold_dmul(
            cold_widen(cold_sub(1.0f, fraction)), snow_riming_rate);
    }
    (void)smoc; (void)graupel_diameter;
    (void)graupel_ilam; (void)graupel_intercept;
    return r;
}

// Apply the source's coupled mass conservation passes (:2862-2950).
// Number members are deliberately held where WRF rescales only mass.
inline ulong cold_limit_process_rates(
        thread const ColdEntryState &e, float ice_mass, float rain_mass,
        float snow_mass, float graupel_mass, bool has_ice,
        bool has_cloud, bool has_snow, bool has_graupel,
        thread ColdIceRates &ice, thread ColdRainRates &rain,
        thread ColdCloudRates &cloud, thread ColdNucleationRates &nucleation,
        thread ColdFrozenVaporRates &frozen) {
    ulong vapor_sum = cold_dadd(nucleation.demott, nucleation.koop);
    vapor_sum = cold_dadd(vapor_sum, ice.ice_rate);
    vapor_sum = cold_dadd(vapor_sum, ice.ice_to_snow_rate);
    vapor_sum = cold_dadd(vapor_sum, frozen.snow);
    vapor_sum = cold_dadd(vapor_sum, frozen.graupel);
    if ((soft64_greater(vapor_sum, 0x3cd203af9ee75616ul)
            && soft64_greater(vapor_sum, cold_widen(e.vapor_limit_sum)))
        || (soft64_less(vapor_sum, 0xbcd203af9ee75616ul)
            && soft64_less(vapor_sum, cold_widen(e.vapor_limit_sum)))) {
        const ulong ratio = cold_ddiv(cold_widen(e.vapor_limit_sum), vapor_sum);
        nucleation.demott = cold_dmul(nucleation.demott, ratio);
        ice.ice_rate = cold_dmul(ice.ice_rate, ratio);
        ice.ice_to_snow_rate = cold_dmul(ice.ice_to_snow_rate, ratio);
        ice.ice_number_rate = cold_dmul(ice.ice_number_rate, ratio);
        frozen.snow = cold_dmul(frozen.snow, ratio);
        frozen.graupel = cold_dmul(frozen.graupel, ratio);
        nucleation.koop = cold_dmul(nucleation.koop, ratio);
        vapor_sum = cold_widen(e.vapor_limit_sum);
    }
    const ulong cloud_limit = cold_widen(cold_mul(-e.cloud_mass,
                                                    e.inverse_dt));
    ulong cloud_sum = cloud.autoconversion ^ 0x8000000000000000ul;
    cloud_sum = cold_dsub(cloud_sum, cloud.freezing);
    cloud_sum = cold_dsub(cloud_sum, cloud.rain_accretion);
    cloud_sum = cold_dsub(cloud_sum, cloud.snow_riming);
    cloud_sum = cold_dsub(cloud_sum, frozen.snow_to_graupel);
    cloud_sum = cold_dsub(cloud_sum, cloud.graupel_riming);
    if (has_cloud && soft64_less(cloud_sum, cloud_limit)) {
        const ulong ratio = cold_ddiv(cloud_limit, cloud_sum);
        cloud.autoconversion = cold_dmul(cloud.autoconversion, ratio);
        cloud.freezing = cold_dmul(cloud.freezing, ratio);
        cloud.rain_accretion = cold_dmul(cloud.rain_accretion, ratio);
        cloud.snow_riming = cold_dmul(cloud.snow_riming, ratio);
        frozen.snow_to_graupel = cold_dmul(frozen.snow_to_graupel, ratio);
        cloud.graupel_riming = cold_dmul(cloud.graupel_riming, ratio);
    }
    const ulong ice_limit = cold_widen(cold_mul(-ice_mass, e.inverse_dt));
    ulong ice_sum = ice.ice_rate;
    ice_sum = cold_dsub(ice_sum, cloud.autoconversion);
    ice_sum = cold_dsub(ice_sum, frozen.snow_collection);
    ice_sum = cold_dsub(ice_sum, ice.rain_ice_ice_rate);
    if (has_ice && soft64_less(ice_sum, 0xbcd203af9ee75616ul)
            && soft64_less(ice_sum, ice_limit)) {
        const ulong ratio = cold_ddiv(ice_limit, ice_sum);
        const ulong old_rate = ice.ice_rate;
        ice.ice_rate = cold_dmul(ice.ice_rate, ratio);
        cloud.autoconversion = cold_dmul(cloud.autoconversion, ratio);
        frozen.snow_collection = cold_dmul(frozen.snow_collection, ratio);
        ice.rain_ice_ice_rate = cold_dmul(ice.rain_ice_ice_rate, ratio);
        vapor_sum = cold_dadd(vapor_sum, cold_dsub(
            ice.ice_rate, old_rate));
    }
    const ulong rain_limit = cold_widen(cold_mul(-rain_mass, e.inverse_dt));
    ulong rain_sum = rain.freeze_graupel_rate ^ 0x8000000000000000ul;
    rain_sum = cold_dsub(rain_sum, rain.freeze_ice_rate);
    rain_sum = cold_dsub(rain_sum, ice.rain_ice_rain_rate);
    rain_sum = cold_dadd(rain_sum, rain.rain_snow_rain_rate);
    rain_sum = cold_dadd(rain_sum, rain.rain_graupel_rain_rate);
    if (ice.rain_active && soft64_less(rain_sum, rain_limit)) {
        const ulong ratio = cold_ddiv(rain_limit, rain_sum);
        rain.freeze_graupel_rate = cold_dmul(rain.freeze_graupel_rate, ratio);
        rain.freeze_ice_rate = cold_dmul(rain.freeze_ice_rate, ratio);
        ice.rain_ice_rain_rate = cold_dmul(ice.rain_ice_rain_rate, ratio);
        rain.rain_snow_rain_rate = cold_dmul(rain.rain_snow_rain_rate, ratio);
        rain.rain_graupel_rain_rate = cold_dmul(
            rain.rain_graupel_rain_rate, ratio);
    }
    const ulong snow_limit = cold_widen(cold_mul(-snow_mass, e.inverse_dt));
    ulong snow_sum = cold_dsub(frozen.snow, cloud.snow_hm);
    snow_sum = cold_dadd(snow_sum, rain.rain_snow_category_rate);
    if (has_snow && soft64_less(snow_sum, snow_limit)) {
        const ulong ratio = cold_ddiv(snow_limit, snow_sum);
        const ulong old_rate = frozen.snow;
        frozen.snow = cold_dmul(frozen.snow, ratio);
        cloud.snow_hm = cold_dmul(cloud.snow_hm, ratio);
        rain.rain_snow_category_rate = cold_dmul(
            rain.rain_snow_category_rate, ratio);
        vapor_sum = cold_dadd(vapor_sum,
                               cold_dsub(frozen.snow, old_rate));
    }
    const ulong graupel_limit = cold_widen(cold_mul(-graupel_mass,
                                                     e.inverse_dt));
    ulong graupel_sum = cold_dsub(frozen.graupel, cloud.graupel_hm);
    graupel_sum = cold_dadd(graupel_sum, rain.rain_graupel_graupel_rate);
    if (has_graupel
            && soft64_less(graupel_sum, graupel_limit)) {
        const ulong ratio = cold_ddiv(graupel_limit, graupel_sum);
        const ulong old_rate = frozen.graupel;
        frozen.graupel = cold_dmul(frozen.graupel, ratio);
        cloud.graupel_hm = cold_dmul(cloud.graupel_hm, ratio);
        rain.rain_graupel_graupel_rate = cold_dmul(
            rain.rain_graupel_graupel_rate, ratio);
        vapor_sum = cold_dadd(vapor_sum,
                               cold_dsub(frozen.graupel, old_rate));
    }
    const ulong paired = cold_dmin(
        soft64_abs_bits(rain.rain_graupel_rain_rate),
        soft64_abs_bits(rain.rain_graupel_graupel_rate));
    rain.rain_graupel_rain_rate = paired ^ 0x8000000000000000ul;
    rain.rain_graupel_graupel_rate = paired;
    return vapor_sum;
}

inline float cold_eff_aero(float diameter, float aerosol_diameter,
                           float viscosity, float rho, float temperature,
                           int species, bool wrf39) {
    constexpr float boltzmann = 1.3806503e-23f;
    constexpr float mean_path = 0.0256e-6f;
    float terminal_velocity = 1.0f;
    if (species == 0) {
        terminal_velocity = cold_add(-0.1021f,
                                     cold_mul(4.932e3f, diameter));
        terminal_velocity = cold_sub(terminal_velocity,
            cold_mul(cold_mul(0.9551e6f, diameter), diameter));
        terminal_velocity = cold_add(terminal_velocity,
            cold_mul(cold_mul(cold_mul(0.07934e9f, diameter), diameter), diameter));
        terminal_velocity = cold_sub(terminal_velocity,
            cold_mul(cold_mul(cold_mul(cold_mul(0.002362e12f, diameter),
                                       diameter), diameter), diameter));
    } else if (species == 1) {
        terminal_velocity = cold_mul(40.0f,
                                     cold_powf_version(diameter, 0.55f, wrf39));
    } else {
        terminal_velocity = cold_mul(442.0f,
                                     cold_powf_version(diameter, 0.89f, wrf39));
    }
    const float slip = cold_add(1.257f, cold_mul(0.4f,
        cold_expf_version(cold_div(cold_mul(-0.55f, aerosol_diameter),
                                   mean_path), wrf39)));
    const float correction = cold_add(1.0f, cold_mul(
        cold_div(cold_mul(2.0f, mean_path), aerosol_diameter), slip));
    const float diffusion = cold_div(
        cold_mul(cold_mul(boltzmann, temperature), correction),
        cold_mul(cold_mul(cold_mul(cold_mul(3.0f, 3.1415926536f),
                                    viscosity), aerosol_diameter), 1.0f));
    const float reynolds = cold_div(
        cold_mul(cold_mul(cold_mul(0.5f, rho), diameter), terminal_velocity),
        viscosity);
    const float schmidt = cold_div(viscosity, cold_mul(rho, diffusion));
    const float stokes = cold_div(
        cold_mul(cold_mul(cold_mul(aerosol_diameter, aerosol_diameter),
                          terminal_velocity), 1000.0f),
        cold_mul(cold_mul(9.0f, viscosity), diameter));
    const float aval = cold_add(1.0f,
        cold_logf_version(cold_add(1.0f, reynolds), wrf39));
    const float stokes2 = cold_div(
        cold_add(1.2f, cold_mul(1.0f / 12.0f, aval)), cold_add(1.0f, aval));
    const float sqrt_re = cold_sqrtf(reynolds);
    float brownian = cold_add(1.0f, cold_mul(
        cold_mul(0.4f, sqrt_re),
        cold_powf_version(schmidt, 0.3333f, wrf39)));
    brownian = cold_add(brownian,
                        cold_mul(cold_mul(0.16f, sqrt_re), cold_sqrtf(schmidt)));
    const float term1 = cold_mul(cold_div(4.0f, cold_mul(reynolds, schmidt)),
                                 brownian);
    const float ratio = cold_div(aerosol_diameter, diameter);
    const float term2 = cold_mul(cold_div(cold_mul(4.0f, aerosol_diameter), diameter),
        cold_add(0.02f, cold_mul(ratio,
            cold_add(1.0f, cold_mul(2.0f, sqrt_re)))));
    float efficiency = cold_add(term1, term2);
    if (stokes > stokes2) {
        const float excess = cold_sub(stokes, stokes2);
        efficiency = cold_add(efficiency,
            cold_powf_version(
                cold_div(excess, cold_add(excess, 0.666667f)), 1.5f, wrf39));
    }
    return cold_fmax(1.0e-5f, cold_fmin(efficiency, 1.0f));
}

// Translation of thompson_aa_bound_ice_number from
// thompson_aerosol_common.cuh, used by the warm-cell entry branch too.
inline void cold_bound_ice_number(float ice_mass, float density,
                                  thread float &ice_number_per_kg,
                                  bool wrf39) {
    if (ice_mass <= 1.0e-12f) {
        ice_number_per_kg = 0.0f;
        return;
    }
    const float am_i = 4.660029297e+02f;
    float ice_number = cold_fmax(1.0e-6f,
                                 cold_mul(ice_number_per_kg, density));
    const float base = cold_div(
        cold_mul(cold_mul(am_i, 6.0f), ice_number), ice_mass);
    const float root = cold_powf(base, as_type<float>(0x3eaaaaabu));
    ulong lambda = soft64_from_f32_bits(cold_f32(root));
    const ulong four = 0x4010000000000000ul;
    const float diameter = cold_f32_value(
        soft64_to_f32_bits(soft64_div(four, lambda)));
    if (diameter < 5.0e-6f) {
        lambda = soft64_div(four, 0x3ed4f8b588e368f1ul);
        const ulong square = soft64_mul(lambda, lambda);
        const ulong cube = soft64_mul(square, lambda);
        const float cube32 = cold_f32_value(soft64_to_f32_bits(cube));
        const float mass_factor = cold_div(
            cold_mul(cold_div(1.0f, 6.0f), ice_mass), am_i);
        ice_number = cold_fmin(wrf39 ? 499.0e3f : 999.0e3f,
                               cold_mul(mass_factor, cube32));
    } else if (diameter > 300.0e-6f) {
        lambda = soft64_div(four, 0x3f33a92a30553261ul);
        const ulong square = soft64_mul(lambda, lambda);
        const ulong cube = soft64_mul(square, lambda);
        const float cube32 = cold_f32_value(soft64_to_f32_bits(cube));
        const float mass_factor = cold_div(
            cold_mul(cold_div(1.0f, 6.0f), ice_mass), am_i);
        ice_number = cold_mul(mass_factor, cube32);
    }
    ice_number_per_kg = cold_div(
        cold_fmin(ice_number, wrf39 ? 499.0e3f : 999.0e3f), density);
}

inline float cold_apply_cell_rate(float state, ulong rate, float inverse_rho,
                                  float dt, bool subtract) {
    ulong scaled = cold_dmul(rate, cold_widen(inverse_rho));
    float tendency = cold_narrow(scaled);
    float increment = cold_mul(tendency, dt);
    return subtract ? cold_sub(state, increment) : cold_add(state, increment);
}

inline float cold_apply_shadow_rate(float state, ulong rate,
                                    float inverse_rho, float dt) {
    return cold_apply_cell_rate(state, rate, inverse_rho, dt, false);
}

// Full production cell dispatch. All float64 CUDA tables are passed as their
// IEEE word storage so every table operation follows the shared D64 path.
kernel void thompson_aa_cold_network(
    constant ThompsonAaColdNetworkResources &r [[buffer(0)]],
    constant ThompsonAaColdNetworkParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    device float *qi = r.qi;
    device float *ni = r.ni;
    device float *qs = r.qs;
    device float *qg = r.qg;
    device float *qr = r.qr;
    device float *nr = r.nr;
    device float *qc = r.qc;
    device float *temperature = r.temperature;
    device const float *pressure = r.pressure;
    device float *qv = r.qv;
    device const float *nc_entry = r.nc_entry;
    device const float *nwfa_entry = r.nwfa_entry;
    device const float *nifa_entry = r.nifa_entry;
    device float *ncten = r.ncten;
    device float *nwfaten = r.nwfaten;
    device float *nifaten = r.nifaten;
    device float *graupel_number_shadow = r.graupel_number_shadow;
    device float *snow_velocity_boost = r.snow_velocity_boost;
    device const ulong *ice_deposition_partition = r.ice_deposition_partition;
    device const ulong *ice_to_snow_mass = r.ice_to_snow_mass;
    device const ulong *ice_to_snow_number = r.ice_to_snow_number;
    device const ulong *tcs_racs1 = r.tcs_racs1;
    device const ulong *tmr_racs1 = r.tmr_racs1;
    device const ulong *tcs_racs2 = r.tcs_racs2;
    device const ulong *tmr_racs2 = r.tmr_racs2;
    device const ulong *tcr_sacr1 = r.tcr_sacr1;
    device const ulong *tms_sacr1 = r.tms_sacr1;
    device const ulong *tcr_sacr2 = r.tcr_sacr2;
    device const ulong *tms_sacr2 = r.tms_sacr2;
    device const ulong *tnr_racs1 = r.tnr_racs1;
    device const ulong *tnr_racs2 = r.tnr_racs2;
    device const ulong *tnr_sacr1 = r.tnr_sacr1;
    device const ulong *tnr_sacr2 = r.tnr_sacr2;
    device const ulong *tcg_racg = r.tcg_racg;
    device const ulong *tmr_racg = r.tmr_racg;
    device const ulong *tcr_gacr = r.tcr_gacr;
    device const ulong *tnr_racg = r.tnr_racg;
    device const ulong *tnr_gacr = r.tnr_gacr;
    device const ulong *rain_to_ice_mass = r.rain_to_ice_mass;
    device const ulong *rain_to_ice_number = r.rain_to_ice_number;
    device const ulong *rain_to_graupel_mass = r.rain_to_graupel_mass;
    device const ulong *rain_to_graupel_number = r.rain_to_graupel_number;
    device const ulong *rain_cloud_efficiency = r.rain_cloud_efficiency;
    device const ulong *cloud_to_ice_mass = r.cloud_to_ice_mass;
    device const ulong *cloud_to_ice_number = r.cloud_to_ice_number;
    const float dt = p.dt;
    const int n = p.n;
    const bool wrf39 = p.wrf39 != 0;
    if (idx >= uint(n)) return;
    const float qi0 = qi[idx], ni0 = ni[idx], qs0 = qs[idx];
    const float qg0 = qg[idx], qr0 = qr[idx], nr0 = nr[idx], qc0 = qc[idx];
    const float temp0 = temperature[idx], pressure0 = pressure[idx];
    const float qv_entry = qv[idx];
    float ni_work = ni0;
    float boost = 1.0f;
    ColdEntryState e;
    if (!cold_network_entry(qi0, ni_work, qs0, qg0, qr0, qc0,
            temp0, pressure0, qv_entry, dt, nc_entry[idx], nwfa_entry[idx],
            nifa_entry[idx], wrf39, boost, e)) {
        ni[idx] = ni_work;
        snow_velocity_boost[idx] = boost;
        return;
    }
    const float ice_mass = cold_mul(qi0, e.rho);
    const float rain_mass = cold_mul(qr0, e.rho);
    const float snow_mass = cold_mul(qs0, e.rho);
    const float graupel_mass = cold_mul(qg0, e.rho);
    const bool has_ice = qi0 > 1.0e-12f;
    const bool has_snow = qs0 > 1.0e-12f;
    const bool has_graupel = qg0 > 1.0e-12f;
    const float demott_target = cold_ice_demott(e.tempc, e.rho,
                                                 e.nifa_work, wrf39);
    const int nuclei_bin = cold_in_bin(demott_target);
    ColdSnowDistribution snow = cold_prepare_snow(qs0, e.rho, e.tempc, wrf39);
    ColdGraupelDistribution graupel = cold_prepare_graupel(
        qg0, e.rho, graupel_number_shadow[idx], wrf39);
    ColdIceRates ice = cold_ice_processes(qi0, ni_work, qr0, nr0, e,
        ice_deposition_partition, ice_to_snow_mass, ice_to_snow_number,
        wrf39);
    ColdRainRates rain = cold_rain_processes(qr0, qs0, qg0, nuclei_bin,
        e, ice, rain_to_ice_mass, rain_to_ice_number,
        rain_to_graupel_mass, rain_to_graupel_number,
        tcs_racs1, tmr_racs1, tcs_racs2, tmr_racs2,
        tcr_sacr1, tms_sacr1, tcr_sacr2, tms_sacr2,
        tnr_racs1, tnr_racs2, tnr_sacr1, tnr_sacr2,
        tcg_racg, tmr_racg, tcr_gacr, tnr_racg, tnr_gacr,
        wrf39, graupel.inverse_lambda);
    ColdCloudRates cloud = cold_cloud_processes(e, ice,
        snow.mass, snow.diameter, snow.smoe, snow.active,
        graupel.mass, graupel.diameter, graupel.inverse_lambda,
        graupel.intercept, nuclei_bin, rain_cloud_efficiency,
        cloud_to_ice_mass, cloud_to_ice_number, wrf39);
    ColdNucleationRates nucleation = cold_nucleation_processes(
        e, ice, rain, cloud, snow.number_ns, snow.smo0, wrf39);
    ColdFrozenVaporRates frozen = cold_frozen_vapor_processes(
        e, ice, snow.mass, snow.diameter, snow.tc0, qi0, ice_mass,
        snow.smob, snow.smoc, snow.smoe, snow.smo0, snow.active,
        has_graupel,
        graupel.mass, graupel.diameter, graupel.inverse_lambda,
        graupel.intercept, e.cloud_mvd, cloud.snow_riming, wrf39, boost);
    cloud.snow_riming = frozen.snow_riming;
    ulong vapor_sum = cold_limit_process_rates(e, ice_mass, rain_mass,
        snow.mass, graupel.mass, has_ice, e.has_cloud, has_snow,
        has_graupel, ice, rain, cloud, nucleation, frozen);
    cloud.hm_mass = cold_dadd(cloud.snow_hm, cloud.graupel_hm);

    // Recompute the post-limit coupled tendencies in source order.
    ulong rain_rate = cold_dadd(cloud.autoconversion, cloud.rain_accretion);
    rain_rate = cold_dsub(rain_rate, rain.freeze_graupel_rate);
    rain_rate = cold_dsub(rain_rate, rain.freeze_ice_rate);
    rain_rate = cold_dsub(rain_rate, ice.rain_ice_rain_rate);
    rain_rate = cold_dadd(rain_rate, rain.rain_snow_rain_rate);
    rain_rate = cold_dadd(rain_rate, rain.rain_graupel_rain_rate);
    ulong rain_number_sink = ice.rain_self_number_rate;
    if (wrf39) {
        rain_number_sink = cold_dadd(rain_number_sink,
                                     rain.freeze_rain_number_rate);
    } else {
        rain_number_sink = cold_dadd(rain_number_sink,
                                     rain.freeze_graupel_number_rate);
        rain_number_sink = cold_dadd(rain_number_sink,
                                     rain.freeze_ice_number_rate);
    }
    rain_number_sink = cold_dadd(rain_number_sink,
                                 ice.rain_ice_rain_number_rate);
    rain_number_sink = cold_dadd(rain_number_sink,
                                 rain.rain_snow_number_rate);
    rain_number_sink = cold_dadd(rain_number_sink,
                                 rain.rain_graupel_number_rate);
    ulong fusion_rate = cold_dadd(cloud.freezing, cloud.snow_riming);
    fusion_rate = cold_dadd(fusion_rate, frozen.snow_to_graupel);
    fusion_rate = cold_dadd(fusion_rate, cloud.graupel_riming);
    fusion_rate = cold_dadd(fusion_rate, rain.freeze_ice_rate);
    fusion_rate = cold_dadd(fusion_rate, rain.freeze_graupel_rate);
    fusion_rate = cold_dadd(fusion_rate, ice.rain_ice_rain_rate);
    fusion_rate = cold_dadd(fusion_rate, rain.rain_snow_category_rate);
    fusion_rate = cold_dadd(fusion_rate, rain.rain_snow_graupel_rate);
    fusion_rate = cold_dadd(fusion_rate, rain.rain_graupel_graupel_rate);
    const float latent_vapor = cold_add(2.5e6f,
        cold_mul(cold_sub(2106.0f, 4218.0f), e.tempc));
    const float latent_fusion = cold_sub(2.834e6f, latent_vapor);
    const float qv_tendency = cold_narrow(
        cold_dmul(vapor_sum, cold_widen(e.orho)));
    qv[idx] = cold_sub(qv_entry, cold_mul(qv_tendency, dt));
    ulong temperature_vapor = cold_dmul(cold_widen(cold_mul(
        2.834e6f, e.inverse_cp)), vapor_sum);
    temperature_vapor = cold_dmul(temperature_vapor, cold_widen(e.orho));
    temperature_vapor = cold_dmul(temperature_vapor, cold_widen(dt));
    ulong temperature_fusion = cold_dmul(cold_widen(cold_mul(
        latent_fusion, e.inverse_cp)), fusion_rate);
    temperature_fusion = cold_dmul(temperature_fusion, cold_widen(e.orho));
    temperature_fusion = cold_dmul(temperature_fusion, cold_widen(dt));
    temperature[idx] = cold_add(temp0,
        cold_narrow(cold_dadd(temperature_vapor, temperature_fusion)));

    ulong qi_rate = cold_dadd(nucleation.demott, nucleation.koop);
    qi_rate = cold_dadd(qi_rate, cloud.hm_mass);
    qi_rate = cold_dadd(qi_rate, cloud.freezing);
    qi_rate = cold_dadd(qi_rate, rain.freeze_ice_rate);
    qi_rate = cold_dadd(qi_rate, ice.ice_rate);
    qi_rate = cold_dsub(qi_rate, cloud.autoconversion);
    qi_rate = cold_dsub(qi_rate, frozen.snow_collection);
    qi_rate = cold_dsub(qi_rate, ice.rain_ice_ice_rate);
    qi[idx] = cold_fmax(0.0f, cold_apply_cell_rate(qi0, qi_rate,
                                                  e.orho, dt, false));
    ulong ni_rate = cold_dadd(nucleation.demott_number,
                              nucleation.koop_number);
    ni_rate = cold_dadd(ni_rate, cloud.hm_number);
    ni_rate = cold_dadd(ni_rate, cloud.freezing_number);
    ni_rate = cold_dadd(ni_rate, rain.freeze_ice_number_rate);
    ni_rate = cold_dadd(ni_rate, ice.ice_number_rate);
    ni_rate = cold_dsub(ni_rate, cloud.autoconversion_number);
    ni_rate = cold_dsub(ni_rate, frozen.snow_collection_number);
    ni_rate = cold_dsub(ni_rate, ice.rain_ice_ice_number_rate);
    ni[idx] = cold_fmax(0.0f, cold_apply_cell_rate(ni_work, ni_rate,
                                                   e.orho, dt, false));
    ulong qs_rate = cold_dadd(ice.ice_to_snow_rate, frozen.snow);
    qs_rate = cold_dadd(qs_rate, cloud.autoconversion);
    qs_rate = cold_dadd(qs_rate, frozen.snow_collection);
    qs_rate = cold_dadd(qs_rate, cloud.snow_riming);
    qs_rate = cold_dadd(qs_rate, rain.rain_snow_category_rate);
    qs_rate = cold_dsub(qs_rate, cloud.snow_hm);
    qs[idx] = cold_fmax(0.0f, cold_apply_cell_rate(qs0, qs_rate,
                                                   e.orho, dt, false));
    ulong qg_rate = cold_dadd(frozen.graupel, rain.freeze_graupel_rate);
    qg_rate = cold_dadd(qg_rate, frozen.snow_to_graupel);
    qg_rate = cold_dadd(qg_rate, cloud.graupel_riming);
    qg_rate = cold_dsub(qg_rate, cloud.graupel_hm);
    qg_rate = cold_dadd(qg_rate, ice.rain_ice_graupel_rate);
    qg_rate = cold_dadd(qg_rate, rain.rain_snow_graupel_rate);
    qg_rate = cold_dadd(qg_rate, rain.rain_graupel_graupel_rate);
    qg[idx] = cold_fmax(0.0f, cold_apply_cell_rate(qg0, qg_rate,
                                                   e.orho, dt, false));
    qr[idx] = cold_fmax(0.0f, cold_apply_cell_rate(qr0, rain_rate,
                                                  e.orho, dt, false));
    ulong nr_rate = cold_dsub(cloud.autoconversion_number, rain_number_sink);
    nr[idx] = cold_fmax(0.0f, cold_apply_cell_rate(nr0, nr_rate,
                                                   e.orho, dt, false));
    ulong qc_rate = cold_dadd(cloud.autoconversion, cloud.freezing);
    qc_rate = cold_dadd(qc_rate, cloud.rain_accretion);
    qc_rate = cold_dadd(qc_rate, cloud.snow_riming);
    qc_rate = cold_dadd(qc_rate, frozen.snow_to_graupel);
    qc_rate = cold_dadd(qc_rate, cloud.graupel_riming);
    qc[idx] = cold_fmax(0.0f, cold_apply_cell_rate(qc0, qc_rate,
                                                  e.orho, dt, true));

    float ni_bounded = ni[idx];
    float nr_bounded = nr[idx];
    cold_bound_ice_number(cold_mul(qi[idx], e.rho), e.rho, ni_bounded,
                          wrf39);
    cold_bound_rain_number(cold_mul(qr[idx], e.rho), e.rho, nr_bounded);
    ni[idx] = ni_bounded;
    nr[idx] = nr_bounded;
    const float qr_entry_wrf = qr0 > 1.0e-12f ? qr0 : 0.0f;
    const float nr_entry_wrf = qr0 > 1.0e-12f ? nr0 : 0.0f;
    if (!(cold_mul(qr[idx], e.rho) > 1.0e-12f)) {
        qr[idx] = cold_add(qr_entry_wrf,
            cold_mul(cold_mul(-qr_entry_wrf, e.inverse_dt), dt));
        nr[idx] = cold_add(nr_entry_wrf,
            cold_mul(cold_mul(-nr_entry_wrf, e.inverse_dt), dt));
    }
    if (!wrf39) {
        const float qg_entry_wrf = qg0 > 1.0e-12f ? qg0 : 0.0f;
        const float ng_entry_wrf = qg0 > 1.0e-12f
            ? graupel_number_shadow[idx] : 0.0f;
        if (!(cold_mul(qg[idx], e.rho) > 1.0e-12f)) {
            qg[idx] = cold_add(qg_entry_wrf,
                cold_mul(cold_mul(-qg_entry_wrf, e.inverse_dt), dt));
            graupel_number_shadow[idx] = cold_add(ng_entry_wrf,
                cold_mul(cold_mul(-ng_entry_wrf, e.inverse_dt), dt));
        } else {
            const float initial_number_per_kg = graupel_number_shadow[idx];
            const float initial_number = qg_entry_wrf > 0.0f
                ? cold_fmax(1.0e-6f,
                    cold_mul(initial_number_per_kg, e.rho)) : 0.0f;
            ulong graupel_number_rate = 0ul;
            if (qg_entry_wrf > 0.0f && graupel_mass > 0.0f)
                graupel_number_rate = cold_ddiv(
                    cold_dmul(frozen.graupel, cold_widen(initial_number)),
                    cold_widen(graupel_mass));
            graupel_number_rate = cold_dadd(graupel_number_rate,
                rain.freeze_graupel_number_rate);
            graupel_number_rate = cold_dadd(graupel_number_rate,
                frozen.snow_to_graupel_number);
            graupel_number_rate = cold_dadd(graupel_number_rate,
                ice.rain_ice_rain_number_rate);
            graupel_number_rate = cold_dadd(graupel_number_rate,
                rain.rain_snow_number_rate);
            graupel_number_shadow[idx] = cold_apply_shadow_rate(
                initial_number_per_kg, graupel_number_rate, e.orho, dt);
            const float xrg = cold_mul(qg[idx], e.rho);
            const float shadow_m3 = cold_mul(
                graupel_number_shadow[idx], e.rho);
            const float xng = cold_fmax(1.0e-6f, shadow_m3);
            const float am_g = cold_div(
                cold_mul(3.1415926536f, 400.0f), 6.0f);
            const float mvd_numerator = cold_add(3.0f, 0.672f);
            const float lambda_arg = cold_div(cold_mul(cold_mul(
                am_g, 6.0f), xng), xrg);
            ulong lambda = cold_widen(cold_powf(
                lambda_arg, 0.3333333433f));
            const float mvd_g = cold_narrow(cold_ddiv(
                cold_widen(mvd_numerator), lambda));
            bool bounded = false;
            if (mvd_g > 25.4e-3f) {
                lambda = cold_widen(cold_div(mvd_numerator, 25.4e-3f));
                bounded = true;
            } else if (mvd_g < 50.0e-6f) {
                lambda = cold_widen(cold_div(mvd_numerator, 50.0e-6f));
                bounded = true;
            }
            if (bounded) {
                const float mass_factor = cold_mul(1.0f / 6.0f, xrg);
                const ulong cube = cold_dpow(lambda,
                                             0x4008000000000000ul);
                const float bounded_xng = cold_narrow(cold_ddiv(
                    cold_dmul(cold_widen(mass_factor), cube),
                    cold_widen(am_g)));
                float shadow_tendency = cold_sub(bounded_xng,
                    cold_mul(ng_entry_wrf, e.rho));
                shadow_tendency = cold_mul(shadow_tendency, e.inverse_dt);
                shadow_tendency = cold_mul(shadow_tendency, e.orho);
                graupel_number_shadow[idx] = cold_add(ng_entry_wrf,
                    cold_mul(shadow_tendency, dt));
            }
        }
    }

    // Aerosol and droplet tendencies are added from the rates held before
    // each source limiter, matching the independent Fortran accumulators.
    ulong cloud_tendency = cloud.autoconversion_number;
    cloud_tendency = cold_dsub(cloud_tendency, cloud.cloud_number_rain);
    cloud_tendency = cold_dsub(cloud_tendency, cloud.freezing_number);
    cloud_tendency = cold_dsub(cloud_tendency, cloud.cloud_number_snow);
    cloud_tendency = cold_dsub(cloud_tendency, cloud.cloud_number_graupel);
    ncten[idx] = cold_narrow(cold_dadd(cold_widen(ncten[idx]),
        cold_dmul(cloud_tendency, cold_widen(e.orho))));
    ulong nwfa_tendency = cold_dadd(cloud.nwfa_rain, cloud.nwfa_snow);
    nwfa_tendency = cold_dadd(nwfa_tendency, cloud.nwfa_graupel);
    nwfa_tendency = cold_dadd(nwfa_tendency, nucleation.koop_number);
    nwfaten[idx] = cold_narrow(cold_dsub(cold_widen(nwfaten[idx]),
        cold_dmul(nwfa_tendency, cold_widen(e.orho))));
    ulong nifa_tendency = cold_dadd(cloud.nifa_rain, cloud.nifa_snow);
    nifa_tendency = cold_dadd(nifa_tendency, cloud.nifa_graupel);
    nifaten[idx] = cold_narrow(cold_dsub(cold_widen(nifaten[idx]),
        cold_dmul(nifa_tendency, cold_widen(e.orho))));
    nifaten[idx] = cold_narrow(cold_dsub(cold_widen(nifaten[idx]),
        cold_dmul(nucleation.demott_number, cold_widen(e.orho))));
    snow_velocity_boost[idx] = boost;
}
