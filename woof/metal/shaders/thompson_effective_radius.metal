#include <metal_stdlib>
#include "thompson_effective_radius_bindings.metalh"
#include "soft_binary64.metalh"

using namespace metal;
#pragma clang fp contract(off)

static inline ulong radius64(float x) {
    return soft64_from_f32_bits(as_type<uint>(x));
}
static inline float radius32(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}

static inline float radius_field_a(float tc, float moment) {
    const float tc2 = tc * tc, moment2 = moment * moment;
    const float loga = 5.065339f + -0.062659f * tc
        + -3.032362f * moment + 0.029469f * tc * moment
        + -0.000285f * tc2 + 0.31255f * moment2
        + 0.000204f * tc2 * moment + 0.003199f * tc * moment2
        + 0.0f * tc2 * tc + -0.015952f * moment2 * moment;
    return precise::pow(10.0f, loga);
}
static inline float radius_field_b(float tc, float moment) {
    const float tc2 = tc * tc, moment2 = moment * moment;
    return 0.476221f + -0.015896f * tc
        + 0.165977f * moment + 0.007468f * tc * moment
        + -0.000141f * tc2 + 0.060366f * moment2
        + 0.000079f * tc2 * moment + 0.000594f * tc * moment2
        + 0.0f * tc2 * tc + -0.003577f * moment2 * moment;
}

kernel void thompson_effective_radius(
    constant ThompsonEffectiveRadiusResources& r [[buffer(0)]],
    constant ThompsonEffectiveRadiusParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;

    if (r.qc[idx] == 0.0f && r.qi[idx] == 0.0f && r.qs[idx] == 0.0f) {
        r.effc[idx] = fmax(2.49e-6f, fmin(2.49e-6f, 50.0e-6f)) * 1.0e6f;
        r.effi[idx] = fmax(4.99e-6f, fmin(4.99e-6f, 125.0e-6f)) * 1.0e6f;
        r.effs[idx] = fmax(9.99e-6f, fmin(9.99e-6f, 999.0e-6f)) * 1.0e6f;
        return;
    }

    const float rho = 0.622f * r.pressure[idx]
        / (287.04f * r.temperature[idx] * (r.qv[idx] + 0.622f));
    const float rc = fmax(1.0e-12f, r.qc[idx] * rho);
    const float ri = fmax(1.0e-12f, r.qi[idx] * rho);
    const float ice_number = fmax(1.0e-6f, r.ni[idx] * rho);
    const float rs = fmax(1.0e-12f, r.qs[idx] * rho);

    float reqc = 2.49e-6f, reqi = 4.99e-6f, reqs = 9.99e-6f;
    if (rc > 1.0e-12f) {
        const float am_r = 3.1415926536f * 1000.0f / 6.0f;
        const float lambda_arg = 100.0e6f * am_r * 2730.0f / rc;
        const float lambda_f32 = precise::pow(lambda_arg, 1.0f / 3.0f);
        const float diagnosed = radius32(soft64_div(
            radius64(7.5f), radius64(lambda_f32)));
        reqc = fmax(2.51e-6f, fmin(diagnosed, 50.0e-6f));
    }
    if (ri > 1.0e-12f && ice_number > 1.0e-6f) {
        const float am_i = 3.1415926536f * 890.0f / 6.0f;
        const float lambda_arg = am_i * 6.0f * ice_number / ri;
        const float lambda_f32 = precise::pow(lambda_arg, 1.0f / 3.0f);
        const float diagnosed = radius32(soft64_div(
            radius64(1.5f), radius64(lambda_f32)));
        reqi = fmax(2.51e-6f, fmin(diagnosed, 125.0e-6f));
    }
    if (rs > 1.0e-12f) {
        const float tc = fmin(-0.1f, r.temperature[idx] - 273.15f);
        const float smob = rs * (1.0f / 0.069f);
        const float a = radius_field_a(tc, 3.0f);
        const float b = radius_field_b(tc, 3.0f);
        const float smoc = a * precise::pow(smob, b);
        const float diagnosed = 0.5f * (smoc / smob);
        reqs = fmax(5.01e-6f, fmin(diagnosed, 999.0e-6f));
    }

    r.effc[idx] = fmax(2.49e-6f, fmin(reqc, 50.0e-6f)) * 1.0e6f;
    r.effi[idx] = fmax(4.99e-6f, fmin(reqi, 125.0e-6f)) * 1.0e6f;
    r.effs[idx] = fmax(9.99e-6f, fmin(reqs, 999.0e-6f)) * 1.0e6f;
}
