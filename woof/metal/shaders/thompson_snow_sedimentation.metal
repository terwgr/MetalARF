#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#include "thompson_snow_sediment_with_rain_profile_and_presence_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"

static inline ulong snow64(float x) {
    return soft64_from_f32_bits(as_type<uint>(x));
}
static inline float snow32(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}
static inline ulong snow_pow64(ulong x, float y) {
    return soft64_plm_pow(x, snow64(y));
}
static inline float snow_rain_root(float x) {
    // CUDA evaluates powf in FP32 before widening its result to double.
    return precise::pow(x, 1.0f / 3.0f);
}

// Serial-column transcription of thompson_snow_sediment_impl in
// woof/core/kernels/thompson.cu. Every column's vertical recurrence remains
// ordered because each level inherits the fall speed from the level above.
kernel void thompson_snow_sediment_with_rain_profile_and_presence(
    constant ThompsonSnowSedimentWithRainProfileAndPresenceResources& r [[buffer(0)]],
    constant ThompsonSnowSedimentWithRainProfileAndPresenceParams& p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny * p.nx);
    if (column >= ncol) return;
    const int nz = p.nz;
    const int nx = p.nx;
    const int ny = p.ny;
    const int j = int(column) / nx;
    const int i = int(column) - j * nx;

    // Match CUDA's terminal writes for its guarded empty-column shortcut.
    bool empty = p.dt >= 0.0f && p.dt <= 100000.0f;
    for (int k = 0; k < nz && empty; ++k) {
        const uint idx = uint(k * ny * nx + j * nx + i);
        empty = r.qs[idx] >= -1.0f && r.qs[idx] <= 1.0e-12f
            && r.temperature[idx] >= 100.0f && r.temperature[idx] <= 400.0f
            && r.pressure[idx] >= 1.0f && r.pressure[idx] <= 120000.0f
            && r.qv[idx] >= -1.0f && r.qv[idx] <= 1.0f
            && r.dz[idx] >= 1.0f && r.dz[idx] <= 100000.0f;
    }
    if (empty) {
        for (int k = 0; k < nz; ++k) {
            const uint idx = uint(k * ny * nx + j * nx + i);
            r.qs[idx] = 0.0f;
        }
        if (p.accumulate_surface) {
            r.rainncv[column] += 0.0f;
            r.snowncv[column] += 0.0f;
        } else {
            r.rainncv[column] = 0.0f;
            r.snowncv[column] = 0.0f;
        }
        r.rainnc[column] += 0.0f;
        r.snownc[column] += 0.0f;
        return;
    }

    float density[256], snow_mass[256], mass_velocity[256], mass_flux[256];
    float qs_tendency[256], qs_initial[256];
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    const float org3 = 1.0f / 6.0f;
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above = 0.0f;
    float rain_velocity_above = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const uint idx = uint(k * ny * nx + j * nx + i);
        const float qvk = fmax(1.0e-10f, r.qv[idx]);
        const float rho = 0.622f * r.pressure[idx]
            / (287.04f * r.temperature[idx] * (qvk + 0.622f));
        density[k] = rho;
        qs_initial[k] = r.qs[idx];
        qs_tendency[k] = 0.0f;

        const float carried_rain_density = r.melt_rain_density[idx];
        const float rain_density = fabs(carried_rain_density);
        const bool l_qr = carried_rain_density != 0.0f;
        const bool rewritten = carried_rain_density < 0.0f;
        const float rain_rr = !l_qr ? 1.0e-12f
            : rewritten ? fmax(1.0e-12f, r.melt_rain_qr[idx] * rain_density)
                        : r.melt_rain_qr[idx] * rain_density;
        float rain_velocity = 0.0f;
        if (rain_rr > 1.0e-12f) {
            const float rain_number = fmax(1.0e-6f,
                r.melt_rain_nr[idx] * rain_density);
            const float lambda_arg = am_r * 6.0f * rain_number / rain_rr;
            const ulong lambda = snow64(snow_rain_root(lambda_arg));
            const float rain_rhof = precise::sqrt(rho_not / rho);
            const float prefix = rain_rhof * 4854.0f * 24.0f * org3;
            const ulong first = soft64_mul(snow64(prefix), snow_pow64(lambda, 4.0f));
            const ulong shifted = soft64_add(lambda, snow64(195.0f));
            rain_velocity = snow32(soft64_mul(first,
                snow_pow64(shifted, -5.0f)));
        } else {
            rain_velocity = rain_velocity_above;
        }
        rain_velocity_above = rain_velocity;

        const float snow_state_rho = r.reference_density[idx];
        if (r.qs[idx] > 1.0e-12f
                && r.qs[idx] * snow_state_rho > 1.0e-12f) {
            const float rs = r.qs[idx] * snow_state_rho;
            const float smob = rs * (1.0f / 0.069f);
            const float tc0 = fmin(-0.1f, r.reference_temperature[idx] - 273.15f);
            const float moment = 3.0f;
            const float tc02 = tc0 * tc0;
            const float moment2 = moment * moment;
            const float loga = 5.065339f + -0.062659f * tc0
                + -3.032362f * moment + 0.029469f * tc0 * moment
                + -0.000285f * tc02 + 0.31255f * moment2
                + 0.000204f * tc02 * moment + 0.003199f * tc0 * moment2
                + 0.0f * tc02 * tc0 + -0.015952f * moment2 * moment;
            const float exponent = 0.476221f + -0.015896f * tc0
                + 0.165977f * moment + 0.007468f * tc0 * moment
                + -0.000141f * tc02 + 0.060366f * moment2
                + 0.000079f * tc02 * moment + 0.000594f * tc0 * moment2
                + 0.0f * tc02 * tc0 + -0.003577f * moment2 * moment;
            const float smoc = precise::pow(10.0f, loga)
                * precise::pow(smob, exponent);
            const float mean_ratio = smob / smoc;
            float ils1 = 1.0f / (mean_ratio * 20.78f + 100.0f);
            float ils2 = 1.0f / (mean_ratio * 3.29f + 100.0f);
            const float ratio_power = precise::pow(mean_ratio, 0.6357f);
            const float numerator1 = 490.6f * 3.51325202f
                * precise::pow(ils1, 3.55f);
            const float numerator2 = 17.46f * ratio_power * 7.61279917f
                * precise::pow(ils2, 4.1857f);
            ils1 = 1.0f / (mean_ratio * 20.78f);
            ils2 = 1.0f / (mean_ratio * 3.29f);
            const float denominator1 = 490.6f * 2.0f * precise::pow(ils1, 3.0f);
            const float denominator2 = 17.46f * ratio_power * 3.87160635f
                * precise::pow(ils2, 3.6357f);
            const float rhof = precise::sqrt(rho_not / rho);
            float snow_velocity = rhof * 40.0f
                * (numerator1 + numerator2) / (denominator1 + denominator2);
            snow_velocity *= r.velocity_boost[idx];
            if (r.snow_melt_marker[idx] != 0.0f) {
                const float solid_fraction = rs / (rs + rain_rr);
                snow_velocity = snow_velocity * solid_fraction
                    + rain_velocity * (1.0f - solid_fraction);
            }
            mass_velocity[k] = snow_velocity;
            snow_mass[k] = rs;
        } else {
            snow_mass[k] = r.qs[idx] > 1.0e-12f
                ? r.qs[idx] * snow_state_rho : 1.0e-12f;
            mass_velocity[k] = velocity_above;
        }
        velocity_above = mass_velocity[k];
        if (mass_velocity[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = r.dz[idx] / mass_velocity[k];
            nstep = max(nstep, int(float(p.dt) / delta_tp + 1.0f));
        }
    }
    if (sediment_top == nz - 1) sediment_top = nz - 2;
    nstep = max(nstep, 1);
    const float onstep = 1.0f / float(nstep);
    const float dt_substep = float(p.dt) * onstep;
    float exported = 0.0f;
    for (int step = 0; step < nstep; ++step) {
        for (int k = nz - 1; k >= 0; --k)
            mass_flux[k] = mass_velocity[k] * snow_mass[k];
        int k = nz - 1;
        uint idx = uint(k * ny * nx + j * nx + i);
        float inv_dz = 1.0f / r.dz[idx];
        float inv_rho = 1.0f / density[k];
        qs_tendency[k] -= mass_flux[k] * inv_dz * onstep * inv_rho;
        snow_mass[k] = fmax(1.0e-12f,
            snow_mass[k] - mass_flux[k] * inv_dz * dt_substep);
        for (k = sediment_top; k >= 0; --k) {
            idx = uint(k * ny * nx + j * nx + i);
            inv_dz = 1.0f / r.dz[idx];
            inv_rho = 1.0f / density[k];
            const float divergence = mass_flux[k + 1] - mass_flux[k];
            qs_tendency[k] += divergence * inv_dz * onstep * inv_rho;
            snow_mass[k] = fmax(1.0e-12f,
                snow_mass[k] + divergence * inv_dz * dt_substep);
        }
        if (snow_mass[0] > 1.0e-9f) exported += mass_flux[0] * dt_substep;
    }
    for (int k = 0; k < nz; ++k) {
        const uint idx = uint(k * ny * nx + j * nx + i);
        const float qs_new = qs_initial[k] + qs_tendency[k] * float(p.dt);
        r.qs[idx] = qs_new <= 1.0e-12f ? 0.0f : qs_new;
    }
    if (p.accumulate_surface) {
        r.rainncv[column] += exported;
        r.snowncv[column] += exported;
    } else {
        r.rainncv[column] = exported;
        r.snowncv[column] = exported;
    }
    r.rainnc[column] += exported;
    r.snownc[column] += exported;
}
