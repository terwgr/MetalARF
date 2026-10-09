#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#include "thompson_graupel_sediment_with_density_column_mask_shadow_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"

static inline ulong graupel64(float x) {
    return soft64_from_f32_bits(as_type<uint>(x));
}
static inline float graupel32(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}
static inline ulong graupel_pow(ulong x, float y) {
    return soft64_plm_pow(x, graupel64(y));
}
static inline ulong graupel_powf_to64(float x, float y) {
    return graupel64(precise::pow(x, y));
}

// Serial column transcription of thompson_graupel_sediment_impl in
// woof/core/kernels/thompson.cu. The diagnostic number moment is private to
// the WRF call and is transported independently of the mass velocity.
kernel void thompson_graupel_sediment_with_density_column_mask_shadow(
    constant ThompsonGraupelSedimentWithDensityColumnMaskShadowResources& r [[buffer(0)]],
    constant ThompsonGraupelSedimentWithDensityColumnMaskShadowParams& p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint columns = uint(p.ny * p.nx);
    if (column >= columns || r.active_columns[column] == 0.0f) return;
    const int nz = p.nz, nx = p.nx;
    const int j = int(column) / nx;
    const int i = int(column) - j * nx;

    // Match CUDA's conservative no-fallout shortcut and all its terminal
    // writes, including signed-zero-preserving shadow arithmetic.
    bool empty = p.dt >= 0.0f && p.dt <= 100000.0f;
    if (empty) {
        for (int k = 0; k < nz; ++k) {
            const uint idx = uint(k * p.ny * nx + j * nx + i);
            if (!(r.qg[idx] >= -1.0f && r.qg[idx] <= 1.0e-12f
                    && isfinite(r.graupel_number_shadow[idx])
                    && r.temperature[idx] >= 100.0f && r.temperature[idx] <= 400.0f
                    && r.pressure[idx] >= 1.0f && r.pressure[idx] <= 120000.0f
                    && r.qv[idx] >= -1.0f && r.qv[idx] <= 1.0f
                    && r.dz[idx] >= 1.0f && r.dz[idx] <= 100000.0f)) {
                empty = false;
                break;
            }
        }
    }
    if (empty) {
        for (int k = 0; k < nz; ++k) {
            const uint idx = uint(k * p.ny * nx + j * nx + i);
            r.qg[idx] = 0.0f;
            r.graupel_number_shadow[idx] = r.graupel_number_shadow[idx] + 0.0f * p.dt;
        }
        if (p.accumulate_surface) r.rainncv[column] += 0.0f;
        else r.rainncv[column] = 0.0f;
        if (p.accumulate_surface) r.graupelncv[column] += 0.0f;
        else r.graupelncv[column] = 0.0f;
        r.rainnc[column] += 0.0f;
        r.graupelnc[column] += 0.0f;
        return;
    }

    float density[256], graupel_mass[256], mass_velocity[256], mass_flux[256];
    float qg_tendency[256], qg_initial[256], diagnostic_number[256];
    float number_velocity[256], number_flux[256], number_tendency[256], number_initial[256];
    const float pi = 3.1415926536f;
    const float am_g = pi * 400.0f / 6.0f;
    const float ogg3 = 1.0f / 6.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above = 0.0f;
    float number_velocity_above = 0.0f;

    for (int k = nz - 1; k >= 0; --k) {
        const uint idx = uint(k * p.ny * nx + j * nx + i);
        const float qvk = fmax(1.0e-10f, r.qv[idx]);
        const float rho = 0.622f * r.pressure[idx]
            / (287.04f * r.temperature[idx] * (qvk + 0.622f));
        density[k] = rho;
        qg_initial[k] = r.qg[idx];
        number_initial[k] = r.graupel_number_shadow[idx];
        qg_tendency[k] = 0.0f;
        number_tendency[k] = 0.0f;
        if (r.qg[idx] > 1.0e-12f) {
            const float state_rho = r.reference_density[idx];
            const float rg = r.qg[idx] * state_rho;
            const float log_mass = precise::log10(fmax(1.0e-9f, rg));
            const float intercept_power = fmax(2.0f, fmin(
                3.0f + (2.0f / 7.0f) * (log_mass + 8.0f), 6.0f));
            const float intercept = precise::pow(10.0f, intercept_power);
            const float lambda_arg = intercept * am_g * 6.0f / rg;
            ulong lambda = graupel_powf_to64(lambda_arg, 0.25f);
            float mvd = graupel32(soft64_div(graupel64(3.672f), lambda));
            if (mvd > 25.4e-3f) {
                mvd = 25.4e-3f;
                lambda = soft64_div(graupel64(3.672f), graupel64(mvd));
            } else if (mvd < 50.0e-6f) {
                mvd = 50.0e-6f;
                lambda = soft64_div(graupel64(3.672f), graupel64(mvd));
            }
            const float rhof = precise::sqrt(rho_not / rho);
            const float prefix = rhof * 442.0f * 20.3632278f * ogg3;
            mass_velocity[k] = graupel32(soft64_mul(graupel64(prefix),
                graupel_pow(soft64_div(0x3ff0000000000000ul, lambda), 0.89f)));
            graupel_mass[k] = rg;
            diagnostic_number[k] = graupel32(soft64_mul(
                graupel64((ogg3 * rg) / am_g), soft64_mul(lambda,
                soft64_mul(lambda, lambda))));
            const float number_prefix = rhof * 442.0f * 2.21880022f;
            number_velocity[k] = graupel32(soft64_mul(graupel64(number_prefix),
                graupel_pow(soft64_div(0x3ff0000000000000ul, lambda), 0.89f)));
        } else {
            graupel_mass[k] = 1.0e-12f;
            mass_velocity[k] = velocity_above;
            diagnostic_number[k] = 1.0e-6f;
            number_velocity[k] = number_velocity_above;
        }
        velocity_above = mass_velocity[k];
        number_velocity_above = number_velocity[k];
        if (mass_velocity[k] > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const uint idx = uint(k * p.ny * nx + j * nx + i);
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
        for (int k = nz - 1; k >= 0; --k) {
            mass_flux[k] = mass_velocity[k] * graupel_mass[k];
            number_flux[k] = number_velocity[k] * diagnostic_number[k];
        }
        int k = nz - 1;
        uint idx = uint(k * p.ny * nx + j * nx + i);
        float inv_dz = 1.0f / r.dz[idx];
        float inv_rho = 1.0f / density[k];
        qg_tendency[k] -= mass_flux[k] * inv_dz * onstep * inv_rho;
        number_tendency[k] -= number_flux[k] * inv_dz * onstep * inv_rho;
        graupel_mass[k] = fmax(1.0e-12f,
            graupel_mass[k] - mass_flux[k] * inv_dz * dt_substep);
        diagnostic_number[k] = fmax(1.0e-6f,
            diagnostic_number[k] - number_flux[k] * inv_dz * dt_substep);
        for (k = sediment_top; k >= 0; --k) {
            idx = uint(k * p.ny * nx + j * nx + i);
            inv_dz = 1.0f / r.dz[idx];
            inv_rho = 1.0f / density[k];
            const float divergence = mass_flux[k + 1] - mass_flux[k];
            qg_tendency[k] += divergence * inv_dz * onstep * inv_rho;
            graupel_mass[k] = fmax(1.0e-12f,
                graupel_mass[k] + divergence * inv_dz * dt_substep);
            const float number_divergence = number_flux[k + 1] - number_flux[k];
            number_tendency[k] += number_divergence * inv_dz * onstep * inv_rho;
            diagnostic_number[k] = fmax(1.0e-6f,
                diagnostic_number[k] + number_divergence * inv_dz * dt_substep);
        }
        if (graupel_mass[0] > 1.0e-9f) exported += mass_flux[0] * dt_substep;
    }
    for (int k = 0; k < nz; ++k) {
        const uint idx = uint(k * p.ny * nx + j * nx + i);
        const float qg_new = qg_initial[k] + qg_tendency[k] * float(p.dt);
        r.qg[idx] = qg_new <= 1.0e-12f ? 0.0f : qg_new;
        r.graupel_number_shadow[idx] = number_initial[k] + number_tendency[k] * float(p.dt);
    }
    if (p.accumulate_surface) {
        r.rainncv[column] += exported;
        r.graupelncv[column] += exported;
    } else {
        r.rainncv[column] = exported;
        r.graupelncv[column] = exported;
    }
    r.rainnc[column] += exported;
    r.graupelnc[column] += exported;
}
