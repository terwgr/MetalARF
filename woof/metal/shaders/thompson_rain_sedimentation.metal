#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#include "thompson_rain_sediment_with_density_and_presence_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"

static inline ulong rain64(float x) {
    return soft64_from_f32_bits(as_type<uint>(x));
}
static inline float rain32(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}
static inline ulong rain_pow(ulong x, float y) {
    return soft64_plm_pow(x, rain64(y));
}
static inline ulong rain_root(float x) {
    // Pinned CUDA first evaluates powf in FP32, then widens that result.
    return rain64(precise::pow(x, 1.0f / 3.0f));
}

// Column-serial transcription of thompson_rain_sediment_impl in
// woof/core/kernels/thompson.cu. The sign/zero state of reference_density
// carries the preceding source's L_qr and rewritten-pair markers. This one
// operation graph covers both upstream KMAX entry points and the 64-level
// level-parallel entry's caller-visible result.
kernel void thompson_rain_sediment_with_density_and_presence(
    constant ThompsonRainSedimentWithDensityAndPresenceResources& r [[buffer(0)]],
    constant ThompsonRainSedimentWithDensityAndPresenceParams& p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny * p.nx);
    if (column >= ncol) return;
    const uint nz = uint(p.nz), ny = uint(p.ny), nx = uint(p.nx);
    const uint j = column / nx;
    const uint i = column - j * nx;

    float density[256], rain_mass[256], rain_number[256];
    float mass_velocity[256], number_velocity[256];
    float mass_flux[256], number_flux[256];
    float qr_tendency[256], nr_tendency[256];
    float qr_initial[256], nr_initial[256];

    const float pi = 3.1415926536f;
    const float am_r = pi * 1000.0f / 6.0f;
    const float org3 = 1.0f / 6.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    int sediment_top = 0;
    int nstep = 0;
    float velocity_above_mass = 0.0f;
    float velocity_above_number = 0.0f;

    for (int kk = int(nz) - 1; kk >= 0; --kk) {
        const uint k = uint(kk);
        const uint idx = k * ny * nx + j * nx + i;
        const float qvk = fmax(1.0e-10f, r.qv[idx]);
        const float rho = 0.622f * r.pressure[idx]
            / (287.04f * r.temperature[idx] * (qvk + 0.622f));
        const float carried = r.reference_density[idx];
        const float rain_density = fabs(carried);
        density[k] = rho;
        qr_initial[k] = r.qr[idx];
        nr_initial[k] = r.nr[idx];
        qr_tendency[k] = 0.0f;
        nr_tendency[k] = 0.0f;

        const bool l_qr = carried != 0.0f;
        const bool rewritten = carried < 0.0f;
        const float rr = !l_qr ? 1.0e-12f
            : rewritten ? fmax(1.0e-12f, r.qr[idx] * rain_density)
                        : r.qr[idx] * rain_density;
        if (rr > 1.0e-12f) {
            const float nn = fmax(1.0e-6f, r.nr[idx] * rain_density);
            const float lambda_arg = am_r * 6.0f * nn / rr;
            const ulong lambda = rain_root(lambda_arg);
            rain_mass[k] = rr;
            rain_number[k] = nn;
            const float rhof = precise::sqrt(rho_not / rho);
            const float mass_prefix = rhof * 4854.0f * 24.0f * org3;
            const ulong shifted_lambda = soft64_add(lambda, rain64(195.0f));
            const ulong mass_first = soft64_mul(rain64(mass_prefix),
                                                  rain_pow(lambda, 4.0f));
            mass_velocity[k] = rain32(soft64_mul(
                mass_first, rain_pow(shifted_lambda, -5.0f)));
            const float number_prefix = (rhof * 4854.0f * 3.3233511f)
                / 1.3293403f;
            const ulong number_first = soft64_mul(rain64(number_prefix),
                                                   rain_pow(lambda, 2.5f));
            number_velocity[k] = rain32(soft64_mul(
                number_first, rain_pow(shifted_lambda, -3.5f)));
        } else {
            rain_mass[k] = rr;
            rain_number[k] = l_qr
                ? fmax(1.0e-6f, r.nr[idx] * rain_density) : 1.0e-6f;
            mass_velocity[k] = velocity_above_mass;
            number_velocity[k] = velocity_above_number;
        }
        velocity_above_mass = mass_velocity[k];
        velocity_above_number = number_velocity[k];

        const float vmax = fmax(mass_velocity[k], number_velocity[k]);
        if (vmax > 1.0e-3f) {
            sediment_top = max(sediment_top, kk);
            const float delta_tp = r.dz[idx] / vmax;
            nstep = max(nstep, int(float(p.dt) / delta_tp + 1.0f));
        }
    }
    if (sediment_top == int(nz) - 1) sediment_top = int(nz) - 2;
    nstep = max(nstep, 1);
    const float onstep = 1.0f / float(nstep);
    const float dt_substep = float(p.dt) * onstep;
    float exported = 0.0f;

    for (int step = 0; step < nstep; ++step) {
        for (int kk = int(nz) - 1; kk >= 0; --kk) {
            const uint k = uint(kk);
            mass_flux[k] = mass_velocity[k] * rain_mass[k];
            number_flux[k] = number_velocity[k] * rain_number[k];
        }
        uint k = nz - 1;
        uint idx = k * ny * nx + j * nx + i;
        float inv_dz = 1.0f / r.dz[idx];
        float inv_rho = 1.0f / density[k];
        qr_tendency[k] -= mass_flux[k] * inv_dz * onstep * inv_rho;
        nr_tendency[k] -= number_flux[k] * inv_dz * onstep * inv_rho;
        rain_mass[k] = fmax(1.0e-12f,
            rain_mass[k] - mass_flux[k] * inv_dz * dt_substep);
        rain_number[k] = fmax(1.0e-6f,
            rain_number[k] - number_flux[k] * inv_dz * dt_substep);

        for (int kk = sediment_top; kk >= 0; --kk) {
            k = uint(kk);
            idx = k * ny * nx + j * nx + i;
            inv_dz = 1.0f / r.dz[idx];
            inv_rho = 1.0f / density[k];
            const float mass_divergence = mass_flux[k + 1] - mass_flux[k];
            const float number_divergence = number_flux[k + 1] - number_flux[k];
            qr_tendency[k] += mass_divergence * inv_dz * onstep * inv_rho;
            nr_tendency[k] += number_divergence * inv_dz * onstep * inv_rho;
            rain_mass[k] = fmax(1.0e-12f,
                rain_mass[k] + mass_divergence * inv_dz * dt_substep);
            rain_number[k] = fmax(1.0e-6f,
                rain_number[k] + number_divergence * inv_dz * dt_substep);
        }
        if (rain_mass[0] > 1.0e-9f) exported += mass_flux[0] * dt_substep;
    }

    for (uint k = 0; k < nz; ++k) {
        const uint idx = k * ny * nx + j * nx + i;
        const float qr_new = qr_initial[k] + qr_tendency[k] * float(p.dt);
        float nr_new = fmax(1.0e-6f / density[k],
                            nr_initial[k] + nr_tendency[k] * float(p.dt));
        if (qr_new <= 1.0e-12f) {
            r.qr[idx] = 0.0f;
            r.nr[idx] = 0.0f;
            continue;
        }
        const float lambda_arg = am_r * 6.0f * nr_new / qr_new;
        ulong lambda = rain_root(lambda_arg);
        const ulong mvd_numerator = 0x400d604189374bc7ul; // binary64 3.672
        float mvd = rain32(soft64_div(mvd_numerator, lambda));
        if (mvd > 2.5e-3f) mvd = 2.5e-3f;
        else if (mvd < 37.5e-6f) mvd = 37.5e-6f;
        lambda = soft64_div(mvd_numerator, rain64(mvd));
        const float prefix = (org3 * qr_new) / am_r;
        nr_new = rain32(soft64_mul(rain64(prefix), rain_pow(lambda, 3.0f)));
        r.qr[idx] = qr_new;
        r.nr[idx] = nr_new;
    }
    if (p.accumulate_surface) r.rainncv[column] += exported;
    else r.rainncv[column] = exported;
    r.rainnc[column] += exported;
}
