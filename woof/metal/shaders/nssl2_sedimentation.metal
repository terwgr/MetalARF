// WRF v4.6.1 NSSL option-18 serial-column sedimentation lowering.
#include <metal_stdlib>
#include "nssl2_sedimentation_bindings.metalh"
#include "nssl2_soft64.metalh"
using namespace metal;
#pragma clang fp contract(off)
static inline float nssl_div_rn(float a, float b) { return a / b; }
#define IDX3(k,j,i) ((k) * ny * nx + (j) * nx + (i))

template <int KMAX>
static inline void nssl2_cloud_sediment_impl(
    const device float *  air_density,
    const device float *  temperature_k,
    device float *  qc,
    device float *  qndrop,
    const device float *  dz,
    device float *  cloud_surface_export,
    float dt, int nz, int ny, int nx, int column_idx)
{
    const int column = column_idx;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float cloud[KMAX];
    float number[KMAX];
    float velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];

    const float pi = 3.14159265358979323846f;
    const float minimum_mass = 1000.0f * 0.523599f
        * (4.0e-6f * 4.0e-6f * 4.0e-6f);
    const float maximum_mass = 1000.0f * 0.523599f
        * (120.0e-6f * 120.0e-6f * 120.0e-6f);
    const float mass_to_diameter = 6.0f / (pi * 1000.0f);
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = (k * ny * nx + j * nx + i);
        rho[k] = air_density[idx];
        cloud[k] = qc[idx];
        number[k] = qndrop[idx];
        velocity[k] = 0.0f;

        const float positive_cloud = fmax(cloud[k], 0.0f);
        if (positive_cloud > 1.0e-13f) {
            const float positive_number = fmax(number[k], 0.0f);
            const float effective_number = positive_number > 1.0e-8f
                ? positive_number
                : fmax(1.0e-8f,
                        nssl_div_rn(rho[k] * positive_cloud, maximum_mass));
            const float particle_mass = fmin(
                maximum_mass,
                fmax(minimum_mass,
                      positive_cloud * rho[k] / effective_number));
            const float diameter = pow(
                particle_mass * mass_to_diameter, 1.0f / 3.0f);
            const float radius = 0.5f * diameter;
            const float temperature = temperature_k[idx];
            const float viscosity = 1.832e-5f
                * (416.16f / (temperature + 120.0f))
                * pow(nssl_div_rn(temperature, 296.0f), 1.5f);
            if (viscosity > 0.0f) {
                velocity[k] = fmin(
                    70.0f,
                    2.0f * 9.8f * 1000.0f * radius * radius
                        / (9.0f * viscosity));
            }
        }
        maximum_courant_rate = fmax(
            maximum_courant_rate, velocity[k] / dz[idx]);
    }

    if (maximum_courant_rate == 0.0f) {
        cloud_surface_export[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(nssl_div_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        for (int k = 0; k < nz; ++k) {
            mass_flux[k] = cloud[k] * velocity[k] * rho[k];
            number_flux[k] = number[k] * velocity[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;

        surface_mean_flux += cloud[0] * velocity[0] * dt_fraction;

        for (int k = 0; k < nz; ++k) {
            const size_t idx = (k * ny * nx + j * nx + i);
            const float inverse_dz = 1.0f / dz[idx];
            cloud[k] += dt_substep * inverse_dz / rho[k]
                * (mass_flux[k + 1] - mass_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = (k * ny * nx + j * nx + i);
        qc[idx] = cloud[k];
        qndrop[idx] = number[k];
    }
    cloud_surface_export[column] = dt * rho[0] * surface_mean_flux;
}

static inline float nssl2_rain_z(
    float q, float number, float rho)
{
    if (!(q > 1.0e-12f) || !(number > 1.0e-15f)) return 0.0f;

    const float minimum_volume =
        0.523599f * (80.0e-6f * 80.0e-6f * 80.0e-6f);
    const float maximum_volume =
        0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f);
    float mean_volume = rho * q / (1000.0f * number);
    float effective_number = number;
    if (mean_volume < minimum_volume || mean_volume > maximum_volume) {
        mean_volume = fmin(maximum_volume,
                            fmax(minimum_volume, mean_volume));
        effective_number = rho * q / (1000.0f * mean_volume);
    }
    const float z_factor =
        (6.0f / (3.14159265358979323846f * 1000.0f))
        * (6.0f / (3.14159265358979323846f * 1000.0f));
    return 120.0f * rho * rho * q * q / effective_number * z_factor;
}

template <int KMAX>
static inline void nssl2_rain_sediment_impl(
    const device float *  air_density,
    device float *  qr,
    device float *  qnr,
    const device float *  dz,
    device float *  rainnc,
    device float *  rainncv,
    float dt, int nz, int ny, int nx, int column_idx)
{
    const int column = column_idx;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float rain[KMAX];
    float number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float z_velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];
    float z_flux[KMAX + 1];
    float mass_number_flux[KMAX + 1];
    float z_initial[KMAX];
    float z_advected[KMAX];
    float number_mass_weighted[KMAX];

    const float pi = 3.14159265358979323846f;
    const float minimum_volume =
        0.523599f * (80.0e-6f * 80.0e-6f * 80.0e-6f);
    const float configured_maximum_volume =
        0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f);
    const float maximum_speed_volume =
        configured_maximum_volume / (64.0f / 6.0f);
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = (k * ny * nx + j * nx + i);
        rho[k] = air_density[idx];
        rain[k] = qr[idx];
        number[k] = qnr[idx];
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;
        z_velocity[k] = 0.0f;

        const float positive_rain = fmax(rain[k], 0.0f);
        if (positive_rain > 1.0e-12f) {
            const float local_number = fmax(number[k], 0.0f);
            float mean_volume = rho[k] * positive_rain
                / (1000.0f * fmax(1.0e-11f, local_number));
            if (mean_volume > maximum_speed_volume) {
                mean_volume = maximum_speed_volume;
            } else if (mean_volume < minimum_volume) {
                mean_volume = minimum_volume;
            }

            const float diameter = pow(
                nssl_div_rn((6.0f / pi) * mean_volume, (3.0f * 2.0f * 1.0f)),
                1.0f / 3.0f);
            const float density_factor = sqrt(
                1.225f * fmin(20.0f, 1.0f / rho[k]));
            const float speed_base = 1.0f + 516.575f * diameter;
            float vm = density_factor * 10.0f
                * (1.0f - pow(speed_base, -4.0f));
            float vn = density_factor * 10.0f
                * (1.0f - pow(speed_base, -1.0f));
            float vz = density_factor * 10.0f
                * (1.0f - pow(speed_base, -7.0f));
            if (vn > vm || (vm > vz && vz > 0.0f)) {
                vm = fmax(vm, vn);
                vz = fmax(vz, vm);
            }
            mass_velocity[k] = fmin(70.0f, fmin(150.0f, vm));
            number_velocity[k] = fmin(70.0f, fmin(150.0f, vn));
            z_velocity[k] = fmin(70.0f, fmin(150.0f, vz));
        }
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmax(
            maximum_courant_rate, mass_velocity[k] * inverse_dz);
        maximum_courant_rate = fmax(
            maximum_courant_rate, number_velocity[k] * inverse_dz);
        maximum_courant_rate = fmax(
            maximum_courant_rate, z_velocity[k] * inverse_dz);
    }

    if (maximum_courant_rate == 0.0f) {
        rainncv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(nssl_div_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        // Diagnose the pre-fallout reflectivity moment and preserve the
        // pre-fallout number for the parallel mass-weighted correction.
        for (int k = 0; k < nz; ++k) {
            z_initial[k] = nssl2_rain_z(rain[k], number[k], rho[k]);
            z_advected[k] = z_initial[k];
            number_mass_weighted[k] = number[k];
            mass_flux[k] = rain[k] * mass_velocity[k] * rho[k];
            number_flux[k] = number[k] * number_velocity[k];
            z_flux[k] = z_initial[k] * z_velocity[k];
            mass_number_flux[k] = number[k] * mass_velocity[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;
        z_flux[nz] = 0.0f;
        mass_number_flux[nz] = 0.0f;

        surface_mean_flux +=
            rain[0] * mass_velocity[0] * dt_fraction;

        // fallout1d computes every flux first, then updates every level.
        for (int k = 0; k < nz; ++k) {
            const size_t idx = (k * ny * nx + j * nx + i);
            const float inverse_dz = 1.0f / dz[idx];
            rain[k] += dt_substep * inverse_dz / rho[k]
                * (mass_flux[k + 1] - mass_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            z_advected[k] += dt_substep * inverse_dz
                * (z_flux[k + 1] - z_flux[k]);
            number_mass_weighted[k] += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        // calcnfromz1d uses binary64 temporaries before REAL(Nz) storage.
        for (int k = 0; k < nz; ++k) {
            if (z_advected[k] > 0.0f) {
                const float diagnosed =
                    nssl2_rain_z(rain[k], number[k], rho[k]);
                if (diagnosed > z_advected[k]
                        && z_advected[k] > z_initial[k]) {
                    const Nssl2Soft64 z_factor =
                        Nssl2Soft64(6.0f / (pi * 1000.0f))
                        * Nssl2Soft64(6.0f / (pi * 1000.0f));
                    const Nssl2Soft64 reconstructed =
                        Nssl2Soft64(120.0f) * Nssl2Soft64(rho[k])
                        * Nssl2Soft64(rho[k]) * Nssl2Soft64(rain[k])
                        * Nssl2Soft64(rain[k])
                        / (Nssl2Soft64(z_advected[k]) / z_factor);
                    const float reconstructed_real = (float)reconstructed;
                    number[k] = fmax(
                        fmin(reconstructed_real,
                              number_mass_weighted[k]),
                        number[k]);
                } else {
                    number[k] = fmax(number_mass_weighted[k], number[k]);
                }
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = (k * ny * nx + j * nx + i);
        qr[idx] = rain[k];
        qnr[idx] = number[k];
    }
    const float exported = dt * rho[0] * surface_mean_flux;
    rainncv[column] = exported;
    rainnc[column] += exported;
}

template <int KMAX>
static inline void nssl2_snow_sediment_impl(
    const device float *  air_density,
    device float *  qs,
    device float *  qns,
    const device float *  dz,
    device float *  snownc,
    device float *  snowncv,
    float dt, int nz, int ny, int nx, int column_idx)
{
    const int column = column_idx;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float snow[KMAX];
    float number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float z_velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];
    float mass_number_flux[KMAX + 1];
    float number_mass_weighted[KMAX];

    const float minimum_volume =
        0.523599f * (0.01e-3f * 0.01e-3f * 0.01e-3f);
    const float maximum_volume =
        0.523599f * (10.0e-3f * 10.0e-3f * 10.0e-3f);
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = (k * ny * nx + j * nx + i);
        rho[k] = air_density[idx];
        snow[k] = qs[idx];
        number[k] = qns[idx];
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;
        z_velocity[k] = 0.0f;

        const float positive_snow = fmax(snow[k], 0.0f);
        if (positive_snow > 1.0e-13f) {
            const float local_number = fmax(number[k], 0.0f);
            float mean_volume = rho[k] * positive_snow
                / (100.0f * fmax(1.0e-9f, local_number));
            mean_volume = fmin(
                maximum_volume, fmax(minimum_volume, mean_volume));
            const float density_factor = sqrt(
                1.225f * fmin(20.0f, 1.0f / rho[k]));
            const float size_factor = pow(mean_volume, 0.14f);
            mass_velocity[k] = fmin(
                70.0f, 11.9495f * density_factor * size_factor);
            number_velocity[k] = fmin(
                70.0f, 7.02909f * density_factor * size_factor);
            z_velocity[k] = fmin(
                70.0f, 13.3436f * density_factor * size_factor);
        }
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmax(
            maximum_courant_rate, mass_velocity[k] * inverse_dz);
        maximum_courant_rate = fmax(
            maximum_courant_rate, number_velocity[k] * inverse_dz);
        maximum_courant_rate = fmax(
            maximum_courant_rate, z_velocity[k] * inverse_dz);
    }

    if (maximum_courant_rate == 0.0f) {
        snowncv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(nssl_div_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        for (int k = 0; k < nz; ++k) {
            mass_flux[k] = snow[k] * mass_velocity[k] * rho[k];
            number_flux[k] = number[k] * number_velocity[k];
            mass_number_flux[k] = number[k] * mass_velocity[k];
            number_mass_weighted[k] = number[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;
        mass_number_flux[nz] = 0.0f;

        surface_mean_flux +=
            snow[0] * mass_velocity[0] * dt_fraction;

        for (int k = 0; k < nz; ++k) {
            const size_t idx = (k * ny * nx + j * nx + i);
            const float inverse_dz = 1.0f / dz[idx];
            snow[k] += dt_substep * inverse_dz / rho[k]
                * (mass_flux[k + 1] - mass_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            number_mass_weighted[k] += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        for (int k = 0; k < nz; ++k) {
            number[k] = fmax(number[k], number_mass_weighted[k]);
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = (k * ny * nx + j * nx + i);
        qs[idx] = snow[k];
        qns[idx] = number[k];
    }
    const float exported = dt * rho[0] * surface_mean_flux;
    snowncv[column] = exported;
    snownc[column] += exported;
}

template <int KMAX>
static inline void nssl2_ice_sediment_impl(
    const device float *  air_density,
    device float *  qi,
    device float *  qni,
    const device float *  dz,
    device float *  icenc,
    device float *  icencv,
    float dt, int nz, int ny, int nx, int column_idx)
{
    const int column = column_idx;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float ice[KMAX];
    float number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];
    float mass_number_flux[KMAX + 1];
    float number_mass_weighted[KMAX];

    const float minimum_mass = 6.88e-13f;
    const float maximum_mass = 1.0e-8f;
    const float gamma_1p18 = 0.922766923904419f;
    const float gamma_2p18 = 1.091937899589539f;
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = (k * ny * nx + j * nx + i);
        rho[k] = air_density[idx];
        ice[k] = qi[idx];
        number[k] = qni[idx];
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;

        const float positive_ice = fmax(ice[k], 0.0f);
        if (positive_ice > 1.0e-13f) {
            float local_number = fmax(number[k], 0.0f);
            local_number = fmax(
                local_number, nssl_div_rn(rho[k] * positive_ice, maximum_mass));
            local_number = fmin(
                local_number, nssl_div_rn(rho[k] * positive_ice, minimum_mass));
            const float particle_mass = fmax(
                rho[k] * positive_ice / local_number, minimum_mass);
            const float mean_volume = nssl_div_rn(particle_mass, 900.0f);
            const float density_factor = sqrt(
                1.225f * fmin(20.0f, 1.0f / rho[k]));
            const float tmp = 47.6273f * density_factor
                / pow(1.0f / mean_volume, 0.18333f);
            number_velocity[k] = fmin(70.0f, tmp * gamma_1p18);
            mass_velocity[k] = fmin(70.0f, tmp * gamma_2p18);
        }
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmax(
            maximum_courant_rate, mass_velocity[k] * inverse_dz);
        maximum_courant_rate = fmax(
            maximum_courant_rate, number_velocity[k] * inverse_dz);
    }

    if (maximum_courant_rate == 0.0f) {
        icencv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(nssl_div_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        for (int k = 0; k < nz; ++k) {
            mass_flux[k] = ice[k] * mass_velocity[k] * rho[k];
            number_flux[k] = number[k] * number_velocity[k];
            mass_number_flux[k] = number[k] * mass_velocity[k];
            number_mass_weighted[k] = number[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;
        mass_number_flux[nz] = 0.0f;

        surface_mean_flux += ice[0] * mass_velocity[0] * dt_fraction;

        for (int k = 0; k < nz; ++k) {
            const size_t idx = (k * ny * nx + j * nx + i);
            const float inverse_dz = 1.0f / dz[idx];
            ice[k] += dt_substep * inverse_dz / rho[k]
                * (mass_flux[k + 1] - mass_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            number_mass_weighted[k] += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        for (int k = 0; k < nz; ++k) {
            number[k] = fmax(number[k], number_mass_weighted[k]);
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = (k * ny * nx + j * nx + i);
        qi[idx] = ice[k];
        qni[idx] = number[k];
    }
    const float exported = dt * rho[0] * surface_mean_flux;
    icencv[column] = exported;
    icenc[column] += exported;
}

static inline float nssl2_gamma_lookup(float argument)
{
    const Nssl2Soft64 hundred(100.0f);
    const Nssl2Soft64 hundredth =
        nssl2_soft64_const(0x3f847ae147ae147bul);
    const Nssl2Soft64 scaled = hundred * Nssl2Soft64(argument);
    const int lower_index = (int)scaled;
    const Nssl2Soft64 lower = hundredth * Nssl2Soft64(lower_index);
    const Nssl2Soft64 fraction = Nssl2Soft64(argument) - lower;
    const Nssl2Soft64 lower_gamma = nssl2_tgamma64(lower);
    const Nssl2Soft64 upper_gamma = nssl2_tgamma64(lower + hundredth);
    return (float)(lower_gamma
        + (upper_gamma - lower_gamma) * fraction * hundred);
}

template <bool CACHED>
static inline float nssl2_velocity_gamma_lookup(
    float argument, const device float *table, int shape)
{
    (void)table;
    (void)shape;
    return nssl2_gamma_lookup(argument);
}

static inline float nssl2_dense_frozen_z(
    float q, float number, float volume, float rho, bool hail)
{
    if (!(q > 1.0e-12f) || !(number > 1.0e-15f)) return 0.0f;

    const float minimum_volume =
        0.523599f * (0.3e-3f * 0.3e-3f * 0.3e-3f);
    const float maximum_diameter = hail ? 40.0e-3f : 20.0e-3f;
    const float maximum_volume = 0.523599f * maximum_diameter
        * maximum_diameter * maximum_diameter;
    float particle_density = hail ? 800.0f : 500.0f;
    if (volume > 0.0f) {
        particle_density = fmin(
            900.0f, fmax(170.0f, rho * q / volume));
    }
    float mean_volume = rho * q / (particle_density * number);
    float effective_number = number;
    if (mean_volume < minimum_volume || mean_volume > maximum_volume) {
        mean_volume = fmin(
            maximum_volume, fmax(minimum_volume, mean_volume));
        effective_number = rho * q / (particle_density * mean_volume);
    }
    const float z_factor =
        (6.0f / (3.14159265358979323846f * 1000.0f))
        * (6.0f / (3.14159265358979323846f * 1000.0f));
    const float moment_ratio = hail ? 8.75f : 20.0f;
    return moment_ratio * rho * rho * q * q
        / effective_number * z_factor;
}

static inline void nssl2_graupel_mm_coefficients(
    float particle_density, thread float *coefficient, thread float *exponent)
{
    const float table_density[9] = {
        50.0f, 150.0f, 250.0f, 350.0f, 450.0f,
        550.0f, 650.0f, 750.0f, 850.0f};
    const float table_coefficient[9] = {
        62.923f, 94.122f, 114.74f, 131.21f, 145.26f,
        157.71f, 168.98f, 179.36f, 189.02f};
    const float table_exponent[9] = {
        0.67819f, 0.63789f, 0.62197f, 0.61240f, 0.60572f,
        0.60066f, 0.59663f, 0.59330f, 0.59048f};

    int index = (int)(nssl_div_rn((particle_density - 50.0f), 100.0f));
    index = max(0, min(8, index));
    if (index < 8) {
        const float fraction = fmax(
            0.0f, 0.01f * (particle_density - table_density[index]));
        *coefficient = table_coefficient[index]
            + fraction * (table_coefficient[index + 1]
                          - table_coefficient[index]);
        *exponent = table_exponent[index]
            + fraction * (table_exponent[index + 1]
                          - table_exponent[index]);
    } else {
        *coefficient = table_coefficient[index];
        *exponent = table_exponent[index];
    }
}

template <int KMAX, bool HAIL, bool CACHED = false>
static inline void nssl2_dense_frozen_sediment_impl(
    const device float *  air_density,
    device float *  qx,
    device float *  qnx,
    device float *  qvolx,
    const device float *  dz,
    device float *  frozennc,
    device float *  frozenncv,
    float dt, int nz, int ny, int nx, int column_idx)
{
    const int column = column_idx;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float graupel[KMAX];
    float number[KMAX];
    float volume[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float z_velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];
    float volume_flux[KMAX + 1];
    float z_flux[KMAX + 1];
    float mass_number_flux[KMAX + 1];
    float z_initial[KMAX];
    float z_advected[KMAX];
    float number_mass_weighted[KMAX];

    const float pi = 3.14159265358979323846f;
    const float minimum_volume =
        0.523599f * (0.3e-3f * 0.3e-3f * 0.3e-3f);
    const float maximum_diameter = HAIL ? 40.0e-3f : 20.0e-3f;
    const float maximum_volume = 0.523599f * maximum_diameter
        * maximum_diameter * maximum_diameter;
    const float shape = HAIL ? 1.0f : 0.0f;
    const float characteristic_factor = pow(
        HAIL ? 24.0f : 6.0f, -1.0f / 3.0f);
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = (k * ny * nx + j * nx + i);
        rho[k] = air_density[idx];
        graupel[k] = qx[idx];
        number[k] = qnx[idx];
        volume[k] = qvolx[idx];
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;
        z_velocity[k] = 0.0f;

        const float positive_graupel = fmax(graupel[k], 0.0f);
        if (positive_graupel > 1.0e-12f) {
            const float minimum_density = HAIL ? 500.0f : 170.0f;
            float particle_density = HAIL ? 800.0f : 500.0f;
            if (volume[k] > rho[k] * 1.0e-15f) {
                particle_density = fmin(
                    900.0f,
                    fmax(minimum_density,
                          rho[k] * positive_graupel / volume[k]));
            }
            float mean_volume = rho[k] * positive_graupel
                / (particle_density * fmax(1.0e-9f, number[k]));
            mean_volume = fmin(
                maximum_volume, fmax(minimum_volume, mean_volume));
            const float mass_diameter = pow(nssl_div_rn(6.0f * mean_volume, pi),
                                             1.0f / 3.0f);
            const float characteristic_diameter =
                characteristic_factor * mass_diameter;
            float coefficient;
            float exponent;
            nssl2_graupel_mm_coefficients(
                particle_density, &coefficient, &exponent);
            const float density_factor = sqrt(
                1.225f * fmin(20.0f, 1.0f / rho[k]));
            const float base_speed = density_factor * coefficient
                * pow(characteristic_diameter, exponent);
            mass_velocity[k] = base_speed
                * nssl2_velocity_gamma_lookup<CACHED>(4.0f + shape + exponent, nullptr, HAIL ? 1 : 0)
                / nssl2_gamma_lookup(4.0f + shape);
            number_velocity[k] = base_speed
                * nssl2_velocity_gamma_lookup<CACHED>(1.0f + shape + exponent, nullptr, HAIL ? 1 : 0)
                / nssl2_gamma_lookup(1.0f + shape);
            z_velocity[k] = base_speed
                * nssl2_velocity_gamma_lookup<CACHED>(7.0f + shape + exponent, nullptr, HAIL ? 1 : 0)
                / nssl2_gamma_lookup(7.0f + shape);
            if (number_velocity[k] > mass_velocity[k]
                    || (mass_velocity[k] > z_velocity[k]
                        && z_velocity[k] > 0.0f)) {
                mass_velocity[k] = fmax(
                    mass_velocity[k], number_velocity[k]);
                z_velocity[k] = fmax(z_velocity[k], mass_velocity[k]);
            }
            mass_velocity[k] = fmin(70.0f, fmin(150.0f, mass_velocity[k]));
            number_velocity[k] = fmin(
                70.0f, fmin(150.0f, number_velocity[k]));
            z_velocity[k] = fmin(70.0f, fmin(150.0f, z_velocity[k]));
        }
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmax(
            maximum_courant_rate, mass_velocity[k] * inverse_dz);
        maximum_courant_rate = fmax(
            maximum_courant_rate, number_velocity[k] * inverse_dz);
        maximum_courant_rate = fmax(
            maximum_courant_rate, z_velocity[k] * inverse_dz);
    }

    if (maximum_courant_rate == 0.0f) {
        frozenncv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(
            2, (int)(nssl_div_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        for (int k = 0; k < nz; ++k) {
            z_initial[k] = nssl2_dense_frozen_z(
                graupel[k], number[k], volume[k], rho[k], HAIL);
            z_advected[k] = z_initial[k];
            number_mass_weighted[k] = number[k];
            mass_flux[k] = graupel[k] * mass_velocity[k] * rho[k];
            number_flux[k] = number[k] * number_velocity[k];
            volume_flux[k] = volume[k] * mass_velocity[k];
            z_flux[k] = z_initial[k] * z_velocity[k];
            mass_number_flux[k] = number[k] * mass_velocity[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;
        volume_flux[nz] = 0.0f;
        z_flux[nz] = 0.0f;
        mass_number_flux[nz] = 0.0f;

        surface_mean_flux +=
            graupel[0] * mass_velocity[0] * dt_fraction;

        for (int k = 0; k < nz; ++k) {
            const size_t idx = (k * ny * nx + j * nx + i);
            const float inverse_dz = 1.0f / dz[idx];
            graupel[k] += dt_substep * inverse_dz / rho[k]
                * (mass_flux[k + 1] - mass_flux[k]);
            volume[k] += dt_substep * inverse_dz
                * (volume_flux[k + 1] - volume_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            z_advected[k] += dt_substep * inverse_dz
                * (z_flux[k + 1] - z_flux[k]);
            number_mass_weighted[k] += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        for (int k = 0; k < nz; ++k) {
            if (z_advected[k] > 0.0f) {
                const float diagnosed = nssl2_dense_frozen_z(
                    graupel[k], number[k], volume[k], rho[k], HAIL);
                if (diagnosed > z_advected[k]
                        && diagnosed > 0.0f
                        && z_advected[k] > z_initial[k]) {
                    const Nssl2Soft64 z_factor =
                        Nssl2Soft64(6.0f / (pi * 1000.0f))
                        * Nssl2Soft64(6.0f / (pi * 1000.0f));
                    const Nssl2Soft64 moment_ratio(
                        HAIL ? 8.75f : 20.0f);
                    const Nssl2Soft64 reconstructed =
                        moment_ratio * Nssl2Soft64(rho[k])
                        * Nssl2Soft64(rho[k]) * Nssl2Soft64(graupel[k])
                        * Nssl2Soft64(graupel[k])
                        / (Nssl2Soft64(z_advected[k]) / z_factor);
                    const float reconstructed_real = (float)reconstructed;
                    number[k] = fmax(
                        fmin(reconstructed_real,
                              number_mass_weighted[k]),
                        number[k]);
                } else {
                    number[k] = fmax(
                        number_mass_weighted[k], number[k]);
                }
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = (k * ny * nx + j * nx + i);
        qx[idx] = graupel[k];
        qnx[idx] = number[k];
        qvolx[idx] = volume[k];
    }
    const float exported = dt * rho[0] * surface_mean_flux;
    frozenncv[column] = exported;
    frozennc[column] += exported;
}

template <int KMAX> static inline void nssl2_sediment_column(const device float *air_density, const device float *temperature_k, device float *state, const device float *dz, device float *ignored_accumulator, device float *category_export, float dt, int nz, int ny, int nx, int column_idx) {
 const int n=nz*ny*nx, ncol=ny*nx;
 nssl2_rain_sediment_impl<KMAX>(air_density,state+2*n,state+8*n,dz,ignored_accumulator,category_export,dt,nz,ny,nx,column_idx);
 nssl2_ice_sediment_impl<KMAX>(air_density,state+3*n,state+9*n,dz,ignored_accumulator,category_export+ncol,dt,nz,ny,nx,column_idx);
 nssl2_snow_sediment_impl<KMAX>(air_density,state+4*n,state+10*n,dz,ignored_accumulator,category_export+2*ncol,dt,nz,ny,nx,column_idx);
 nssl2_dense_frozen_sediment_impl<KMAX,false>(air_density,state+5*n,state+11*n,state+14*n,dz,ignored_accumulator,category_export+3*ncol,dt,nz,ny,nx,column_idx);
 nssl2_dense_frozen_sediment_impl<KMAX,true>(air_density,state+6*n,state+12*n,state+15*n,dz,ignored_accumulator,category_export+4*ncol,dt,nz,ny,nx,column_idx);
 nssl2_cloud_sediment_impl<KMAX>(air_density,temperature_k,state+n,state+7*n,dz,ignored_accumulator,dt,nz,ny,nx,column_idx);
}
kernel void nssl2_sediment_all_64(constant Nssl2SedimentAll64Resources &r [[buffer(0)]], constant Nssl2SedimentAll64Params &p [[buffer(1)]], uint column [[thread_position_in_grid]]) { if(column>=uint(p.ny*p.nx)) return; nssl2_sediment_column<64>(r.air_density,r.temperature_k,r.state,r.dz,r.ignored_accumulator,r.category_export,p.dt,p.nz,p.ny,p.nx,int(column)); }
kernel void nssl2_sediment_all_128(constant Nssl2SedimentAll128Resources &r [[buffer(0)]], constant Nssl2SedimentAll128Params &p [[buffer(1)]], uint column [[thread_position_in_grid]]) { if(column>=uint(p.ny*p.nx)) return; nssl2_sediment_column<128>(r.air_density,r.temperature_k,r.state,r.dz,r.ignored_accumulator,r.category_export,p.dt,p.nz,p.ny,p.nx,int(column)); }
