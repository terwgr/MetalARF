// Current aerosol cloud fallout with a retained cloud-water tendency.
// The ordinary in-place entry points remain in their existing source file.
#include <metal_stdlib>
#include "thompson_aerosol_exner_libm.metalh"
#include "soft_libm64.metalh"
#include "thompson_aerosol_common.metalh"
#include "thompson_aerosol_cloud_tendency_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline size_t cloud_tendency_idx3(int k, int j, int i, int ny, int nx) {
    return (size_t(k) * size_t(ny) + size_t(j)) * size_t(nx) + size_t(i);
}

template <typename Resources, typename Parameters>
static inline float2 cloud_tendency_velocity(
    constant Resources &r,
    constant Parameters &p,
    size_t idx,
    float cloud_mass,
    float cloud_number,
    float refreshed_density,
    float rho_not,
    bool rain_refreshes_rhof)
{
    if (!(cloud_mass > p.r1 && r.vertical_velocity[idx] < p.sed_w_limit))
        return float2(0.0f);
    const int nu_c = thompson_aa_nu_c(cloud_number);
    const float lambda_arg = thompson_aa_div(
        thompson_aa_mul(
            thompson_aa_mul(
                thompson_aa_mul(cloud_number, p.am_r), r.ccg2[nu_c]),
            r.ocg1[nu_c]),
        cloud_mass);
    const float lambda_f32 = as_type<float>(thompson_exner_powf_bits(
        as_type<uint>(lambda_arg), as_type<uint>(p.obmr)));
    const ulong lambda64 = soft64_from_f32_bits(as_type<uint>(lambda_f32));
    const ulong inverse_lambda64 = soft64_div(0x3ff0000000000000ul, lambda64);
    const float velocity_density = rain_refreshes_rhof
        ? refreshed_density : r.reference_density[idx];
    const float rhof = thompson_aa_sqrtf_rn(thompson_aa_div(rho_not, velocity_density));
    const float mass_prefix = thompson_aa_mul(
        thompson_aa_mul(thompson_aa_mul(rhof, p.av_c), r.ccg5[nu_c]),
        r.ocg2[nu_c]);
    const float number_prefix = thompson_aa_mul(
        thompson_aa_mul(thompson_aa_mul(rhof, p.av_c), r.ccg4[nu_c]),
        r.ocg1[nu_c]);
    ulong mass64 = soft64_mul(
        soft64_from_f32_bits(as_type<uint>(mass_prefix)), inverse_lambda64);
    mass64 = soft64_mul(mass64, inverse_lambda64);
    ulong number64 = soft64_mul(
        soft64_from_f32_bits(as_type<uint>(number_prefix)), inverse_lambda64);
    number64 = soft64_mul(number64, inverse_lambda64);
    return float2(as_type<float>(soft64_to_f32_bits(mass64)),
                  as_type<float>(soft64_to_f32_bits(number64)));
}

template <int KMAX, typename Resources, typename Parameters>
static inline void cloud_tendency_column(
    constant Resources &r,
    constant Parameters &p,
    int column)
{
    if (column < 0 || column >= p.ny * p.nx || p.nz < 2 || p.nz > KMAX)
        return;
    const int j = column / p.nx;
    const int i = column - j * p.nx;
    if (p.has_cloud_mask != 0u && r.cloud_active_columns[column] == 0.0f)
        return;

    bool empty = p.dt >= 0.0f && p.dt <= 100000.0f;
    for (int k = 0; empty && k < p.nz; ++k) {
        const size_t idx = cloud_tendency_idx3(k, j, i, p.ny, p.nx);
        const float working = thompson_aa_add(r.qc[idx], thompson_aa_mul(r.qcten[idx], p.dt));
        empty = working >= -1.0f && working <= p.r1
            && isfinite(r.cloud_number_entry[idx])
            && isfinite(r.cloud_number_tendency[idx])
            && r.reference_density[idx] >= 0.00001f && r.reference_density[idx] <= 100.0f
            && r.temperature[idx] >= 100.0f && r.temperature[idx] <= 400.0f
            && r.pressure[idx] >= 1.0f && r.pressure[idx] <= 120000.0f
            && r.qv[idx] >= -1.0f && r.qv[idx] <= 1.0f
            && r.dz[idx] >= 1.0f && r.dz[idx] <= 100000.0f;
    }
    if (empty) {
        const size_t bottom = cloud_tendency_idx3(0, j, i, p.ny, p.nx);
        r.cloud_number_tendency[bottom] = thompson_aa_add(r.cloud_number_tendency[bottom], 0.0f);
        return;
    }

    float density[KMAX];
    float cloud_mass[KMAX];
    float cloud_number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float mass_flux[KMAX];
    float number_flux[KMAX];
    float qc_tendency[KMAX];

    const bool rain_refreshes_rhof = p.has_rain_mask != 0u
        && r.rain_active_columns[column] != 0.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    for (int k = 0; k < p.nz; ++k) {
        const size_t idx = cloud_tendency_idx3(k, j, i, p.ny, p.nx);
        const float qvk = thompson_aa_fmaxf(1.0e-10f, r.qv[idx]);
        density[k] = thompson_aa_div(thompson_aa_mul(0.622f, r.pressure[idx]),
            thompson_aa_mul(thompson_aa_mul(287.04f, r.temperature[idx]),
                            thompson_aa_add(qvk, 0.622f)));
        qc_tendency[k] = r.qcten[idx];
        const float working = thompson_aa_add(r.qc[idx], thompson_aa_mul(r.qcten[idx], p.dt));
        const float held = r.reference_density[idx];
        cloud_mass[k] = working > p.r1 ? thompson_aa_mul(working, held) : p.r1;
        cloud_number[k] = thompson_aa_clamp_nc(thompson_aa_mul(
            thompson_aa_add(r.cloud_number_entry[idx],
                            thompson_aa_mul(r.cloud_number_tendency[idx], p.dt)),
            held));
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;
    }

    int sediment_top = 0;
    float height_agl = 0.0f;
    for (int k = 0; k < p.nz - 1; ++k) {
        const size_t idx = cloud_tendency_idx3(k, j, i, p.ny, p.nx);
        if (cloud_mass[k] > p.r2) sediment_top = k;
        height_agl = thompson_aa_add(height_agl, r.dz[idx]);
        if (height_agl > p.sed_hgt_agl) break;
    }

    for (int k = sediment_top; k >= 0; --k) {
        const size_t idx = cloud_tendency_idx3(k, j, i, p.ny, p.nx);
        const float2 velocity = cloud_tendency_velocity(
            r, p, idx, cloud_mass[k], cloud_number[k], density[k],
            rho_not, rain_refreshes_rhof);
        mass_velocity[k] = velocity.x;
        number_velocity[k] = velocity.y;
    }

    for (int k = p.nz - 1; k >= 0; --k) {
        mass_flux[k] = thompson_aa_mul(mass_velocity[k], cloud_mass[k]);
        number_flux[k] = thompson_aa_mul(number_velocity[k], cloud_number[k]);
    }
    // One WRF pass only. Level nz-1 is never a sedimentation update; it is
    // only the upper flux value consumed by k <= nz-2.
    for (int k = sediment_top; k >= 0; --k) {
        const size_t idx = cloud_tendency_idx3(k, j, i, p.ny, p.nx);
        const float odzq = thompson_aa_div(1.0f, r.dz[idx]);
        const float orho = thompson_aa_div(1.0f, density[k]);
        const float mass_divergence = thompson_aa_sub(mass_flux[k + 1], mass_flux[k]);
        const float number_divergence = thompson_aa_sub(number_flux[k + 1], number_flux[k]);
        qc_tendency[k] = thompson_aa_add(qc_tendency[k],
            thompson_aa_mul(thompson_aa_mul(mass_divergence, odzq), orho));
        r.cloud_number_tendency[idx] = thompson_aa_add(
            r.cloud_number_tendency[idx],
            thompson_aa_mul(thompson_aa_mul(number_divergence, odzq), orho));
        cloud_mass[k] = thompson_aa_fmaxf(p.r1, thompson_aa_add(cloud_mass[k],
            thompson_aa_mul(thompson_aa_mul(mass_divergence, odzq), p.dt)));
        cloud_number[k] = thompson_aa_fmaxf(p.nc_sed_floor, thompson_aa_add(cloud_number[k],
            thompson_aa_mul(thompson_aa_mul(number_divergence, odzq), p.dt)));
    }

    for (int k = 0; k < p.nz; ++k) {
        const size_t idx = cloud_tendency_idx3(k, j, i, p.ny, p.nx);
        r.qcten[idx] = qc_tendency[k];
    }
}

kernel void thompson_aa_cloud_sediment_64_tendency(
    constant ThompsonAaCloudSediment64TendencyResources &r [[buffer(0)]],
    constant ThompsonAaCloudSediment64TendencyParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= uint(p.ny * p.nx)) return;
    cloud_tendency_column<64>(r, p, int(column));
}

kernel void thompson_aa_cloud_sediment_256_tendency(
    constant ThompsonAaCloudSediment256TendencyResources &r [[buffer(0)]],
    constant ThompsonAaCloudSediment256TendencyParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= uint(p.ny * p.nx)) return;
    cloud_tendency_column<256>(r, p, int(column));
}
