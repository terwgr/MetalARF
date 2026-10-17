// Dependency-blocked source translation for packet 72.
//
// Proposed Metal entrypoints are present, but no production signatures or
// generated binding headers exist yet. The serial entrypoints preserve the
// source column traversal; the level-parallel proposal maps eight columns by
// 64 levels. Shared soft64/pow/helper contracts remain unreviewed. Do not
// register or consume these proposals until central review clears every
// dependency.

#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "soft_gfk_pow.metalh"
#include "thompson_aerosol_common.metalh"
#include "thompson_aa_cloud_sediment_64_source_bindings.metalh"
#include "thompson_aa_cloud_sediment_256_source_bindings.metalh"
#include "thompson_aa_cloud_sediment_levels_64_source_bindings.metalh"
#include "thompson_aa_final_phase_cleanup_source_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

// Source thompson_aa_powf_cr is (float)pow((double)x,(double)y), not gfk_pow.
// The exact shared binary64 pow interface must be reviewed separately.

static inline ulong taa_f32_to_f64(uint bits)
{
    return soft64_from_f32_bits(bits);
}
static inline uint taa_f64_to_f32(ulong bits)
{
    return soft64_to_f32_bits(bits);
}
// C fmin on binary64 words: a NaN operand yields the other operand.
static inline ulong thompson_aa_soft64_fmin(ulong a, ulong b)
{
    if (soft64_is_nan(a)) return b;
    if (soft64_is_nan(b)) return a;
    if (soft64_is_zero(a) && soft64_is_zero(b))
        return (a & 0x8000000000000000ul) != 0ul ? a : b;
    return soft64_less(a, b) ? a : b;
}
static inline ulong taa_f64_one() { return 0x3ff0000000000000ul; }
static inline ulong taa_f64_three() { return 0x4008000000000000ul; }

struct TaaCloudResources {
    device float *qc;
    device const float *cloud_number_entry;
    device float *cloud_number_tendency;
    device const float *temperature;
    device const float *pressure;
    device const float *qv;
    device const float *reference_density;
    device const float *rain_active_columns;
    device const float *cloud_active_columns;
    device const float *vertical_velocity;
    device const float *dz;
    device const float *ccg2;
    device const float *ccg4;
    device const float *ccg5;
    device const float *ocg1;
    device const float *ocg2;
    device float *out_mass_velocity;
    device float *out_number_velocity;
    device float *out_cloud_mass;
    device float *out_cloud_number;
};

struct TaaCloudParams {
    float dt;
    float r1;
    float r2;
    float am_r;
    float obmr;
    float av_c;
    float nc_sed_floor;
    float sed_w_limit;
    float sed_hgt_agl;
    int nz;
    int ny;
    int nx;
    uint has_rain_mask;
    uint has_cloud_mask;
    uint diagnostic_mask;
};

static inline size_t taa_idx3(int k, int j, int i, int ny, int nx)
{
    return (size_t(k) * size_t(ny) + size_t(j)) * size_t(nx) + size_t(i);
}

static inline float2 taa_cloud_velocity(
    thread const TaaCloudResources &r,
    thread const TaaCloudParams &p,
    size_t idx,
    float cloud_mass,
    float cloud_number,
    float refreshed_density,
    float rho_not,
    bool rain_refreshes_rhof)
{
    if (cloud_mass <= p.r1 || r.vertical_velocity[idx] >= p.sed_w_limit)
        return float2(0.0f);
    const int nu_c = thompson_aa_nu_c(cloud_number);
    const float lambda_arg = thompson_aa_div(
        thompson_aa_mul(
            thompson_aa_mul(
                thompson_aa_mul(cloud_number, p.am_r), r.ccg2[nu_c]),
            r.ocg1[nu_c]),
        cloud_mass);
    const float lambda_f32 = thompson_aa_powf_cr(lambda_arg, p.obmr);
    const ulong lambda64 = taa_f32_to_f64(as_type<uint>(lambda_f32));
    const ulong inverse_lambda64 = soft64_div(taa_f64_one(), lambda64);
    const float velocity_density = rain_refreshes_rhof
        ? refreshed_density : r.reference_density[idx];
    const float rhof = thompson_aa_sqrtf_rn(rho_not / velocity_density);
    const float mass_prefix = thompson_aa_mul(
        thompson_aa_mul(thompson_aa_mul(rhof, p.av_c), r.ccg5[nu_c]),
        r.ocg2[nu_c]);
    const float number_prefix = thompson_aa_mul(
        thompson_aa_mul(thompson_aa_mul(rhof, p.av_c), r.ccg4[nu_c]),
        r.ocg1[nu_c]);
    ulong mass64 = soft64_mul(
        taa_f32_to_f64(as_type<uint>(mass_prefix)), inverse_lambda64);
    mass64 = soft64_mul(mass64, inverse_lambda64);
    ulong number64 = soft64_mul(
        taa_f32_to_f64(as_type<uint>(number_prefix)), inverse_lambda64);
    number64 = soft64_mul(number64, inverse_lambda64);
    return float2(as_type<float>(taa_f64_to_f32(mass64)),
                  as_type<float>(taa_f64_to_f32(number64)));
}

// Serial-column transcription of thompson_aa_cloud_sediment_impl. The five
// moments are readonly resources supplied from the centrally-owned shared
// Thompson aerosol tables. Optional mask and diagnostic buffers are bound to
// safe dummy arrays when their corresponding flags are clear.
template <int KMAX>
static inline void thompson_aa_cloud_sediment_column(
    thread const TaaCloudResources &r,
    thread const TaaCloudParams &p,
    int column)
{
    if (column < 0 || column >= p.ny * p.nx || p.nz < 2 || p.nz > KMAX)
        return;
    const int j = column / p.nx;
    const int i = column - j * p.nx;
    if (p.has_cloud_mask != 0u && r.cloud_active_columns[column] == 0.0f)
        return;

    float density[KMAX];
    float cloud_mass[KMAX];
    float cloud_number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float mass_flux[KMAX];
    float number_flux[KMAX];
    float qc_tendency[KMAX];
    float qc_initial[KMAX];

    const bool rain_refreshes_rhof = p.has_rain_mask != 0u
        && r.rain_active_columns[column] != 0.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    for (int k = 0; k < p.nz; ++k) {
        const size_t idx = taa_idx3(k, j, i, p.ny, p.nx);
        const float qvk = thompson_aa_fmaxf(1.0e-10f, r.qv[idx]);
        density[k] = 0.622f * r.pressure[idx]
            / (287.04f * r.temperature[idx] * (qvk + 0.622f));
        qc_initial[k] = r.qc[idx];
        qc_tendency[k] = 0.0f;
        const float held = r.reference_density[idx];
        cloud_mass[k] = r.qc[idx] > p.r1 ? r.qc[idx] * held : p.r1;
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
        const size_t idx = taa_idx3(k, j, i, p.ny, p.nx);
        if (cloud_mass[k] > p.r2) sediment_top = k;
        height_agl = thompson_aa_add(height_agl, r.dz[idx]);
        if (height_agl > p.sed_hgt_agl) break;
    }

    for (int k = sediment_top; k >= 0; --k) {
        const size_t idx = taa_idx3(k, j, i, p.ny, p.nx);
        const float2 velocity = taa_cloud_velocity(
            r, p, idx, cloud_mass[k], cloud_number[k], density[k],
            rho_not, rain_refreshes_rhof);
        mass_velocity[k] = velocity.x;
        number_velocity[k] = velocity.y;
    }

    for (int k = p.nz - 1; k >= 0; --k) {
        mass_flux[k] = mass_velocity[k] * cloud_mass[k];
        number_flux[k] = number_velocity[k] * cloud_number[k];
    }
    if ((p.diagnostic_mask & 1u) != 0u || (p.diagnostic_mask & 2u) != 0u
            || (p.diagnostic_mask & 4u) != 0u) {
        for (int k = 0; k < p.nz; ++k) {
            const size_t idx = taa_idx3(k, j, i, p.ny, p.nx);
            if ((p.diagnostic_mask & 1u) != 0u) r.out_mass_velocity[idx] = mass_velocity[k];
            if ((p.diagnostic_mask & 2u) != 0u) r.out_number_velocity[idx] = number_velocity[k];
            if ((p.diagnostic_mask & 4u) != 0u) r.out_cloud_mass[idx] = cloud_mass[k];
        }
    }

    // One WRF pass only. Level nz-1 is never a sedimentation update; it is
    // only the upper flux value consumed by k <= nz-2.
    for (int k = sediment_top; k >= 0; --k) {
        const size_t idx = taa_idx3(k, j, i, p.ny, p.nx);
        const float odzq = 1.0f / r.dz[idx];
        const float orho = 1.0f / density[k];
        const float mass_divergence = mass_flux[k + 1] - mass_flux[k];
        const float number_divergence = number_flux[k + 1] - number_flux[k];
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
        const size_t idx = taa_idx3(k, j, i, p.ny, p.nx);
        r.qc[idx] = thompson_aa_add(qc_initial[k],
                                    thompson_aa_mul(qc_tendency[k], p.dt));
        if ((p.diagnostic_mask & 8u) != 0u) r.out_cloud_number[idx] = cloud_number[k];
    }
}

struct TaaCleanupResources {
    device float *qc;
    device float *qi;
    device float *ni;
    device float *temperature;
    device const float *cloud_number_entry;
    device const float *ice_number_entry;
    device float *cloud_number_tendency;
    device const float *pressure;
    device const float *qv;
};

// Source-order transcription of thompson_aa_final_phase_cleanup. The shared
// bound helper and all powers remain external prerequisites; this function
// must not be emitted as a kernel before those contracts clear review.
static inline void thompson_aa_final_phase_cleanup_item(
    thread const TaaCleanupResources &r, float dt, int idx,
    float r1, float r2, float am_i, float number_ceiling)
{
    const float temp0 = r.temperature[idx];
    const float qv0 = thompson_aa_fmaxf(1.0e-10f, r.qv[idx]);
    const float rho = 0.622f * r.pressure[idx]
        / (287.04f * temp0 * (qv0 + 0.622f));
    const float inverse_cp = 1.0f / (1004.0f * (1.0f + 0.887f * qv0));
    const float odt = 1.0f / dt;
    if (temp0 > 273.15f && r.qi[idx] > 0.0f) {
        const float transferred = thompson_aa_fmaxf(0.0f, r.qi[idx]);
        r.qc[idx] += transferred;
        r.cloud_number_tendency[idx] = thompson_aa_add(
            r.cloud_number_tendency[idx],
            thompson_aa_mul(r.ice_number_entry[idx], odt));
        r.qi[idx] = 0.0f;
        r.ni[idx] = 0.0f;
        r.temperature[idx] -= 334000.0f * inverse_cp * transferred;
    }
    if (temp0 < 235.16f && r.qc[idx] > 0.0f) {
        const float transferred = thompson_aa_fmaxf(0.0f, r.qc[idx]);
        const float latent_vapor = 2.5e6f
            + (2106.0f - 4218.0f) * (temp0 - 273.15f);
        const float latent_fusion = 2.834e6f - latent_vapor;
        const float xnc = thompson_aa_add(r.cloud_number_entry[idx],
            thompson_aa_mul(r.cloud_number_tendency[idx], dt));
        r.qc[idx] = 0.0f;
        r.qi[idx] += transferred;
        r.ni[idx] += xnc;
        r.cloud_number_tendency[idx] = thompson_aa_sub(
            r.cloud_number_tendency[idx], thompson_aa_mul(xnc, odt));
        r.temperature[idx] += latent_fusion * inverse_cp * transferred;
    }
    if (r.qc[idx] <= r1) r.qc[idx] = 0.0f;
    if (r.qi[idx] <= r1) {
        r.qi[idx] = 0.0f;
        r.ni[idx] = 0.0f;
    } else if (r.qi[idx] * rho > r1) {
        thompson_aa_bound_ice_number(
            r.qi[idx] * rho, rho, &r.ni[idx], number_ceiling);
    } else {
        const float qi_local = r.qi[idx];
        const float ni_local = thompson_aa_fmaxf(
            thompson_aa_div(r2, rho), r.ni[idx]);
        const float lambda_arg = thompson_aa_div(
            thompson_aa_mul(thompson_aa_mul(am_i, 6.0f), ni_local),
            qi_local);
        const float lambda32 = thompson_aa_powf_cr(lambda_arg,
                                                    3.333333433e-01f);
        ulong lami = taa_f32_to_f64(as_type<uint>(lambda32));
        const ulong inv_lami = soft64_div(taa_f64_one(), lami);
        const float xdi = as_type<float>(taa_f64_to_f32(soft64_mul(
            taa_f32_to_f64(as_type<uint>(4.0f)), inv_lami)));
        if (xdi < 5.0e-6f) {
            lami = taa_f32_to_f64(as_type<uint>(thompson_aa_div(4.0f, 5.0e-6f)));
        } else if (xdi > 300.0e-6f) {
            lami = taa_f32_to_f64(as_type<uint>(thompson_aa_div(4.0f, 300.0e-6f)));
        }
        const ulong cube = soft64_plm_pow(lami, taa_f64_three());
        const float mass_ratio32 = thompson_aa_div(
            thompson_aa_mul(1.0f / 6.0f, qi_local), am_i);
        const ulong mass_ratio = taa_f32_to_f64(as_type<uint>(mass_ratio32));
        const ulong candidate = soft64_mul(mass_ratio, cube);
        const ulong cap = soft64_div(
            taa_f32_to_f64(as_type<uint>(number_ceiling)),
            taa_f32_to_f64(as_type<uint>(rho)));
        r.ni[idx] = as_type<float>(taa_f64_to_f32(
            thompson_aa_soft64_fmin(candidate, cap)));
    }
}

// The CUDA optional-argument variants select these flags and output buffers;
// the numerical body stays common. They remain proposal entrypoints until the
// declared helper/math interfaces and binding layout receive central review.
template <int KMAX, typename Resources, typename Parameters>
static inline void taa_cloud_entry(constant Resources &r,
                                   constant Parameters &params,
                                   uint column)
{
    TaaCloudResources view;
    view.qc = r.qc;
    view.cloud_number_entry = r.cloud_number_entry;
    view.cloud_number_tendency = r.cloud_number_tendency;
    view.temperature = r.temperature;
    view.pressure = r.pressure;
    view.qv = r.qv;
    view.reference_density = r.reference_density;
    view.rain_active_columns = r.rain_active_columns;
    view.cloud_active_columns = r.cloud_active_columns;
    view.vertical_velocity = r.vertical_velocity;
    view.dz = r.dz;
    view.ccg2 = r.ccg2;
    view.ccg4 = r.ccg4;
    view.ccg5 = r.ccg5;
    view.ocg1 = r.ocg1;
    view.ocg2 = r.ocg2;
    view.out_mass_velocity = r.out_mass_velocity;
    view.out_number_velocity = r.out_number_velocity;
    view.out_cloud_mass = r.out_cloud_mass;
    view.out_cloud_number = r.out_cloud_number;
    TaaCloudParams p;
    p.dt = params.dt;
    p.r1 = params.r1;
    p.r2 = params.r2;
    p.am_r = params.am_r;
    p.obmr = params.obmr;
    p.av_c = params.av_c;
    p.nc_sed_floor = params.nc_sed_floor;
    p.sed_w_limit = params.sed_w_limit;
    p.sed_hgt_agl = params.sed_hgt_agl;
    p.nz = params.nz;
    p.ny = params.ny;
    p.nx = params.nx;
    p.has_rain_mask = params.has_rain_mask;
    p.has_cloud_mask = params.has_cloud_mask;
    p.diagnostic_mask = params.diagnostic_mask;
    thompson_aa_cloud_sediment_column<KMAX>(view, p, int(column));
}

kernel void thompson_aa_cloud_sediment_64_source(
    constant ThompsonAaCloudSediment64SourceResources &r [[buffer(0)]],
    constant ThompsonAaCloudSediment64SourceParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    taa_cloud_entry<64>(r, p, column);
}

kernel void thompson_aa_cloud_sediment_256_source(
    constant ThompsonAaCloudSediment256SourceResources &r [[buffer(0)]],
    constant ThompsonAaCloudSediment256SourceParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]])
{
    taa_cloud_entry<256>(r, p, column);
}

// Direct 8-column x 64-level mapping of the CUDA level-parallel export.
// Threadgroup storage and barriers match its shared-column staging; the
// serial 64-column entrypoint above remains the fallback mapping when this
// launch shape is not available.
kernel void thompson_aa_cloud_sediment_levels_64_source(
    constant ThompsonAaCloudSedimentLevels64SourceResources &r [[buffer(0)]],
    constant ThompsonAaCloudSedimentLevels64SourceParams &p [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]])
{
    if (p.nz < 2 || p.nz > 64) return;
    const uint c = lane & 7u;
    const uint k = lane >> 3;
    const int column = int(group * 8u + c);
    const int j = column / p.nx;
    const int i = column - j * p.nx;
    const bool live = column < p.ny * p.nx
        && (p.has_cloud_mask == 0 || r.cloud_active_columns[column] != 0.0f);
    threadgroup float depth[64][8];
    threadgroup float mass[64][8];
    threadgroup float mass_flux[64][8];
    threadgroup float number_flux[64][8];
    threadgroup int top[8];
    float density, cloud_mass, cloud_number, mass_velocity, number_velocity;
    float qc_tendency, qc_initial;
    const bool rain_refreshes_rhof = live && p.has_rain_mask != 0
        && r.rain_active_columns[column] != 0.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);

    if (live && k < uint(p.nz)) {
        const size_t idx = taa_idx3(int(k), j, i, p.ny, p.nx);
        const float qvk = thompson_aa_fmaxf(1.0e-10f, r.qv[idx]);
        density = 0.622f * r.pressure[idx]
            / (287.04f * r.temperature[idx] * (qvk + 0.622f));
        qc_initial = r.qc[idx];
        qc_tendency = 0.0f;
        cloud_mass = r.qc[idx] > p.r1
            ? r.qc[idx] * r.reference_density[idx] : p.r1;
        cloud_number = thompson_aa_clamp_nc(thompson_aa_mul(
            thompson_aa_add(r.cloud_number_entry[idx],
                thompson_aa_mul(r.cloud_number_tendency[idx], p.dt)),
            r.reference_density[idx]));
        mass_velocity = 0.0f;
        number_velocity = 0.0f;
        depth[k][c] = r.dz[idx];
        mass[k][c] = cloud_mass;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k == 0u) {
        int sediment_top = 0;
        float height_agl = 0.0f;
        if (live) {
            for (int level = 0; level < p.nz - 1; ++level) {
                if (mass[level][c] > p.r2) sediment_top = level;
                height_agl = thompson_aa_add(height_agl, depth[level][c]);
                if (height_agl > p.sed_hgt_agl) break;
            }
        }
        top[c] = sediment_top;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int sediment_top = top[c];
    if (live && int(k) <= sediment_top) {
        const size_t idx = taa_idx3(int(k), j, i, p.ny, p.nx);
        TaaCloudResources view;
        view.qc = r.qc;
        view.cloud_number_entry = r.cloud_number_entry;
        view.cloud_number_tendency = r.cloud_number_tendency;
        view.temperature = r.temperature;
        view.pressure = r.pressure;
        view.qv = r.qv;
        view.reference_density = r.reference_density;
        view.rain_active_columns = r.rain_active_columns;
        view.cloud_active_columns = r.cloud_active_columns;
        view.vertical_velocity = r.vertical_velocity;
        view.dz = r.dz;
        view.ccg2 = r.ccg2;
        view.ccg4 = r.ccg4;
        view.ccg5 = r.ccg5;
        view.ocg1 = r.ocg1;
        view.ocg2 = r.ocg2;
        view.out_mass_velocity = r.qc;
        view.out_number_velocity = r.qc;
        view.out_cloud_mass = r.qc;
        view.out_cloud_number = r.qc;
        TaaCloudParams cp;
        cp.dt = p.dt; cp.r1 = p.r1; cp.r2 = p.r2; cp.am_r = p.am_r;
        cp.obmr = p.obmr; cp.av_c = p.av_c; cp.nc_sed_floor = p.nc_sed_floor;
        cp.sed_w_limit = p.sed_w_limit; cp.sed_hgt_agl = p.sed_hgt_agl;
        cp.nz = p.nz; cp.ny = p.ny; cp.nx = p.nx;
        cp.has_rain_mask = p.has_rain_mask; cp.has_cloud_mask = p.has_cloud_mask;
        cp.diagnostic_mask = 0u;
        const float2 velocity = taa_cloud_velocity(
            view, cp, idx, cloud_mass, cloud_number, density,
            rho_not, rain_refreshes_rhof);
        mass_velocity = velocity.x;
        number_velocity = velocity.y;
    }
    if (live && k < uint(p.nz)) {
        mass_flux[k][c] = mass_velocity * cloud_mass;
        number_flux[k][c] = number_velocity * cloud_number;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live && int(k) <= sediment_top) {
        const size_t idx = taa_idx3(int(k), j, i, p.ny, p.nx);
        const float odzq = 1.0f / r.dz[idx];
        const float orho = 1.0f / density;
        const float mass_divergence = mass_flux[k + 1u][c] - mass_flux[k][c];
        const float number_divergence = number_flux[k + 1u][c] - number_flux[k][c];
        qc_tendency = thompson_aa_add(qc_tendency,
            thompson_aa_mul(thompson_aa_mul(mass_divergence, odzq), orho));
        r.cloud_number_tendency[idx] = thompson_aa_add(
            r.cloud_number_tendency[idx],
            thompson_aa_mul(thompson_aa_mul(number_divergence, odzq), orho));
        cloud_mass = thompson_aa_fmaxf(p.r1, thompson_aa_add(cloud_mass,
            thompson_aa_mul(thompson_aa_mul(mass_divergence, odzq), p.dt)));
        cloud_number = thompson_aa_fmaxf(p.nc_sed_floor, thompson_aa_add(cloud_number,
            thompson_aa_mul(thompson_aa_mul(number_divergence, odzq), p.dt)));
    }
    if (live && k < uint(p.nz)) {
        const size_t idx = taa_idx3(int(k), j, i, p.ny, p.nx);
        r.qc[idx] = thompson_aa_add(qc_initial,
                                    thompson_aa_mul(qc_tendency, p.dt));
    }
}

kernel void thompson_aa_final_phase_cleanup_source(
    constant ThompsonAaFinalPhaseCleanupSourceResources &r [[buffer(0)]],
    constant ThompsonAaFinalPhaseCleanupSourceParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= uint(p.size)) return;
    TaaCleanupResources view;
    view.qc = r.qc;
    view.qi = r.qi;
    view.ni = r.ni;
    view.temperature = r.temperature;
    view.cloud_number_entry = r.cloud_number_entry;
    view.ice_number_entry = r.ice_number_entry;
    view.cloud_number_tendency = r.cloud_number_tendency;
    view.pressure = r.pressure;
    view.qv = r.qv;
    thompson_aa_final_phase_cleanup_item(
        view, p.dt, int(idx), p.r1, p.r2, p.am_i,
        p.wrf39 ? 499.0e3f : 999.0e3f);
}
