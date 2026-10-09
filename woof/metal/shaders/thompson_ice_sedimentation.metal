#include <metal_stdlib>
using namespace metal;
#include "thompson_ice_sedimentation_seed_bindings.metalh"
#include "thompson_ice_sedimentation_active_ice_bindings.metalh"
#include "thompson_ice_sedimentation_select_timestep_bindings.metalh"
#include "thompson_ice_sedimentation_subcycling_bindings.metalh"
#include "thompson_ice_sedimentation_finalize_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"

static inline ulong ice64(float x) { return soft64_from_f32_bits(as_type<uint>(x)); }
static inline ulong ice64pow3(ulong x) {
    return soft64_plm_pow(x, 0x4008000000000000ul);
}
static inline float ice64_to_float(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}

// Seed density, zero tendencies, and the inactive-state baseline.
kernel void thompson_ice_sedimentation_seed(
    constant ThompsonIceSedimentationSeedResources& r [[buffer(0)]],
    constant ThompsonIceSedimentationSeedParams& p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint columns = uint(p.ny * p.nx);
    if (column >= columns) return;
    const int nz = p.nz, nx = p.nx, ny = p.ny;
    const int j = int(column) / nx;
    const int i = int(column) - j * nx;
    for (int k = nz - 1; k >= 0; --k) {
        const uint idx = uint(k * ny * nx + j * nx + i);
        const float qvk = fmax(1.0e-10f, r.qv[idx]);
        const float rho = 0.622f * r.pressure[idx]
            / (287.04f * r.temperature[idx] * (qvk + 0.622f));
        r.density[idx] = rho;
        r.qi_tendency[idx] = 0.0f;
        r.ni_tendency[idx] = 0.0f;
        const float ice_state_rho = r.reference_density[idx];
        const bool l_qi = r.qi[idx] > 1.0e-12f;
        r.ice_mass[idx] = l_qi ? r.qi[idx] * ice_state_rho : 1.0e-12f;
        r.ice_number[idx] = l_qi
            ? fmax(1.0e-6f, r.ni[idx] * ice_state_rho) : 1.0e-6f;
        r.mass_velocity[idx] = 0.0f;
        r.number_velocity[idx] = 0.0f;
        r.active_marker[idx] = 0.0f;
    }
}

// Active ice limiter/speed calculation is independent by level. It repeats
// the source active predicate, including clamp-then-number-recompute order.
kernel void thompson_ice_sedimentation_active_ice(
    constant ThompsonIceSedimentationActiveIceResources& r [[buffer(0)]],
    constant ThompsonIceSedimentationActiveIceParams& p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint columns = uint(p.ny * p.nx);
    if (column >= columns) return;
    const int nz = p.nz, nx = p.nx, ny = p.ny;
    const int j = int(column) / nx, i = int(column) - j * nx;
    const float pi = 3.1415926536f;
    const float am_i = pi * 890.0f / 6.0f;
    const float oig2 = 1.0f / 6.0f;
    const float rho_not = 101325.0f / (287.05f * 298.0f);
    for (int k = nz - 1; k >= 0; --k) {
        const uint idx = uint(k * ny * nx + j * nx + i);
        const float ice_state_rho = r.reference_density[idx];
        if (r.qi[idx] > 1.0e-12f && r.qi[idx] * ice_state_rho > 1.0e-12f) {
            const float ri = r.qi[idx] * ice_state_rho;
            float nn = fmax(1.0e-6f, r.ni[idx] * ice_state_rho);
            if (nn <= 1.0e-6f) {
                const ulong lambda = 0x41286a0000000000ul;
                const float prefix = (oig2 * ri) / am_i;
                nn = fmin(999.0e3f, ice64_to_float(
                    soft64_mul(ice64(prefix), ice64pow3(lambda))));
            }
            const float lambda_arg = am_i * 6.0f * nn / ri;
            const float lambda_f = precise::pow(lambda_arg, 1.0f / 3.0f);
            ulong lambda = ice64(lambda_f);
            float diameter = ice64_to_float(soft64_div(
                0x4010000000000000ul, lambda));
            if (diameter < 5.0e-6f) {
                diameter = 5.0e-6f;
                lambda = soft64_div(0x4010000000000000ul, ice64(diameter));
                const float prefix = (oig2 * ri) / am_i;
                nn = fmin(999.0e3f, ice64_to_float(soft64_mul(
                    ice64(prefix), ice64pow3(lambda))));
            } else if (diameter > 300.0e-6f) {
                diameter = 300.0e-6f;
                lambda = soft64_div(0x4010000000000000ul, ice64(diameter));
                const float prefix = (oig2 * ri) / am_i;
                nn = ice64_to_float(soft64_mul(ice64(prefix), ice64pow3(lambda)));
            }
            r.ice_mass[idx] = ri;
            r.ice_number[idx] = nn;
            const float rhof = precise::sqrt(rho_not / r.density[idx]);
            const ulong inverse_lambda = soft64_div(0x3ff0000000000000ul, lambda);
            const float mass_prefix = rhof * 1493.9f * 24.0f * oig2;
            r.mass_velocity[idx] = ice64_to_float(soft64_mul(
                ice64(mass_prefix), inverse_lambda));
            const float number_prefix = (rhof * 1493.9f * 3.3233511f) / 1.3293403f;
            r.number_velocity[idx] = ice64_to_float(soft64_mul(
                ice64(number_prefix), inverse_lambda));
            r.active_marker[idx] = 1.0f;
        }
    }
}

// Descending scan carries active-layer velocities through inactive layers,
// then performs the source timestep and sediment-top selection.
kernel void thompson_ice_sedimentation_select_timestep(
    constant ThompsonIceSedimentationSelectTimestepResources& r [[buffer(0)]],
    constant ThompsonIceSedimentationSelectTimestepParams& p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint columns = uint(p.ny * p.nx);
    if (column >= columns) return;
    const int nz = p.nz, nx = p.nx, ny = p.ny;
    const int j = int(column) / nx, i = int(column) - j * nx;
    int sediment_top = 0, nstep = 0;
    float velocity_above_mass = 0.0f, velocity_above_number = 0.0f;
    for (int k = nz - 1; k >= 0; --k) {
        const uint idx = uint(k * ny * nx + j * nx + i);
        if (r.active_marker[idx] != 0.0f) {
            velocity_above_mass = r.mass_velocity[idx];
            velocity_above_number = r.number_velocity[idx];
        } else {
            r.mass_velocity[idx] = velocity_above_mass;
            r.number_velocity[idx] = velocity_above_number;
        }
        const float mass_velocity = r.mass_velocity[idx];
        if (mass_velocity > 1.0e-3f) {
            sediment_top = max(sediment_top, k);
            const float delta_tp = r.dz[idx] / mass_velocity;
            nstep = max(nstep, int(float(p.dt) / delta_tp + 1.0f));
        }
    }
    if (sediment_top == nz - 1) sediment_top = nz - 2;
    nstep = max(nstep, 1);
    r.sediment_top[column] = sediment_top;
    r.nstep[column] = nstep;
}

static void ice_subcycling_fluxes(
    constant ThompsonIceSedimentationSubcyclingResources& r,
    int nz, int ny, int nx, int j, int i) {
    for (int k = nz - 1; k >= 0; --k) {
        const uint idx = uint(k * ny * nx + j * nx + i);
        r.mass_flux[idx] = r.mass_velocity[idx] * r.ice_mass[idx];
        r.number_flux[idx] = r.number_velocity[idx] * r.ice_number[idx];
    }
}

static void ice_subcycling_top_update(
    constant ThompsonIceSedimentationSubcyclingResources& r,
    uint idx, float onstep, float dt_substep) {
    const float inv_dz = 1.0f / r.dz[idx];
    const float inv_rho = 1.0f / r.density[idx];
    r.qi_tendency[idx] -= r.mass_flux[idx] * inv_dz * onstep * inv_rho;
    r.ni_tendency[idx] -= r.number_flux[idx] * inv_dz * onstep * inv_rho;
    r.ice_mass[idx] = fmax(1.0e-12f,
        r.ice_mass[idx] - r.mass_flux[idx] * inv_dz * dt_substep);
    r.ice_number[idx] = fmax(1.0e-6f,
        r.ice_number[idx] - r.number_flux[idx] * inv_dz * dt_substep);
}

static void ice_subcycling_layer_update(
    constant ThompsonIceSedimentationSubcyclingResources& r,
    uint idx, uint layer_stride, float onstep, float dt_substep) {
    const float inv_dz = 1.0f / r.dz[idx];
    const float inv_rho = 1.0f / r.density[idx];
    const float mass_divergence = r.mass_flux[idx + layer_stride] - r.mass_flux[idx];
    const float number_divergence = r.number_flux[idx + layer_stride] - r.number_flux[idx];
    r.qi_tendency[idx] += mass_divergence * inv_dz * onstep * inv_rho;
    r.ni_tendency[idx] += number_divergence * inv_dz * onstep * inv_rho;
    r.ice_mass[idx] = fmax(1.0e-12f,
        r.ice_mass[idx] + mass_divergence * inv_dz * dt_substep);
    r.ice_number[idx] = fmax(1.0e-6f,
        r.ice_number[idx] + number_divergence * inv_dz * dt_substep);
}

// One source-ordered subcycle: flux snapshot, top boundary, descending
// interior mutation, then the per-step surface-export test.
static __attribute__((noinline)) float ice_subcycling_step(
    constant ThompsonIceSedimentationSubcyclingResources& r,
    int nz, int ny, int nx, int j, int i, int sediment_top,
    float onstep, float dt_substep) {
    ice_subcycling_fluxes(r, nz, ny, nx, j, i);
    const uint top_idx = uint((nz - 1) * ny * nx + j * nx + i);
    ice_subcycling_top_update(r, top_idx, onstep, dt_substep);
    for (int k = sediment_top; k >= 0; --k) {
        const uint idx = uint(k * ny * nx + j * nx + i);
        ice_subcycling_layer_update(r, idx, uint(ny * nx), onstep, dt_substep);
    }
    const uint bottom_idx = uint(j * nx + i);
    return r.ice_mass[bottom_idx] > 1.0e-9f
        ? r.mass_flux[bottom_idx] * dt_substep : 0.0f;
}

// Subcycling stage. The helper boundaries match the contiguous statements
// inside each source step while the column's step and layer loops stay ordered.
kernel void thompson_ice_sedimentation_subcycling(
    constant ThompsonIceSedimentationSubcyclingResources& r [[buffer(0)]],
    constant ThompsonIceSedimentationSubcyclingParams& p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint columns = uint(p.ny * p.nx);
    if (column >= columns) return;
    const int nz = p.nz, nx = p.nx, ny = p.ny;
    const int j = int(column) / nx, i = int(column) - j * nx;
    const int sediment_top = r.sediment_top[column];
    const int nstep = r.nstep[column];
    const float onstep = 1.0f / float(nstep);
    const float dt_substep = float(p.dt) * onstep;
    float exported = 0.0f;
    for (int step = 0; step < nstep; ++step) {
        exported += ice_subcycling_step(
            r, nz, ny, nx, j, i, sediment_top, onstep, dt_substep);
    }
    r.exported[column] = exported;
}

// Terminal tendency and surface updates retain the source's diameter clamp
// followed by number recomputation order.
kernel void thompson_ice_sedimentation_finalize(
    constant ThompsonIceSedimentationFinalizeResources& r [[buffer(0)]],
    constant ThompsonIceSedimentationFinalizeParams& p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint columns = uint(p.ny * p.nx);
    if (column >= columns) return;
    const int nz = p.nz, nx = p.nx, ny = p.ny;
    const int j = int(column) / nx, i = int(column) - j * nx;
    const float pi = 3.1415926536f;
    const float am_i = pi * 890.0f / 6.0f;
    const float oig2 = 1.0f / 6.0f;
    for (int k = 0; k < nz; ++k) {
        const uint idx = uint(k * ny * nx + j * nx + i);
        const float qi_new = r.qi[idx] + r.qi_tendency[idx] * float(p.dt);
        float ni_new = fmax(1.0e-6f / r.density[idx],
            r.ni[idx] + r.ni_tendency[idx] * float(p.dt));
        if (qi_new <= 1.0e-12f) {
            r.qi[idx] = qi_new;
            r.ni[idx] = r.ni[idx] + r.ni_tendency[idx] * float(p.dt);
            continue;
        }
        const float lambda_arg = am_i * 6.0f * ni_new / qi_new;
        ulong lambda = ice64(precise::pow(lambda_arg, 1.0f / 3.0f));
        float diameter = ice64_to_float(soft64_div(0x4010000000000000ul, lambda));
        if (diameter < 5.0e-6f) diameter = 5.0e-6f;
        else if (diameter > 300.0e-6f) diameter = 300.0e-6f;
        lambda = soft64_div(0x4010000000000000ul, ice64(diameter));
        const float prefix = (oig2 * qi_new) / am_i;
        ni_new = fmin(ice64_to_float(soft64_mul(
            ice64(prefix), ice64pow3(lambda))), 999.0e3f / r.density[idx]);
        r.qi[idx] = qi_new;
        r.ni[idx] = ni_new;
    }
    const float exported = r.exported[column];
    r.rainncv[column] = exported;
    r.snowncv[column] = exported;
    r.rainnc[column] += exported;
    r.snownc[column] += exported;
}
