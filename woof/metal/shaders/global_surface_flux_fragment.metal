// needs integration: FP64 Global surface exchange update.
//
// This is the per-column surface_fluxes update in ReferencePhysics. The
// caller supplies the prepared saturation mixing ratio, wetness factor,
// exchange coefficient and lowest-level density/thermodynamics. The ABI
// owner must register this source fragment before dispatch.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_surface_flux_fragment_bindings.metalh"

using namespace metal;

static inline ulong global_surface_min(ulong a, ulong b) {
    return soft64_less(a, b) ? a : b;
}

static inline ulong global_surface_max(ulong a, ulong b) {
    return soft64_greater(a, b) ? a : b;
}

kernel void global_surface_flux_fragment(
    constant GlobalSurfaceFluxFragmentResources &r [[buffer(0)]],
    constant GlobalSurfaceFluxFragmentParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= q.count) return;
    const ulong one = 0x3ff0000000000000ul;
    const ulong zero = 0ul;
    const ulong safe_dp = soft64_greater(r.dp_bottom[tid], one)
        ? r.dp_bottom[tid] : one;
    const ulong sensible_rate = soft64_mul(
        soft64_mul(soft64_mul(r.rho_bottom[tid], r.dry_air_cp[0]), r.cd[tid]),
        soft64_mul(r.wind[tid], soft64_sub(r.surface_temperature[tid], r.air_temperature[tid])));
    ulong evaporation_rate = soft64_mul(
        soft64_mul(soft64_mul(r.beta[tid], r.rho_bottom[tid]), r.cd[tid]),
        soft64_mul(r.wind[tid], soft64_sub(r.saturation_mixing_ratio[tid], r.qv[tid])));
    evaporation_rate = global_surface_min(evaporation_rate, r.available_water_rate[tid]);
    evaporation_rate = global_surface_max(evaporation_rate, zero);
    const ulong dq = soft64_div(soft64_mul(evaporation_rate, r.gravity[0]), safe_dp);
    const ulong dtemp = soft64_div(soft64_mul(sensible_rate, r.gravity[0]),
        soft64_mul(r.dry_air_cp[0], safe_dp));
    const ulong stress = soft64_mul(soft64_mul(r.rho_bottom[tid], r.cd[tid]), r.wind[tid]);
    const ulong momentum_factor = soft64_div(
        soft64_mul(soft64_mul(r.dt[0], stress), r.gravity[0]), safe_dp);
    r.sensible[tid] = sensible_rate;
    r.evaporation[tid] = evaporation_rate;
    r.updated_qv[tid] = soft64_add(r.qv[tid], soft64_mul(r.dt[0], dq));
    r.updated_theta[tid] = soft64_add(r.theta[tid],
        soft64_div(soft64_mul(r.dt[0], dtemp), r.exner_bottom[tid]));
    r.updated_u[tid] = soft64_sub(r.u[tid], soft64_mul(momentum_factor, r.u[tid]));
    r.updated_v[tid] = soft64_sub(r.v[tid], soft64_mul(momentum_factor, r.v[tid]));
    r.updated_surface_water[tid] = soft64_sub(
        r.surface_water[tid], soft64_mul(r.dt[0], evaporation_rate));
    const ulong net_surface = soft64_sub(r.net_surface_radiation[tid],
        soft64_add(sensible_rate, soft64_mul(r.latent_heat[0], evaporation_rate)));
    const ulong safe_heat_capacity = soft64_greater(
        r.surface_heat_capacity[tid], 0x40c3880000000000ul) // 1.0e4
        ? r.surface_heat_capacity[tid] : 0x40c3880000000000ul;
    r.updated_surface_temperature[tid] = soft64_add(r.surface_temperature[tid],
        soft64_div(soft64_mul(r.dt[0], net_surface), safe_heat_capacity));
}
