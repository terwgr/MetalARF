// Derive surface-layer inputs from resident lowest-level Global prognostics.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "global_surface_flux_prep_fragment_bindings.metalh"
using namespace metal;

static inline ulong gsf_min(ulong a, ulong b) {
    return soft64_less(a, b) ? a : b;
}

static inline ulong gsf_max(ulong a, ulong b) {
    return soft64_greater(a, b) ? a : b;
}

kernel void global_surface_flux_prep_fragment(
    constant GlobalSurfaceFluxPrepFragmentResources &r [[buffer(0)]],
    constant GlobalSurfaceFluxPrepFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong one = 0x3ff0000000000000ul;
    const ulong zero = 0ul;
    const ulong temperature = soft64_mul(r.theta[tid], r.exner_bottom[tid]);
    const ulong virtual_temperature = soft64_mul(
        temperature, soft64_add(one, soft64_mul(
            0x3fe3851eb851eb85ul, r.qv[tid])));
    const ulong safe_virtual = gsf_max(virtual_temperature,
        0x4062c00000000000ul); // 150 K
    const ulong density = soft64_div(
        r.p_full_bottom[tid], soft64_mul(r.dry_air_gas_constant[0], safe_virtual));
    const ulong speed = gsf_max(soft64_sqrt(soft64_add(
        soft64_mul(r.u[tid], r.u[tid]), soft64_mul(r.v[tid], r.v[tid]))),
        0x3fe0000000000000ul); // 0.5 m/s
    const ulong z0 = gsf_max(r.roughness[tid], 0x3ee4f8b588e368f1ul); // 1e-5 m
    const ulong z0_ratio = soft64_div(z0, 0x3f1a36e2eb1c432dul); // 1e-4 m
    const ulong exchange = soft64_mul(r.bulk_exchange_coefficient[0],
        soft64_add(one, soft64_mul(0x3fb47ae147ae147bul, // 0.08
            soft64_plm_log(soft64_add(one, z0_ratio)))));
    const ulong wetness = gsf_min(gsf_max(soft64_div(
        r.soil_water_fraction[tid], r.soil_wetness_capacity[0]), zero), one);
    const ulong beta = soft64_add(soft64_sub(one, r.land_fraction[tid]),
        soft64_mul(r.land_fraction[tid], wetness));

    const ulong skin_temperature = r.surface_temperature[tid];
    ulong clipped_temperature = gsf_min(gsf_max(skin_temperature,
        0x4062c00000000000ul), 0x4075e00000000000ul); // [150 K, 350 K]
    const ulong exponent = soft64_div(
        soft64_mul(0x4031ae147ae147aeul, // 17.67
            soft64_sub(clipped_temperature, 0x4071126666666666ul)), // 273.15 K
        gsf_max(soft64_sub(clipped_temperature, 0x403da66666666666ul), one)); // T - 29.65
    const ulong vapor_pressure = gsf_min(
        soft64_mul(0x408319999999999aul, soft64_plm_exp(exponent)),
        soft64_mul(0x3fee666666666666ul, r.p_surface[tid])); // 0.95 p
    const ulong qsat = soft64_div(soft64_mul(r.epsilon[0], vapor_pressure),
        gsf_max(soft64_sub(r.p_surface[tid], vapor_pressure), one));

    r.air_temperature[tid] = temperature;
    r.rho_bottom[tid] = density;
    r.wind[tid] = speed;
    r.cd[tid] = exchange;
    r.saturation_mixing_ratio[tid] = qsat;
    r.beta[tid] = beta;
}
