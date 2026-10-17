// FP64 Cartesian wind and full-level rate for the default Global SL route.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "global_semilag_cartesian_wind_fragment_bindings.metalh"
#include "global_semilag_level_rate_fragment_bindings.metalh"
#include "global_semilag_level_rate_extrapolate_fragment_bindings.metalh"
#include "global_semilag_zonal_derivative_fragment_bindings.metalh"
#include "global_semilag_wind_synthesis_fragment_bindings.metalh"
#include "global_semilag_thermodynamics_fragment_bindings.metalh"
#include "global_semilag_hydrostatic_fragment_bindings.metalh"
#include "global_semilag_mass_flux_fragment_bindings.metalh"
#include "global_semilag_mass_flux_reduction_fragment_bindings.metalh"
#include "global_semilag_continuity_fragment_bindings.metalh"

using namespace metal;

kernel void global_semilag_cartesian_wind_fragment(
    constant GlobalSemilagCartesianWindFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagCartesianWindFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count=uint(p.nlevel)*uint(p.nlat)*uint(p.nlon);
    if(tid>=count)return;
    const uint plane=uint(p.nlat)*uint(p.nlon);
    const uint cell=tid%plane;
    const uint row=cell/uint(p.nlon);
    const uint col=cell-row*uint(p.nlon);
    const ulong u=r.u[tid], v=r.v[tid];
    const ulong sinlat=r.sinlat[row], coslat=r.coslat[row];
    const ulong sinlon=r.sinlon[col], coslon=r.coslon[col];
    r.vx[tid]=soft64_sub(
        soft64_negate_bits(soft64_mul(u,sinlon)),
        soft64_mul(soft64_mul(v,sinlat),coslon));
    r.vy[tid]=soft64_sub(
        soft64_mul(u,coslon),
        soft64_mul(soft64_mul(v,sinlat),sinlon));
    r.vz[tid]=soft64_mul(v,coslat);
}

kernel void global_semilag_level_rate_fragment(
    constant GlobalSemilagLevelRateFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagLevelRateFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count=uint(p.nlevel)*uint(p.nlat)*uint(p.nlon);
    if(tid>=count)return;
    const uint plane=uint(p.nlat)*uint(p.nlon);
    const uint level=tid/plane;
    ulong below=0ul, above=0ul;
    if(level>0u) {
        const ulong thickness=soft64_mul(0x3fe0000000000000ul,
            soft64_add(r.dp[tid-plane],r.dp[tid]));
        below=soft64_div(r.omega_half[tid],thickness);
    }
    if(level+1u<uint(p.nlevel)) {
        const ulong thickness=soft64_mul(0x3fe0000000000000ul,
            soft64_add(r.dp[tid],r.dp[tid+plane]));
        above=soft64_div(r.omega_half[tid+plane],thickness);
    }
    r.level_rate[tid]=soft64_mul(0x3fe0000000000000ul,
                                 soft64_add(below,above));
}

kernel void global_semilag_level_rate_extrapolate_fragment(
    constant GlobalSemilagLevelRateExtrapolateFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagLevelRateExtrapolateFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.count))return;
    r.extrapolated[tid]=soft64_sub(
        soft64_mul(0x4000000000000000ul,r.current[tid]),r.previous[tid]);
}

kernel void global_semilag_zonal_derivative_fragment(
    constant GlobalSemilagZonalDerivativeFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagZonalDerivativeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.coefficient_count)) return;
    const uint width = uint(p.truncation) + 1u;
    const uint plane = width * width;
    const uint coefficient = tid % plane;
    const uint degree = coefficient / width;
    const uint order = coefficient - degree * width;
    if (order == 0u || order > degree) {
        r.derivative[2u * tid] = 0ul;
        r.derivative[2u * tid + 1u] = 0ul;
        return;
    }
    const ulong m = soft64_from_i32(int(order));
    r.derivative[2u * tid] = soft64_negate_bits(
        soft64_mul(r.coefficients[2u * tid + 1u], m));
    r.derivative[2u * tid + 1u] = soft64_mul(
        r.coefficients[2u * tid], m);
}

kernel void global_semilag_wind_synthesis_fragment(
    constant GlobalSemilagWindSynthesisFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagWindSynthesisFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlevel) * uint(p.nlat) * uint(p.nlon);
    if (tid >= count) return;
    const uint plane = uint(p.nlat) * uint(p.nlon);
    const uint row = (tid % plane) / uint(p.nlon);
    const ulong radius = r.radius[0];
    const ulong zonal_scale = soft64_mul(radius, r.coslat[row]);
    const ulong east0 = soft64_div(r.east[tid], zonal_scale);
    const ulong east1 = soft64_div(r.east[count + tid], zonal_scale);
    const ulong north0 = soft64_div(r.north[tid], radius);
    const ulong north1 = soft64_div(r.north[count + tid], radius);
    r.u[tid] = soft64_add(soft64_negate_bits(north0), east1);
    r.v[tid] = soft64_add(east0, north1);
}

kernel void global_semilag_thermodynamics_fragment(
    constant GlobalSemilagThermodynamicsFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagThermodynamicsFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong normalized = soft64_div(r.pressure_full[tid], 0x40f86a0000000000ul);
    const ulong exner = soft64_plm_pow(normalized, 0x3fd24b795eda4351ul);
    const ulong temperature = soft64_mul(r.theta[tid], exner);
    ulong condensate = soft64_add(r.qc[tid], r.qr[tid]);
    condensate = soft64_add(condensate, r.qi[tid]);
    condensate = soft64_add(condensate, r.qs[tid]);
    condensate = soft64_add(condensate, r.qg[tid]);
    ulong virtual_factor = soft64_add(0x3ff0000000000000ul,
        soft64_mul(0x3fe3851eb851eb85ul, r.qv[tid]));
    virtual_factor = soft64_sub(virtual_factor, condensate);
    r.temperature[tid] = temperature;
    r.virtual_temperature[tid] = soft64_mul(temperature, virtual_factor);
}

kernel void global_semilag_hydrostatic_fragment(
    constant GlobalSemilagHydrostaticFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagHydrostaticFragmentParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint columns = uint(p.nlat) * uint(p.nlon);
    if (column >= columns) return;
    ulong below = 0ul;
    const ulong gas_constant = r.gas_constant[0];
    for (int level = p.nlevel - 1; level >= 0; --level) {
        const uint index = uint(level) * columns + column;
        const ulong tv = r.virtual_temperature[index];
        ulong gradient;
        ulong log_spacing;
        if (level == 0) {
            log_spacing = soft64_mul(0x3fe0000000000000ul,
                soft64_add(r.ln_ratio[index + columns], r.ln_ratio[index]));
            gradient = soft64_div(
                soft64_sub(r.virtual_temperature[index + columns], tv),
                log_spacing);
        } else if (level == p.nlevel - 1) {
            log_spacing = soft64_mul(0x3fe0000000000000ul,
                soft64_add(r.ln_ratio[index], r.ln_ratio[index - columns]));
            gradient = soft64_div(
                soft64_sub(tv, r.virtual_temperature[index - columns]),
                log_spacing);
        } else {
            ulong log_spacing_sum = soft64_add(
                r.ln_ratio[index + columns], r.ln_ratio[index - columns]);
            log_spacing_sum = soft64_add(log_spacing_sum,
                soft64_mul(0x4000000000000000ul, r.ln_ratio[index]));
            log_spacing = soft64_mul(0x3fe0000000000000ul, log_spacing_sum);
            gradient = soft64_div(
                soft64_sub(r.virtual_temperature[index + columns],
                           r.virtual_temperature[index - columns]),
                log_spacing);
        }
        const ulong half_log = soft64_mul(r.ln_ratio[index],
                                          0x3fe0000000000000ul);
        const ulong correction = soft64_mul(
            gradient, soft64_mul(soft64_mul(half_log, half_log),
                                 0x3fe0000000000000ul));
        const ulong midpoint = soft64_add(
            soft64_mul(tv, half_log), correction);
        const ulong half_layer = soft64_mul(gas_constant, midpoint);
        r.geopotential[index] = soft64_add(
            soft64_add(r.surface_geopotential[column], below), half_layer);
        below = soft64_add(
            below, soft64_mul(soft64_mul(gas_constant, tv),
                              r.ln_ratio[index]));
    }
}

kernel void global_semilag_mass_flux_fragment(
    constant GlobalSemilagMassFluxFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagMassFluxFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    r.flux_u[tid] = soft64_mul(r.dp[tid], r.u[tid]);
    r.flux_v[tid] = soft64_mul(r.dp[tid], r.v[tid]);
}

kernel void global_semilag_mass_flux_reduction_fragment(
    constant GlobalSemilagMassFluxReductionFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagMassFluxReductionFragmentParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint columns = uint(p.nlat) * uint(p.nlon);
    if (column >= columns) return;
    ulong sum = 0ul;
    for (uint level = 0; level < uint(p.nlevel); ++level)
        sum = soft64_add(sum,
            r.divergence_mass[level * columns + column]);
    r.ps_t[column] = soft64_negate_bits(sum);
}

kernel void global_semilag_continuity_fragment(
    constant GlobalSemilagContinuityFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagContinuityFragmentParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint columns = uint(p.nlat) * uint(p.nlon);
    if (column >= columns) return;
    ulong omega = 0ul;
    r.omega[column] = omega;
    for (uint level = 0; level < uint(p.nlevel); ++level) {
        const ulong pressure = soft64_mul(
            r.delta_b[level], r.ps_t[column]);
        omega = soft64_sub(
            soft64_sub(omega, pressure),
            r.divergence_mass[level * columns + column]);
        r.omega[(level + 1u) * columns + column] = omega;
    }
    r.residual[column] = omega;
    r.omega[uint(p.nlevel) * columns + column] = 0ul;
}
