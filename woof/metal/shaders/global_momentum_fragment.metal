// needs integration: FP64 Global momentum forcing and Bernoulli stage.
//
// These are the per-grid-point operations in MoistHybridModel._rhs_from_sources.
// The ABI owner must register this fragment before the Global caller can
// dispatch it. Keep the source's multiply/add association explicit.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_momentum_fragment_bindings.metalh"

using namespace metal;

kernel void global_momentum_fragment(
    constant GlobalMomentumFragmentResources &r [[buffer(0)]],
    constant GlobalMomentumFragmentParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(q.count)) return;
    const ulong absolute_vorticity = soft64_add(r.vorticity[tid], r.coriolis[tid]);
    const ulong pressure_force = soft64_mul(
        soft64_mul(r.gas_constant[0], r.virtual_temperature[tid]), r.pressure_gradient[tid]);
    const ulong zonal_coriolis = soft64_mul(absolute_vorticity, r.meridional_wind[tid]);
    const ulong meridional_coriolis = soft64_negate_bits(
        soft64_mul(absolute_vorticity, r.zonal_wind[tid]));
    const ulong zonal_pressure = soft64_mul(pressure_force, r.grad_logps_zonal[tid]);
    const ulong meridional_pressure = soft64_mul(pressure_force, r.grad_logps_meridional[tid]);
    r.tendency_zonal[tid] = soft64_add(
        soft64_add(zonal_coriolis, soft64_negate_bits(r.vertical_zonal[tid])),
        soft64_negate_bits(zonal_pressure));
    r.tendency_meridional[tid] = soft64_add(
        soft64_add(meridional_coriolis, soft64_negate_bits(r.vertical_meridional[tid])),
        soft64_negate_bits(meridional_pressure));
    const ulong speed_squared = soft64_add(
        soft64_mul(r.zonal_wind[tid], r.zonal_wind[tid]),
        soft64_mul(r.meridional_wind[tid], r.meridional_wind[tid]));
    r.bernoulli[tid] = soft64_add(
        soft64_mul(r.half_bits[0], speed_squared), r.geopotential[tid]);
}
