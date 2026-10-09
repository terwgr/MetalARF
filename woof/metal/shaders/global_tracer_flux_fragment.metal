// needs integration: FP64 Global tracer flux construction.
//
// This source fragment is intentionally not admitted by the Metal registry.
// The ABI owner must add the resource/parameter records and kernel signature
// before a caller can dispatch it.  The Global NumPy expression builds the
// mass-weighted zonal and meridional fluxes as (dp * tracer) * wind, with
// distinct roundings; keep that order here.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_tracer_flux_fragment_bindings.metalh"

using namespace metal;

kernel void global_tracer_flux_fragment(
    constant GlobalTracerFluxFragmentResources &r [[buffer(0)]],
    constant GlobalTracerFluxFragmentParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(q.count)) return;
    const ulong mass_tracer = soft64_mul(r.dp[tid], r.tracer[tid]);
    r.zonal_flux[tid] = soft64_mul(mass_tracer, r.zonal_wind[tid]);
    r.meridional_flux[tid] = soft64_mul(mass_tracer, r.meridional_wind[tid]);
}
