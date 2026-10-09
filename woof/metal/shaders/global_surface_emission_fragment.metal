// Emit upward longwave radiation from a resident surface-temperature plane.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_surface_emission_fragment_bindings.metalh"
using namespace metal;

kernel void global_surface_emission_fragment(
    constant GlobalSurfaceEmissionFragmentResources &r [[buffer(0)]],
    constant GlobalSurfaceEmissionFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong t2 = soft64_mul(r.surface_temperature[tid], r.surface_temperature[tid]);
    const ulong t4 = soft64_mul(t2, t2);
    r.surface_emission[tid] = soft64_mul(
        soft64_mul(r.emissivity[tid], r.stefan_boltzmann[0]), t4);
}
