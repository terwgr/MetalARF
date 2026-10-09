// Apply the remaining surface energy tendency to a resident skin-temperature plane.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_surface_energy_update_fragment_bindings.metalh"
using namespace metal;

static inline ulong gse_max(ulong a, ulong b) {
    return soft64_greater(a, b) ? a : b;
}

kernel void global_surface_energy_update_fragment(
    constant GlobalSurfaceEnergyUpdateFragmentResources &r [[buffer(0)]],
    constant GlobalSurfaceEnergyUpdateFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong zero = 0ul;
    const ulong minimum_heat_capacity = 0x40c3880000000000ul; // 1e4
    const ulong net = soft64_sub(r.net_surface_radiation[tid],
        soft64_add(r.sensible[tid], soft64_mul(
            r.latent_heat_vaporization[0], r.evaporation[tid])));
    const ulong capacity = gse_max(r.heat_capacity[tid], minimum_heat_capacity);
    const ulong increment = soft64_div(soft64_mul(r.dt[0], net), capacity);
    r.updated_surface_temperature[tid] = soft64_add(
        r.surface_temperature[tid], soft64_add(increment, zero));
}
