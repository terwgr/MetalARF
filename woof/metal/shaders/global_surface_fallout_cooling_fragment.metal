// Debit fusion energy for frozen fallout without materializing skin temperature.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_surface_fallout_cooling_fragment_bindings.metalh"
using namespace metal;

kernel void global_surface_fallout_cooling_fragment(
    constant GlobalSurfaceFalloutCoolingFragmentResources &r [[buffer(0)]],
    constant GlobalSurfaceFalloutCoolingFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong freezing = 0x4071126666666666ul; // 273.15 K
    const ulong min_heat_capacity = 0x40c3880000000000ul; // 1e4 J m^-2 K^-1
    const ulong safe_capacity = soft64_greater(r.heat_capacity[tid], min_heat_capacity)
        ? r.heat_capacity[tid] : min_heat_capacity;
    ulong temperature = r.surface_temperature[tid];
    if (soft64_greater(temperature, freezing)) {
        const ulong frozen = soft64_add(r.snow[tid], r.graupel[tid]);
        const ulong cooling = soft64_div(
            soft64_mul(r.latent_heat_fusion[0], frozen), safe_capacity);
        temperature = soft64_sub(temperature, cooling);
    }
    r.updated_surface_temperature[tid] = temperature;
}
