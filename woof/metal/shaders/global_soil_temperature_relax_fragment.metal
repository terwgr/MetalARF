// Relax all soil layers toward the resident surface temperature.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_soil_temperature_relax_fragment_bindings.metalh"
using namespace metal;

kernel void global_soil_temperature_relax_fragment(
    constant GlobalSoilTemperatureRelaxFragmentResources &r [[buffer(0)]],
    constant GlobalSoilTemperatureRelaxFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong one_quarter = 0x3fd0000000000000ul;
    const ulong first = soft64_add(r.soil_temperature[tid],
        soft64_mul(r.relax[0], soft64_sub(r.surface_temperature[tid],
                                         r.soil_temperature[tid])));
    r.updated_soil_temperature[tid] = first;
    ulong previous = first;
    for (int layer = 1; layer < p.nlayer; ++layer) {
        const ulong offset = ulong(layer) * ulong(p.count) + ulong(tid);
        const ulong old_value = r.soil_temperature[offset];
        const ulong updated = soft64_add(old_value,
            soft64_mul(soft64_mul(one_quarter, r.relax[0]),
                       soft64_sub(previous, old_value)));
        r.updated_soil_temperature[offset] = updated;
        previous = updated;
    }
}
