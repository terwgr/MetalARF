// Assemble resident non-atmospheric water stores by forecast column.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_total_water_stores_fragment_bindings.metalh"

using namespace metal;

kernel void global_total_water_stores_fragment(
    constant GlobalTotalWaterStoresFragmentResources &r [[buffer(0)]],
    constant GlobalTotalWaterStoresFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong zero = 0ul;
    ulong soil = zero;
    const ulong thickness[4] = {
        0x3fb999999999999aul, 0x3fd3333333333333ul,
        0x3fe3333333333333ul, 0x3ff0000000000000ul
    };
    for (uint layer = 0; layer < 4u; ++layer) {
        const ulong raw_fraction = r.soil_fraction[ulong(layer) * uint(p.count) + tid];
        const ulong fraction = soft64_greater(raw_fraction, zero)
            ? raw_fraction : zero;
        soil = soft64_add(soil, soft64_mul(fraction, thickness[layer]));
    }
    soil = soft64_mul(soil, 0x408f400000000000ul); // 1000 kg m-3
    soil = soft64_mul(soil, r.land_fraction[tid]);
    ulong total = soft64_add(r.surface_water[tid], soil);
    const ulong canopy = soft64_greater(r.canopy[tid], zero) ? r.canopy[tid] : zero;
    const ulong snow = soft64_greater(r.snow[tid], zero) ? r.snow[tid] : zero;
    total = soft64_add(total, canopy);
    total = soft64_add(total, snow);
    total = soft64_add(total, r.outflow[tid]);
    r.total[tid] = total;
}
