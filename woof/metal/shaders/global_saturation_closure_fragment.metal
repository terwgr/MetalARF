// Validate/clamp resident Global water state and close column water locally.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_saturation_closure_fragment_bindings.metalh"

using namespace metal;

static inline ulong sat_clamp_zero(ulong value) {
    if (!soft64_is_finite(value)) return value;
    return soft64_greater(value, 0ul) ? value : 0ul;
}

kernel void global_saturation_closure_fragment(
    constant GlobalSaturationClosureFragmentResources &r [[buffer(0)]],
    constant GlobalSaturationClosureFragmentParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= uint(p.ncolumn)) return;
    ulong atmospheric_water = 0ul;
    uint status = 0u;
    for (int level = 0; level < p.nlevel; ++level) {
        const uint index = uint(level) * uint(p.ncolumn) + column;
        ulong values[6] = {r.qv[index], r.qc[index], r.qr[index],
                           r.qi[index], r.qs[index], r.qg[index]};
        for (uint species = 0; species < 6u; ++species) {
            if (soft64_less(values[species], 0xbddb7cdfd9d7bdbbul))
                status |= (1u << species);
            values[species] = sat_clamp_zero(values[species]);
        }
        if (!soft64_is_finite(r.theta[index])) status |= 64u;
        r.qv_out[index] = values[0]; r.qc_out[index] = values[1];
        r.qr_out[index] = values[2]; r.qi_out[index] = values[3];
        r.qs_out[index] = values[4]; r.qg_out[index] = values[5];
        ulong total = values[0];
        total = soft64_add(total, values[1]);
        total = soft64_add(total, values[2]);
        total = soft64_add(total, values[3]);
        total = soft64_add(total, values[4]);
        total = soft64_add(total, values[5]);
        const ulong mass = soft64_div(
            soft64_mul(total, r.dp[index]), r.gravity[0]);
        atmospheric_water = soft64_add(atmospheric_water, mass);
    }
    const ulong old_surface = r.surface_water[column];
    if (soft64_less(old_surface, 0xbe45798ee2308c3aul)) status |= 128u; // -1e-8
    const ulong after = soft64_add(atmospheric_water, old_surface);
    const ulong repair = soft64_sub(after, r.before_water[column]);
    ulong updated_surface = soft64_sub(old_surface, repair);
    if (soft64_less(updated_surface, 0xbe45798ee2308c3aul)) status |= 256u;
    updated_surface = sat_clamp_zero(updated_surface);
    r.surface_water_out[column] = updated_surface;
    r.partial_status[column] = status;
    r.partial_repair_max[column] = soft64_abs_bits(repair);
}
