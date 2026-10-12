#include <metal_stdlib>
#include "soft_gfk_pow.metalh"
#include "p3_entry_prepare_bindings.metalh"

using namespace metal;

// P3's WRF wrapper entry initialization (module_mp_p3.F:2270-2288) and
// p3_step_prep_levels (p3.cu:620-643).  A thread owns one horizontal
// column, so its level loop preserves the source's independent column order.
// The source uses the repository-pinned r_pow implementation.
#pragma clang fp contract(off)

kernel void p3_entry_prepare(
    constant P3EntryPrepareResources &r [[buffer(0)]],
    constant P3EntryPrepareParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny) * uint(p.nx);
    if (column >= ncol) return;
    const uint nz = uint(p.nz);

    for (uint k = 0; k < nz; ++k) {
        const ulong idx = ulong(k) * ulong(ncol) + ulong(column);
        // Keep the source multiply before exponentiation and the separate
        // theta-times-Exner store.
        const float scaled_pressure = r.pressure[idx] * p.pressure_scale;
        const float tm = as_type<float>(soft_gfk_pow_bits(
            as_type<uint>(scaled_pressure), as_type<uint>(p.exponent)));
        r.tmparr1[idx] = tm;
        r.temperature[idx] = r.theta[idx] * tm;
        // WRF moist_physics_prep_em parks this same full theta before the
        // scheme changes it. The later moist_physics_finish consumes it.
        r.h_diabatic[idx] = r.theta[idx];
        r.qv[idx] = fmax(r.qv[idx], 0.0f);

        r.zdbz[idx] = p.zdbz_default;
        r.effc[idx] = p.effc_default;
        r.effi[idx] = p.effi_default;
        r.vmi[idx] = p.vmi_default;
        r.di[idx] = p.di_default;
        r.rhopo[idx] = p.rhopo_default;
    }

    r.prt_liq[column] = p.prt_liq_default;
    r.prt_sol[column] = p.prt_sol_default;
    r.flags[column] = 0.0f;
    r.flags[ncol + column] = 0.0f;
}
