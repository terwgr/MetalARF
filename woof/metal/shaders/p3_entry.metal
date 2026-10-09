#include <metal_stdlib>
#include "p3_entry_prepare_bindings.metalh"

using namespace metal;

// P3's WRF wrapper entry initialization (module_mp_p3.F:2270-2288) and
// p3_step_prep_levels (p3.cu:620-643).  A thread owns one horizontal
// column, so its level loop preserves the source's independent column order.
// The source uses the repository-pinned r_pow; native MSL pow below is an
// ordinary-mode translation whose CUDA libm identity is not established.
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
        // theta-times-Exner store.  This is the only unresolved libm edge in
        // this entry slice; later P3 processing is not implied by this kernel.
        const float scaled_pressure = r.pressure[idx] * 1.0e-5f;
        const float tm = pow(scaled_pressure, p.exponent);
        r.tmparr1[idx] = tm;
        r.temperature[idx] = r.theta[idx] * tm;
        // WRF moist_physics_prep_em parks this same full theta before the
        // scheme changes it. The later moist_physics_finish consumes it.
        r.h_diabatic[idx] = r.theta[idx];
        r.qv[idx] = fmax(r.qv[idx], 0.0f);

        r.zdbz[idx] = -99.0f;
        r.effc[idx] = 10.0e-6f;
        r.effi[idx] = 25.0e-6f;
        r.vmi[idx] = 0.0f;
        r.di[idx] = 0.0f;
        r.rhopo[idx] = 0.0f;
    }

    r.prt_liq[column] = 0.0f;
    r.prt_sol[column] = 0.0f;
    r.flags[column] = 0.0f;
    r.flags[ncol + column] = 0.0f;
}
