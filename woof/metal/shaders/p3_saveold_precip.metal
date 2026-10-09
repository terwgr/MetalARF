#include <metal_stdlib>
#include "p3_saveold_precip_bindings.metalh"

using namespace metal;

// P3 cross-step state and wrapper precipitation conversion
// (p3.cu:1479-1494, 2033-2034; module_mp_p3.F:892-898).
#pragma clang fp contract(off)

kernel void p3_saveold_precip(
    constant P3SaveoldPrecipResources &r [[buffer(0)]],
    constant P3SaveoldPrecipParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny) * uint(p.nx);
    if (column >= ncol) return;
    for (uint k = 0; k < uint(p.nz); ++k) {
        const ulong idx = ulong(k) * ulong(ncol) + ulong(column);
        r.th_old[idx] = r.theta[idx];
        r.qv_old[idx] = r.qv[idx];
    }

    const float dum1 = 1000.0f * p.dt;
    const float prt_liq = r.prt_liq[column];
    const float prt_sol = r.prt_sol[column];
    const float total = prt_liq + prt_sol;
    r.rainnc[column] = r.rainnc[column] + total * dum1;
    r.rainncv[column] = total * dum1;
    r.snownc[column] = r.snownc[column] + prt_sol * dum1;
    r.snowncv[column] = prt_sol * dum1;
    r.sr[column] = prt_sol / (prt_liq + prt_sol + 1.0e-12f);
}
