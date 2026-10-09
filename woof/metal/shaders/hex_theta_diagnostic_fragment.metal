#include <metal_stdlib>
#include "hex_theta_diagnostic_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// Cellwise potential-temperature diagnostic used by the dry pressure rebuild.
kernel void hex_theta_diagnostic(
        constant HexThetaDiagnosticResources &r [[buffer(0)]],
        constant HexThetaDiagnosticParams &q [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong count = ulong(q.nlev) * ulong(q.ncells);
    if (q.nlev <= 0 || q.ncells <= 0 || ulong(gid) >= count) return;
    const float density = r.rho[gid];
    if (density == 0.0f) return;
    r.theta_m[gid] = r.rho_theta[gid] / density;
}
