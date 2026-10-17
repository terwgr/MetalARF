#include <metal_stdlib>
#include "pressure_power.metalh"
#include "hex_exner_pressure_diagnostic_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// Rebuild dry Exner and pressure for one level/cell. The positive argument
// uses the frozen multiply order before the power operation.
kernel void hex_exner_pressure_diagnostic(
        constant HexExnerPressureDiagnosticResources &r [[buffer(0)]],
        constant HexExnerPressureDiagnosticParams &q [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong count = ulong(q.nlev) * ulong(q.ncells);
    if (q.nlev <= 0 || q.ncells <= 0 || ulong(gid) >= count) return;
    const int level = int(gid / uint(q.ncells));
    const float argument =
        r.zz[level] * q.rgas_over_reference_pressure * r.rho_theta[gid];
    if (argument <= 0.0f) return;
    const float exner = fp32_pressure_power(argument, 1.0f, q.exner_exponent);
    r.exner[gid] = exner;
    r.pressure[gid] = r.zz[level] * q.rgas * r.rho_theta[gid] * exner;
}
