#include <metal_stdlib>
#include "hex_pressure_perturbation_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// Rebuild the saved dry pressure perturbation at one level/cell. The staged
// operands are supplied by the pressure/Exner diagnostic route.
kernel void hex_pressure_perturbation(
        constant HexPressurePerturbationResources &r [[buffer(0)]],
        constant HexPressurePerturbationParams &q [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong count = ulong(q.nlev) * ulong(q.ncells);
    if (q.nlev <= 0 || q.ncells <= 0 || ulong(gid) >= count) return;

    const int level = int(gid / uint(q.ncells));
    const float exner = r.exner[gid];
    const float exner_perturbation = exner - r.exner_base[gid];
    const float thermodynamic_perturbation =
        exner * r.rho_theta_perturbation[gid]
        + r.rho_theta_base[gid] * exner_perturbation;
    r.pressure_perturbation[gid] =
        r.zz[level] * q.rgas * thermodynamic_perturbation;
}
