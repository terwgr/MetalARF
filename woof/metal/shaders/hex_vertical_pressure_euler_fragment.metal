// needs integration: Hex vertical pressure Euler tendency.
#include <metal_stdlib>
#include "hex_vertical_pressure_euler_fragment_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void hex_vertical_pressure_euler_fragment(
    constant HexVerticalPressureEulerFragmentResources &r [[buffer(0)]],
    constant HexVerticalPressureEulerFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total = uint(p.nlev + 1) * uint(p.ncells);
    if (tid >= total) return;
    const uint level = tid / uint(p.ncells);
    const uint cell = tid % uint(p.ncells);
    if (level == 0u || level >= uint(p.nlev)) {
        r.tendency[tid] = 0.0f;
        return;
    }
    const uint current = level * uint(p.ncells) + cell;
    const uint previous = (level - 1u) * uint(p.ncells) + cell;
    const float inner = r.rdzu[level] *
        (r.pressure_perturbation[current] - r.pressure_perturbation[previous])
        - (r.fzm[level] * r.dpdz[current] + r.fzp[level] * r.dpdz[previous]);
    r.tendency[tid] = -(r.cqw[current] * inner);
}
