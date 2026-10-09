// needs integration: Hex density tendency from horizontal and vertical fluxes.
#include <metal_stdlib>
#include "hex_density_tendency_fragment_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void hex_density_tendency_fragment(
    constant HexDensityTendencyFragmentResources &r [[buffer(0)]],
    constant HexDensityTendencyFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.ncells);
    if (tid >= count) return;
    const uint level = tid / uint(p.ncells);
    const uint cell = tid % uint(p.ncells);
    const uint lower = level * uint(p.ncells) + cell;
    const uint upper = (level + 1u) * uint(p.ncells) + cell;
    r.tendency[tid] = -r.horizontal[tid]
        - r.rdzw[level] * (r.rho_w[upper] - r.rho_w[lower])
        + r.physics_tendency[tid];
}
