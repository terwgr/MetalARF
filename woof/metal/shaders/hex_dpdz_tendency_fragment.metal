// needs integration: Hex saved vertical pressure-gradient tendency.
#include <metal_stdlib>
#include "hex_dpdz_tendency_fragment_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void hex_dpdz_tendency_fragment(
    constant HexDpdzTendencyFragmentResources &r [[buffer(0)]],
    constant HexDpdzTendencyFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.ncells);
    if (tid >= count) return;
    const float perturbation = r.density_perturbation[tid];
    if (p.use_moist == 0) {
        r.dpdz[tid] = -p.gravity * perturbation;
        return;
    }
    const float qtot = r.qtot[tid];
    const float one = 1.0f;
    r.dpdz[tid] = -p.gravity *
        (r.rho_base[tid] * qtot + perturbation * (one + qtot));
}
