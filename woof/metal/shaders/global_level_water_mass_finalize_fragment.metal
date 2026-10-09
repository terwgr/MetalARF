// Finish the deterministic latitude fold of resident Global water rows.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_level_water_mass_finalize_fragment_bindings.metalh"

using namespace metal;

kernel void global_level_water_mass_finalize_fragment(
    constant GlobalLevelWaterMassFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalLevelWaterMassFinalizeFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nspecies) * uint(p.nlev);
    if (tid >= count) return;
    const uint base = tid * uint(p.nlat);
    ulong sum = 0ul;
    for (uint row = 0; row < uint(p.nlat); ++row) {
        sum = soft64_add(sum, soft64_mul(
            r.rows[base + row], r.quadrature_weights[row]));
    }
    const ulong half_total = soft64_mul(sum, 0x3fe0000000000000ul);
    r.mass[tid] = soft64_div(half_total, r.gravity[0]);
}
