// Assemble the resident Global semi-Lagrangian pressure-gradient force.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_force_fragment_bindings.metalh"

using namespace metal;

kernel void global_semilag_force_fragment(
    constant GlobalSemilagForceFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagForceFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    r.force[tid] = soft64_mul(r.gas_constant[0],
        soft64_mul(r.virtual_temperature[tid], r.factor[tid]));
}
