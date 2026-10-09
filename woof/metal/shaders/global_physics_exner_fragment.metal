// Resident FP64 Exner synthesis for Global physics exchange.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "global_physics_exner_fragment_bindings.metalh"
using namespace metal;

kernel void global_physics_exner_fragment(
    constant GlobalPhysicsExnerFragmentResources &r [[buffer(0)]],
    constant GlobalPhysicsExnerFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong ratio = soft64_div(r.pressure[tid], 0x40f86a0000000000ul);
    r.exner[tid] = soft64_plm_pow(ratio, 0x3fd24b795eda4351ul);  // KAPPA = R/cp
}
