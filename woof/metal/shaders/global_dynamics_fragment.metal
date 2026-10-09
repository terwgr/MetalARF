#include <metal_stdlib>
#include "global_dynamics_fragment_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;

#pragma clang fp contract(off)

// The host provides the already-contracted pressure, vorticity and metric terms.
kernel void global_dynamics_fragment(
    constant GlobalDynamicsFragmentResources &r [[buffer(0)]],
    constant GlobalDynamicsFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    r.tendency[tid] = soft64_add(
        soft64_sub(r.advective[tid], r.pressure_gradient[tid]),
        r.coriolis[tid]);
}
