#include <metal_stdlib>
#include "global_vertical_continuity_fragment_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;

// needs integration: move the hybrid-coordinate continuity prefix recurrence
// from woof/globe/vertical.py onto the typed Metal route.
#pragma clang fp contract(off)

kernel void global_vertical_continuity_fragment(
    constant GlobalVerticalContinuityFragmentResources &r [[buffer(0)]],
    constant GlobalVerticalContinuityFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong base=ulong(tid)*ulong(p.nlevel+1);
    ulong omega=0ul;
    r.omega[base]=0ul;
    for(uint level=0;level<uint(p.nlevel);++level) {
        const ulong source=ulong(tid)*ulong(p.nlevel)+ulong(level);
        omega=soft64_sub(omega,r.pressure_tendency[source]);
        omega=soft64_sub(omega,r.divergence_mass[source]);
        r.omega[base+ulong(level)+1ul]=omega;
    }
    r.residual[tid]=omega;
    // The closure residual is retained separately; WRF pins the top boundary.
    r.omega[base+ulong(p.nlevel)]=0ul;
}
