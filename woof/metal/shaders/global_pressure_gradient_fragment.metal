#include <metal_stdlib>
#include "global_pressure_gradient_fragment_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;

// needs integration: FP64 hybrid-pressure gradient factor used by the
// Global momentum force assembly.
#pragma clang fp contract(off)

kernel void global_pressure_gradient_fragment(
    constant GlobalPressureGradientFragmentResources &r [[buffer(0)]],
    constant GlobalPressureGradientFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.count)) return;
    const uint ncolumn=uint(p.ncolumn);
    const uint level=tid/ncolumn, col=tid%ncolumn;
    const ulong upper=soft64_div(r.b_upper[level],
        r.pressure_half[ulong(level)*ulong(ncolumn)+col]);
    const ulong lower=soft64_div(r.b_lower[level],
        r.pressure_half[ulong(level+1u)*ulong(ncolumn)+col]);
    const ulong mean=soft64_mul(0x3fe0000000000000ul,soft64_add(upper,lower));
    r.factor[tid]=soft64_mul(mean,r.surface_pressure[col]);
}
