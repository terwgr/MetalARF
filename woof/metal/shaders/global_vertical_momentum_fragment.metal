#include <metal_stdlib>
#include "global_vertical_momentum_fragment_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;

// needs integration: pointwise FP64 vertical momentum advection from
// MoistDycore._vertical_momentum_advection.
#pragma clang fp contract(off)

kernel void global_vertical_momentum_fragment(
    constant GlobalVerticalMomentumFragmentResources &r [[buffer(0)]],
    constant GlobalVerticalMomentumFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.count)) return;
    const uint nlev=uint(p.nlevel), ncolumn=uint(p.ncolumn);
    const uint tracer=tid/(nlev*ncolumn);
    const uint within=tid%(nlev*ncolumn);
    const uint level=within/ncolumn, col=within%ncolumn;
    const ulong source=ulong(tracer)*ulong(nlev)*ulong(ncolumn)+within;
    ulong derivative;
    if(level==0u) {
        const ulong next=source+ulong(ncolumn);
        derivative=soft64_div(soft64_sub(r.field[next],r.field[source]),
            soft64_sub(r.pressure_full[ulong(ncolumn)+col],r.pressure_full[col]));
    } else if(level+1u==nlev) {
        const ulong prev=source-ulong(ncolumn);
        derivative=soft64_div(soft64_sub(r.field[source],r.field[prev]),
            soft64_sub(r.pressure_full[ulong(level)*ulong(ncolumn)+col],
                       r.pressure_full[ulong(level-1u)*ulong(ncolumn)+col]));
    } else {
        const ulong prev=source-ulong(ncolumn), next=source+ulong(ncolumn);
        derivative=soft64_div(soft64_sub(r.field[next],r.field[prev]),
            soft64_sub(r.pressure_full[ulong(level+1u)*ulong(ncolumn)+col],
                       r.pressure_full[ulong(level-1u)*ulong(ncolumn)+col]));
    }
    const ulong upper=r.omega_half[ulong(level)*ulong(ncolumn)+col];
    const ulong lower=r.omega_half[ulong(level+1u)*ulong(ncolumn)+col];
    const ulong omega_full=soft64_mul(0x3fe0000000000000ul,soft64_add(upper,lower));
    r.advection[tid]=soft64_mul(omega_full,derivative);
}
