// Pointwise material tendencies in the default semi-Lagrangian Global step.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_tendencies_fragment_bindings.metalh"

using namespace metal;

kernel void global_semilag_tendencies_fragment(
    constant GlobalSemilagTendenciesFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagTendenciesFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint npoint=uint(p.nlevel)*uint(p.nlat)*uint(p.nlon);
    if(tid>=npoint)return;
    const uint plane=uint(p.nlat)*uint(p.nlon);
    const uint level=tid/plane, cell=tid-level*plane;
    const ulong coriolis=r.coriolis[cell];
    r.a_u[tid]=soft64_sub(soft64_sub(
        soft64_mul(coriolis,r.v[tid]),r.grad_phi_east[tid]),
        soft64_mul(r.force[tid],r.grad_lnps_east[cell]));
    r.a_v[tid]=soft64_sub(soft64_sub(
        soft64_negate_bits(soft64_mul(coriolis,r.u[tid])),r.grad_phi_north[tid]),
        soft64_mul(r.force[tid],r.grad_lnps_north[cell]));
    if(level==uint(p.nlevel-1)) {
        r.a_lnps[cell]=soft64_add(
            soft64_div(r.ps_t[cell],r.ps[cell]),
            soft64_add(soft64_mul(r.u[tid],r.grad_lnps_east[cell]),
                       soft64_mul(r.v[tid],r.grad_lnps_north[cell])));
    }
}
