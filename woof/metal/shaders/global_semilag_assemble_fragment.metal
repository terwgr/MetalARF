// FP64 assembly of the default Global SL bundle before trajectory gather.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_assemble_fragment_bindings.metalh"

using namespace metal;

kernel void global_semilag_assemble_fragment(
    constant GlobalSemilagAssembleFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagAssembleFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint npoint=uint(p.nlevel)*uint(p.nlat)*uint(p.nlon);
    if(tid>=npoint)return;
    const uint plane=uint(p.nlat)*uint(p.nlon);
    const uint level=tid/plane, cell=tid-level*plane;
    const ulong nu=soft64_sub(r.a_u[tid],r.l_u[tid]);
    const ulong nv=soft64_sub(r.a_v[tid],r.l_v[tid]);
    const ulong nt=soft64_negate_bits(r.l_theta[tid]);
    r.n_u[tid]=nu; r.n_v[tid]=nv; r.n_theta[tid]=nt;
    r.bundle_u[tid]=soft64_add(soft64_add(r.u[tid],
        soft64_mul(r.lead[0],r.l_u[tid])),soft64_mul(r.half_dt[0],nu));
    r.bundle_v[tid]=soft64_add(soft64_add(r.v[tid],
        soft64_mul(r.lead[0],r.l_v[tid])),soft64_mul(r.half_dt[0],nv));
    r.bundle_theta[tid]=soft64_add(soft64_add(r.theta[tid],
        soft64_mul(r.lead[0],r.l_theta[tid])),soft64_mul(r.half_dt[0],nt));
    if(level==uint(p.nlevel-1)) {
        const ulong nl=soft64_sub(r.a_lnps[cell],r.l_lnps[cell]);
        r.n_lnps[cell]=nl;
        r.bundle_lnps[cell]=soft64_add(soft64_add(r.logps[cell],
            soft64_mul(r.lead[0],r.l_lnps[cell])),soft64_mul(r.half_dt[0],nl));
    }
}
