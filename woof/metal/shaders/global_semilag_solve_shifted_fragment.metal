// Batched small dense semi-implicit solve for Global sl_si spectral columns.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_solve_shifted_fragment_bindings.metalh"

using namespace metal;

kernel void global_semilag_solve_shifted_fragment(
    constant GlobalSemilagSolveShiftedFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagSolveShiftedFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint nmodes=uint(p.ndegree)*uint(p.mcount);
    const uint total=2u*nmodes;
    if(tid>=total)return;
    const uint component=tid/nmodes, mode=tid-component*nmodes;
    const uint degree=mode/uint(p.mcount), order=mode-degree*uint(p.mcount);
    const uint nx=uint(p.nlevel+1)*nmodes;
    const uint nd=uint(p.nlevel)*nmodes;
    const ulong tw=r.tau_weight[0], k2=r.k2[degree];
    if(degree==0u) {
        for(int row=0;row<p.nlevel;++row) {
            const uint index=2u*(uint(row)*nmodes+mode)+component;
            r.divergence_new[index]=r.divergence[index];
        }
        for(int row=0;row<p.nlevel;++row) {
            const uint index=2u*(uint(row)*nmodes+mode)+component;
            r.x_new[index]=r.theta[index];
        }
        const uint surface_index=2u*mode+component;
        r.x_new[2u*(uint(p.nlevel)*nmodes+mode)+component]=r.logps[surface_index];
        return;
    }
    thread ulong forcing[128];
    thread ulong solved[128];
    for(int row=0;row<p.nlevel;++row) {
        ulong value=r.divergence[2u*(uint(row)*nmodes+mode)+component];
        for(int col=0;col<p.nlevel+1;++col) {
            const ulong matrix=r.b_matrix[uint(row)*(p.nlevel+1)+uint(col)];
            const uint x_index = col < p.nlevel
                ? 2u*(uint(col)*nmodes+mode)+component
                : 2u*mode+component;
            const ulong x_value = col < p.nlevel
                ? r.theta[x_index]
                : r.logps[x_index];
            value=soft64_add(value,soft64_mul(
                soft64_mul(tw,k2),soft64_mul(matrix,
                    x_value)));
        }
        forcing[row]=value;
    }
    for(int row=0;row<p.nlevel;++row) {
        ulong value=0ul;
        for(int col=0;col<p.nlevel;++col)
            value=soft64_add(value,soft64_mul(
                r.inverse[(degree*uint(p.nlevel)+uint(row))*uint(p.nlevel)+uint(col)],
                forcing[col]));
        solved[row]=value;
    }
    (void)nd;
    (void)nx;
    for(int row=0;row<p.nlevel+1;++row) {
        ulong value = row < p.nlevel
            ? r.theta[2u*(uint(row)*nmodes+mode)+component]
            : r.logps[2u*mode+component];
        for(int col=0;col<p.nlevel;++col)
            value=soft64_add(value,soft64_mul(tw,soft64_mul(
                r.c_matrix[uint(row)*uint(p.nlevel)+uint(col)],solved[col])));
        r.x_new[2u*(uint(row)*nmodes+mode)+component]=value;
    }
    for(int row=0;row<p.nlevel;++row)
        r.divergence_new[2u*(uint(row)*nmodes+mode)+component]=solved[row];
    (void)order;
}
