#include <metal_stdlib>
#include "acoustic_boundary_frame_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void acoustic_boundary_frame(
    constant AcousticBoundaryFrameResources &r [[buffer(0)]],
    constant AcousticBoundaryFrameParams &q [[buffer(1)]],
    uint column_id [[thread_position_in_grid]]) {
    const uint ny=uint(q.ny), nx=uint(q.nx), nz=uint(q.nz);
    const ulong s=ulong(ny)*nx;
    if (ulong(column_id)>=s) return;
    const int j=int(column_id)/q.nx, i=int(column_id)-j*q.nx;
    if (i>=q.spec_zone && i<q.nx-q.spec_zone &&
        j>=q.spec_zone && j<q.ny-q.spec_zone) return;

    const ulong c=ulong(column_id);
    const float mut=r.mub2d[c]+r.mup[c];
    const float muts=mut+r.mu_pp[c];
    const float dt_rmu=q.dtau*r.rmu_t[c];
    const float mu_old=muts-dt_rmu;
    const int source_j=min(max(j,q.spec_zone),q.ny-1-q.spec_zone);
    const int source_i=min(max(i,q.spec_zone),q.nx-1-q.spec_zone);
    const ulong source_c=ulong(source_j)*nx+ulong(source_i);

    for (uint k=0;k<=nz;++k) {
        const ulong tid=ulong(k)*s+c;
        const float numerator=r.c1f[k]*mu_old+r.c2f[k];
        const float denominator=r.c1f[k]*muts+r.c2f[k];
        const float ratio=numerator/denominator;
        const float value0=r.ph_pp[tid]*ratio;
        const float value1=q.dtau*r.rph_t[tid];
        const float denominator2=r.c1f[k]*muts+r.c2f[k];
        const float value2=value1/denominator2;
        const float value3=value0+value2;
        const float value4=ratio-1.0f;
        const float value5=r.php[tid]*value4;
        r.ph_pp[tid]=value3+value5;
        if (q.table_w) {
            r.w_pp[tid]=r.w_pp[tid]+q.dtau*r.rw_t[tid];
        } else {
            r.w_pp[tid]=r.w_pp[ulong(k)*s+source_c];
        }
    }

    for (uint k=0;k<nz;++k) {
        const ulong tid=ulong(k)*s+c;
        const float chm=r.c1h[k]*(r.mub2d[c]+r.mup[c]+r.mu_pp[c])+r.c2h[k];
        const float th_ref=r.thb[q.base3d?tid:ulong(k)]+r.thp[tid];
        const float al=-(r.alt[tid]*(r.c1h[k]*r.mu_pp[c])
                        +r.rdnw[k]*(r.ph_pp[tid+s]-r.ph_pp[tid]))/chm;
        r.al_pp[tid]=al;
        r.p_pp[tid]=r.c2a[tid]*(r.alt[tid]
            *(r.th_pp[tid]-r.c1h[k]*r.mu_pp[c]*th_ref)
            /(chm*th_ref)-al);
    }
}
