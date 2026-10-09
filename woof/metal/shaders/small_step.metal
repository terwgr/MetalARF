#include <metal_stdlib>
#include "small_step_init_uv_bindings.metalh"
#include "small_step_init_column_bindings.metalh"
#include "small_step_finish_uv_bindings.metalh"
#include "small_step_finish_column_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline int ss_prev(int i, int n, bool boundary) {
    return boundary ? max(i-1,0) : (i-1+n)%n;
}
static inline int ss_next(int i, int n, bool boundary) {
    return boundary ? min(i,n-1) : i%n;
}

kernel void small_step_init_uv(constant SmallStepInitUvResources &r [[buffer(0)]],
                               constant SmallStepInitUvParams &q [[buffer(1)]],
                               uint tid [[thread_position_in_grid]]) {
    const int nyf=q.ny+1, nxf=q.nx+1;
    const ulong su=ulong(q.ny)*ulong(nxf), sv=ulong(nyf)*ulong(q.nx);
    const ulong count=ulong(q.nz)*ulong(nyf)*ulong(nxf);
    if (ulong(tid)>=count) return;
    const int k=int(ulong(tid)/(ulong(nyf)*ulong(nxf)));
    const ulong rem=ulong(tid)-ulong(k)*ulong(nyf)*ulong(nxf);
    const int j=int(rem/ulong(nxf)), i=int(rem-ulong(j)*ulong(nxf));
    if(j<q.ny) {
        const int ia=ss_next(i,q.nx,bool(q.boundary_x));
        const int ib=ss_prev(i,q.nx,bool(q.boundary_x));
        const ulong ca=ulong(j)*ulong(q.nx)+ulong(ia), cb=ulong(j)*ulong(q.nx)+ulong(ib);
        const float mt_a=r.mub2d[ca]+r.mup0[ca], mt_b=r.mub2d[cb]+r.mup0[cb];
        const float ms_a=r.mub2d[ca]+r.mup[ca], ms_b=r.mub2d[cb]+r.mup[cb];
        const float mtf=0.5f*(mt_a+mt_b), msf=0.5f*(ms_a+ms_b);
        const float ct=r.c1h[k]*mtf+r.c2h[k], cs=r.c1h[k]*msf+r.c2h[k];
        const ulong ix=ulong(k)*su+ulong(j)*ulong(nxf)+ulong(i);
        float value=ct*r.u0[ix]-cs*r.u[ix];
        if(q.has_msf) value=value/r.msfu[ulong(j)*ulong(nxf)+ulong(i)];
        r.u_pp[ix]=value;
    }
    if(i<q.nx) {
        const int ja=ss_next(j,q.ny,bool(q.boundary_y));
        const int jb=ss_prev(j,q.ny,bool(q.boundary_y));
        const ulong ca=ulong(ja)*ulong(q.nx)+ulong(i), cb=ulong(jb)*ulong(q.nx)+ulong(i);
        const float mt_a=r.mub2d[ca]+r.mup0[ca], mt_b=r.mub2d[cb]+r.mup0[cb];
        const float ms_a=r.mub2d[ca]+r.mup[ca], ms_b=r.mub2d[cb]+r.mup[cb];
        const float mtf=0.5f*(mt_a+mt_b), msf=0.5f*(ms_a+ms_b);
        const float ct=r.c1h[k]*mtf+r.c2h[k], cs=r.c1h[k]*msf+r.c2h[k];
        const ulong ix=ulong(k)*sv+ulong(j)*ulong(q.nx)+ulong(i);
        float value=ct*r.v0[ix]-cs*r.v[ix];
        if(q.has_msf) value=value/r.msfv[ulong(j)*ulong(q.nx)+ulong(i)];
        r.v_pp[ix]=value;
    }
}

kernel void small_step_init_column(constant SmallStepInitColumnResources &r [[buffer(0)]],
                                   constant SmallStepInitColumnParams &q [[buffer(1)]],
                                   uint column_id [[thread_position_in_grid]]) {
    const ulong s=ulong(q.ny)*ulong(q.nx);
    if(ulong(column_id)>=s) return;
    const ulong c=ulong(column_id), bstr=q.base3d?s:1ul, boff=q.base3d?c:0ul;
    const float mut=r.mub2d[c]+r.mup0[c], mus=r.mub2d[c]+r.mup[c];
    const float mupp=r.mup0[c]-r.mup[c];
    r.mu_pp[c]=mupp;
    for(int k=0;k<=q.nz;++k) {
        const ulong f=ulong(k)*s+c;
        const float cft=r.c1f[k]*mut+r.c2f[k], cfs=r.c1f[k]*mus+r.c2f[k];
        float value=cft*r.w0[f]-cfs*r.w[f];
        if(q.has_msf) value=value/r.msft[c];
        r.w_pp[f]=value;
        r.ph_pp[f]=r.php0[f]-r.php[f];
    }
    for(int k=0;k<q.nz;++k) {
        const ulong h=ulong(k)*s+c, b=ulong(k)*bstr+boff;
        const float cht=r.c1h[k]*mut+r.c2h[k], chs=r.c1h[k]*mus+r.c2h[k];
        const float tht=r.thb[b]+r.thp0[h], ths=r.thb[b]+r.thp[h];
        const float thpp=cht*tht-chs*ths;
        r.th_pp[h]=thpp;
        const float c2a=q.gamma*r.p[h]/r.alt[h];
        const float al0=r.c1h[k]*mupp, al1=r.alt[h]*al0;
        const float dph=r.ph_pp[h+s]-r.ph_pp[h], al2=r.rdnw[k]*dph;
        const float al=-(al1+al2)/cht;
        r.al_pp[h]=al;
        const float p0=r.c1h[k]*mupp, p1=p0*ths, p2=thpp-p1;
        const float p3=r.alt[h]*p2, p4=cht*ths, p5=p3/p4;
        const float pnew=c2a*(p5-al);
        r.p_pp[h]=pnew; r.p_pp_old[h]=pnew;
    }
}

kernel void small_step_finish_uv(constant SmallStepFinishUvResources &r [[buffer(0)]],
                                 constant SmallStepFinishUvParams &q [[buffer(1)]],
                                 uint tid [[thread_position_in_grid]]) {
    const int nyf=q.ny+1, nxf=q.nx+1;
    const ulong su=ulong(q.ny)*ulong(nxf), sv=ulong(nyf)*ulong(q.nx);
    const ulong count=ulong(q.nz)*ulong(nyf)*ulong(nxf);
    if(ulong(tid)>=count) return;
    const int k=int(ulong(tid)/(ulong(nyf)*ulong(nxf)));
    const ulong rem=ulong(tid)-ulong(k)*ulong(nyf)*ulong(nxf);
    const int j=int(rem/ulong(nxf)), i=int(rem-ulong(j)*ulong(nxf));
    if(j<q.ny) {
        const int ia=ss_next(i,q.nx,bool(q.boundary_x));
        const int ib=ss_prev(i,q.nx,bool(q.boundary_x));
        const ulong ca=ulong(j)*ulong(q.nx)+ulong(ia), cb=ulong(j)*ulong(q.nx)+ulong(ib);
        const float msa=r.mub2d[ca]+r.mup[ca], msb=r.mub2d[cb]+r.mup[cb];
        const float mna=msa+r.mu_pp[ca], mnb=msb+r.mu_pp[cb];
        const float msf=0.5f*(msa+msb), mnf=0.5f*(mna+mnb);
        const float cs=r.c1h[k]*msf+r.c2h[k], cn=r.c1h[k]*mnf+r.c2h[k];
        const ulong ix=ulong(k)*su+ulong(j)*ulong(nxf)+ulong(i);
        float pp=r.u_pp[ix];
        if(q.has_msf) pp=pp*r.msfu[ulong(j)*ulong(nxf)+ulong(i)];
        r.u[ix]=(cs*r.u[ix]+pp)/cn;
    }
    if(i<q.nx) {
        const int ja=ss_next(j,q.ny,bool(q.boundary_y));
        const int jb=ss_prev(j,q.ny,bool(q.boundary_y));
        const ulong ca=ulong(ja)*ulong(q.nx)+ulong(i), cb=ulong(jb)*ulong(q.nx)+ulong(i);
        const float msa=r.mub2d[ca]+r.mup[ca], msb=r.mub2d[cb]+r.mup[cb];
        const float mna=msa+r.mu_pp[ca], mnb=msb+r.mu_pp[cb];
        const float msf=0.5f*(msa+msb), mnf=0.5f*(mna+mnb);
        const float cs=r.c1h[k]*msf+r.c2h[k], cn=r.c1h[k]*mnf+r.c2h[k];
        const ulong ix=ulong(k)*sv+ulong(j)*ulong(q.nx)+ulong(i);
        float pp=r.v_pp[ix];
        if(q.has_msf) pp=pp*r.msfv[ulong(j)*ulong(q.nx)+ulong(i)];
        r.v[ix]=(cs*r.v[ix]+pp)/cn;
    }
}

kernel void small_step_finish_column(constant SmallStepFinishColumnResources &r [[buffer(0)]],
                                     constant SmallStepFinishColumnParams &q [[buffer(1)]],
                                     uint column_id [[thread_position_in_grid]]) {
    const ulong s=ulong(q.ny)*ulong(q.nx);
    if(ulong(column_id)>=s) return;
    const ulong c=ulong(column_id), bstr=q.base3d?s:1ul, boff=q.base3d?c:0ul;
    const float mus=r.mub2d[c]+r.mup[c], mun=mus+r.mu_pp[c];
    for(int k=0;k<=q.nz;++k) {
        const ulong f=ulong(k)*s+c;
        float pp=r.w_pp[f];
        if(q.has_msf) pp=pp*r.msft[c];
        const float cs=r.c1f[k]*mus+r.c2f[k], cn=r.c1f[k]*mun+r.c2f[k];
        r.w[f]=(cs*r.w[f]+pp)/cn;
        r.php[f]=r.php[f]+r.ph_pp[f];
    }
    for(int k=0;k<q.nz;++k) {
        const ulong h=ulong(k)*s+c, b=ulong(k)*bstr+boff;
        const float cs=r.c1h[k]*mus+r.c2h[k];
        const float theta=r.thb[b]+r.thp[h];
        float numerator=cs*theta+r.th_pp[h];
        if(q.remove_hdiab) numerator=numerator-(q.hdiab_dt*cs)*r.h_diabatic[h];
        const float cn=r.c1h[k]*mun+r.c2h[k];
        r.thp[h]=numerator/cn-r.thb[b];
    }
    r.mup[c]=r.mup[c]+r.mu_pp[c];
}
