#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_small_step_init_uv_bindings.metalh"
#include "wrf_exact_small_step_init_column_bindings.metalh"
#include "wrf_exact_small_step_finish_uv_bindings.metalh"
#include "wrf_exact_small_step_finish_column_bindings.metalh"
using namespace metal;
static inline device const Exact32 *ss_words(device const float *p) {
    return reinterpret_cast<device const Exact32 *>(p);
}
static inline device Exact32 *ss_words(device float *p) {
    return reinterpret_cast<device Exact32 *>(p);
}
#pragma clang fp contract(off)

static inline int ss_prev(int i, int n, bool boundary) {
    return boundary ? max(i-1,0) : (i-1+n)%n;
}
static inline int ss_next(int i, int n, bool boundary) {
    return boundary ? min(i,n-1) : i%n;
}

kernel void wrf_exact_small_step_init_uv(constant WrfExactSmallStepInitUvResources &r [[buffer(0)]],
                               constant WrfExactSmallStepInitUvParams &q [[buffer(1)]],
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
        const Exact32 msf=0.5f*(((ss_words(r.mup)[ca]+ss_words(r.mup)[cb])+ss_words(r.mub2d)[ca])+ss_words(r.mub2d)[cb]);
        const Exact32 mtf=q.rk_step==1 ? msf : 0.5f*(((ss_words(r.mub2d)[ca]+ss_words(r.mup0)[ca])+ss_words(r.mub2d)[cb])+ss_words(r.mup0)[cb]);
        const Exact32 ct=ss_words(r.c1h)[k]*mtf+ss_words(r.c2h)[k], cs=ss_words(r.c1h)[k]*msf+ss_words(r.c2h)[k];
        const ulong ix=ulong(k)*su+ulong(j)*ulong(nxf)+ulong(i);
        Exact32 value=ct*ss_words(r.u0)[ix]-cs*ss_words(r.u)[ix];
        if(q.has_msf) value=value/ss_words(r.msfu)[ulong(j)*ulong(nxf)+ulong(i)];
        ss_words(r.u_pp)[ix]=value;
    }
    if(i<q.nx) {
        const int ja=ss_next(j,q.ny,bool(q.boundary_y));
        const int jb=ss_prev(j,q.ny,bool(q.boundary_y));
        const ulong ca=ulong(ja)*ulong(q.nx)+ulong(i), cb=ulong(jb)*ulong(q.nx)+ulong(i);
        const Exact32 msf=0.5f*(((ss_words(r.mup)[ca]+ss_words(r.mup)[cb])+ss_words(r.mub2d)[ca])+ss_words(r.mub2d)[cb]);
        const Exact32 mtf=q.rk_step==1 ? msf : 0.5f*(((ss_words(r.mub2d)[ca]+ss_words(r.mup0)[ca])+ss_words(r.mub2d)[cb])+ss_words(r.mup0)[cb]);
        const Exact32 ct=ss_words(r.c1h)[k]*mtf+ss_words(r.c2h)[k], cs=ss_words(r.c1h)[k]*msf+ss_words(r.c2h)[k];
        const ulong ix=ulong(k)*sv+ulong(j)*ulong(q.nx)+ulong(i);
        Exact32 value=ct*ss_words(r.v0)[ix]-cs*ss_words(r.v)[ix];
        if(q.has_msf) value=value*ss_words(r.msfv)[ulong(j)*ulong(q.nx)+ulong(i)];
        ss_words(r.v_pp)[ix]=value;
    }
}

kernel void wrf_exact_small_step_init_column(constant WrfExactSmallStepInitColumnResources &r [[buffer(0)]],
                                   constant WrfExactSmallStepInitColumnParams &q [[buffer(1)]],
                                   uint column_id [[thread_position_in_grid]]) {
    const ulong s=ulong(q.ny)*ulong(q.nx);
    if(ulong(column_id)>=s) return;
    const ulong c=ulong(column_id);
    const Exact32 mut=ss_words(r.mub2d)[c]+ss_words(r.mup0)[c], mus=ss_words(r.mub2d)[c]+ss_words(r.mup)[c];
    const Exact32 mupp=ss_words(r.mup0)[c]-ss_words(r.mup)[c];
    ss_words(r.mu_pp)[c]=mupp;
    for(int k=0;k<=q.nz;++k) {
        const ulong f=ulong(k)*s+c;
        const Exact32 cft=ss_words(r.c1f)[k]*mut+ss_words(r.c2f)[k], cfs=ss_words(r.c1f)[k]*mus+ss_words(r.c2f)[k];
        Exact32 value=cft*ss_words(r.w0)[f]-cfs*ss_words(r.w)[f];
        if(q.has_msf) value=value/ss_words(r.msft)[c];
        ss_words(r.w_pp)[f]=value;
        ss_words(r.ph_pp)[f]=ss_words(r.php0)[f]-ss_words(r.php)[f];
    }
    for(int k=0;k<q.nz;++k) {
        const ulong h=ulong(k)*s+c;
        const Exact32 cht=ss_words(r.c1h)[k]*mut+ss_words(r.c2h)[k], chs=ss_words(r.c1h)[k]*mus+ss_words(r.c2h)[k];
        const Exact32 tht=ss_words(r.thp0)[h], ths=ss_words(r.thp)[h];
        const Exact32 thpp=cht*tht-chs*ths;
        ss_words(r.th_pp)[h]=thpp;
        const Exact32 c2a=q.gamma*ss_words(r.p)[h]/ss_words(r.alt)[h];
        const Exact32 al0=ss_words(r.c1h)[k]*mupp, al1=ss_words(r.alt)[h]*al0;
        const Exact32 dph=ss_words(r.ph_pp)[h+s]-ss_words(r.ph_pp)[h], al2=ss_words(r.rdnw)[k]*dph;
        const Exact32 al=(Exact32(-1.0f)/cht)*(al1+al2);
        ss_words(r.al_pp)[h]=al;
        const Exact32 p0=ss_words(r.c1h)[k]*mupp, p1=p0*ths, p2=thpp-p1;
        const Exact32 p3=ss_words(r.alt)[h]*p2, p4=cht*(Exact32(300.0f)+ths), p5=p3/p4;
        const Exact32 pnew=c2a*(p5-al);
        ss_words(r.p_pp)[h]=pnew; ss_words(r.p_pp_old)[h]=pnew;
    }
}

kernel void wrf_exact_small_step_finish_uv(constant WrfExactSmallStepFinishUvResources &r [[buffer(0)]],
                                 constant WrfExactSmallStepFinishUvParams &q [[buffer(1)]],
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
        const Exact32 mna=ss_words(r.muts_wrf)[ca], mnb=ss_words(r.muts_wrf)[cb];
        const Exact32 msf=0.5f*(((ss_words(r.mup)[ca]+ss_words(r.mup)[cb])+ss_words(r.mub2d)[ca])+ss_words(r.mub2d)[cb]), mnf=0.5f*(mna+mnb);
        const Exact32 cs=ss_words(r.c1h)[k]*msf+ss_words(r.c2h)[k], cn=ss_words(r.c1h)[k]*mnf+ss_words(r.c2h)[k];
        const ulong ix=ulong(k)*su+ulong(j)*ulong(nxf)+ulong(i);
        Exact32 pp=ss_words(r.u_pp)[ix];
        if(q.has_msf) pp=pp*ss_words(r.msfu)[ulong(j)*ulong(nxf)+ulong(i)];
        ss_words(r.u)[ix]=(cs*ss_words(r.u)[ix]+pp)/cn;
    }
    if(i<q.nx) {
        const int ja=ss_next(j,q.ny,bool(q.boundary_y));
        const int jb=ss_prev(j,q.ny,bool(q.boundary_y));
        const ulong ca=ulong(ja)*ulong(q.nx)+ulong(i), cb=ulong(jb)*ulong(q.nx)+ulong(i);
        const Exact32 mna=ss_words(r.muts_wrf)[ca], mnb=ss_words(r.muts_wrf)[cb];
        const Exact32 msf=0.5f*(((ss_words(r.mup)[ca]+ss_words(r.mup)[cb])+ss_words(r.mub2d)[ca])+ss_words(r.mub2d)[cb]), mnf=0.5f*(mna+mnb);
        const Exact32 cs=ss_words(r.c1h)[k]*msf+ss_words(r.c2h)[k], cn=ss_words(r.c1h)[k]*mnf+ss_words(r.c2h)[k];
        const ulong ix=ulong(k)*sv+ulong(j)*ulong(q.nx)+ulong(i);
        Exact32 pp=ss_words(r.v_pp)[ix];
        if(q.has_msf) pp=pp*ss_words(r.msfv)[ulong(j)*ulong(q.nx)+ulong(i)];
        ss_words(r.v)[ix]=(cs*ss_words(r.v)[ix]+pp)/cn;
    }
}

kernel void wrf_exact_small_step_finish_column(constant WrfExactSmallStepFinishColumnResources &r [[buffer(0)]],
                                     constant WrfExactSmallStepFinishColumnParams &q [[buffer(1)]],
                                     uint column_id [[thread_position_in_grid]]) {
    const ulong s=ulong(q.ny)*ulong(q.nx);
    if(ulong(column_id)>=s) return;
    const ulong c=ulong(column_id);
    const Exact32 mus=ss_words(r.mub2d)[c]+ss_words(r.mup)[c], mun=ss_words(r.muts_wrf)[c];
    for(int k=0;k<=q.nz;++k) {
        const ulong f=ulong(k)*s+c;
        Exact32 pp=ss_words(r.w_pp)[f];
        if(q.has_msf) pp=pp*ss_words(r.msft)[c];
        const Exact32 cs=ss_words(r.c1f)[k]*mus+ss_words(r.c2f)[k], cn=ss_words(r.c1f)[k]*mun+ss_words(r.c2f)[k];
        ss_words(r.w)[f]=(cs*ss_words(r.w)[f]+pp)/cn;
        ss_words(r.php)[f]=ss_words(r.php)[f]+ss_words(r.ph_pp)[f];
    }
    for(int k=0;k<q.nz;++k) {
        const ulong h=ulong(k)*s+c;
        const Exact32 cs=ss_words(r.c1h)[k]*mus+ss_words(r.c2h)[k];
        const Exact32 theta=ss_words(r.thp)[h];
        Exact32 numerator=ss_words(r.th_pp)[h];
        if(q.remove_hdiab) numerator=numerator-(q.hdiab_dt*cs)*ss_words(r.h_diabatic)[h];
        numerator=numerator+theta*cs;
        const Exact32 cn=ss_words(r.c1h)[k]*mun+ss_words(r.c2h)[k];
        ss_words(r.thp)[h]=numerator/cn;
    }
    ss_words(r.mup)[c]=ss_words(r.mup)[c]+ss_words(r.mu_pp)[c];
}
