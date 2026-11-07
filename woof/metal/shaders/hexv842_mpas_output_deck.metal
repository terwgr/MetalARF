// Numerical Metal derivative of MPAS diagnostics; original source remains unchanged.
// Copyright (c) 2016, Los Alamos National Security, LLC (LANS)
// and the University Corporation for Atmospheric Research (UCAR).
// MPAS distribution license:
// Copyright (c) 2013-2019,  Los Alamos National Security, LLC (LANS) (Ocean: LA-CC-13-047;
// Land Ice: LA-CC-13-117) and the University Corporation for Atmospheric Research (UCAR).
//
// All rights reserved.
//
// LANS is the operator of the Los Alamos National Laboratory under Contract No.
// DE-AC52-06NA25396 with the U.S. Department of Energy.  UCAR manages the National
// Center for Atmospheric Research under Cooperative Agreement ATM-0753581 with the
// National Science Foundation.  The U.S. Government has rights to use, reproduce,
// and distribute this software.  NO WARRANTY, EXPRESS OR IMPLIED IS OFFERED BY
// LANS, UCAR OR THE GOVERNMENT AND NONE OF THEM ASSUME ANY LIABILITY FOR THE USE
// OF THIS SOFTWARE.  If software is modified to produce derivative works, such
// modified software should be clearly marked, so as not to confuse it with the
// version available from LANS and UCAR.
//
// Additionally, redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1) Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2) Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
//
// 3) None of the names of LANS, UCAR or the names of its contributors, if any, may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include <metal_stdlib>
using namespace metal;
#include "cumulus_support.metalh"
#include "hexv842_mpas_output_deck_bindings.metalh"
#pragma clang fp contract(off)
#define __fadd_rn(a,b) ((a)+(b))
#define __fsub_rn(a,b) ((a)-(b))
#define __fmul_rn(a,b) ((a)*(b))
#define __fdiv_rn(a,b) cu_div((a),(b))
#define __fsqrt_rn(a) cu_sqrt(a)
// Contracted expf from flt32_expf_fma.cuh; binary64 operations use Soft64.
static inline float deck_exp_fma(float value) {
    const uint bits=as_type<uint>(value);
    const uint abstop=(bits>>20u)&0x7ffu;
    if (abstop>=0x42bu) return as_type<float>(soft_gfk_exp(bits));
    const ulong xd=soft64_from_f32_bits(bits);
    const ulong inv=0x40471547652b82feul, shift=0x4338000000000000ul;
    ulong kd=soft64_fma(inv,xd,shift);
    kd=soft64_sub(kd,shift);
    const ulong r=soft64_fma(inv,xd,kd^0x8000000000000000ul);
    const ulong scale=soft_gfk_exp2_core(kd,shift,0ul,0ul,0ul,0ul);
    const ulong z=soft64_fma(0x3ebc6af84b912394ul,r,0x3f2ebfce50fac4f3ul);
    const ulong r2=soft64_mul(r,r);
    ulong y=soft64_fma(0x3f962e42ff0c52d6ul,r,0x3ff0000000000000ul);
    y=soft64_mul(soft64_fma(z,r2,y),scale);
    return as_type<float>(soft64_to_f32_bits(y));
}

#define gfk_exp_fma deck_exp_fma

// MPAS-A v8.4.2. RKIND is REAL(4).
// Elementary functions come from the engine's shared glibc device headers.
#define A(a,b) __fadd_rn((a),(b))
#define S(a,b) __fsub_rn((a),(b))
#define M(a,b) __fmul_rn((a),(b))
#define D(a,b) __fdiv_rn((a),(b))

// mpas_isobaric_diagnostics.F:1005-1091, including the collective kupper
// search and exact-top equality behavior. One cooperative block handles
// arbitrary ncol with strided work, preserving native kount semantics.
kernel void hexv842_deck_fixed_pressure(constant Hexv842DeckFixedPressureResources &r [[buffer(0)]],constant Hexv842DeckFixedPressureParams &abi [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=1) return;
const int nc=abi.nc;
const int ni=abi.ni;
const int no=abi.no;
device const float* p=r.p;
device const float* f=r.f;
device const float* po=r.po;
device float* out=r.out;
device int* upper=r.upper;
    int t=0;
    int start,count,all;
    for(int c=t;c<nc;c+=1) upper[c]=0;

    for(int k=0;k<no;k++) {
        if(t==0) {start=ni-1;count=0;all=0;}

        for(int c=t;c<nc;c+=1) start=min(start,upper[c]);

        for(int kk=start;kk<ni-1;kk++) {
            for(int c=t;c<nc;c+=1) {
                float q=po[c*no+k];
                if(q>p[c*ni+kk] && q<=p[c*ni+kk+1]) {
                    upper[c]=kk;count+=1;
                }
            }

            if(count==nc) {if(t==0) all=1;break;}
        }

        for(int c=t;c<nc;c+=1) {
            float q=po[c*no+k]; int u=upper[c];
            if(!all && q<p[c*ni]) out[c*no+k]=D(M(f[c*ni],q),p[c*ni]);
            else if(!all && q>p[c*ni+ni-1]) out[c*no+k]=f[c*ni+ni-1];
            else {
                float dpu=S(q,p[c*ni+u]),dpl=S(p[c*ni+u+1],q);
                out[c*no+k]=D(A(M(f[c*ni+u],dpl),M(f[c*ni+u+1],dpu)),A(dpl,dpu));
            }
        }

    }

}


// The same routine across the whole grid (fixed_pressure is one block, kept as
// the referee). Per output level k, with start[k] = min over columns of the
// carried kupper: count[k*ni+kk] counts bracket hits at kk; a one-thread scan
// finds the first kk whose running count equals ncol exactly (native kount);
// apply re-finds each column's last hit through that kk, interpolates and
// folds its kupper into start[k+1]. Same words as the one-block kernel.




// mpas_isobaric_diagnostics.F:1245-1351.
kernel void hexv842_deck_layer_mean(constant Hexv842DeckLayerMeanResources &r [[buffer(0)]],constant Hexv842DeckLayerMeanParams &abi [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(abi.dispatch_count)) return;
const int nc=abi.nc;
const int nl=abi.nl;
const float p1=abi.p1;
const float p2=abi.p2;
device const float* f=r.f;
device const float* p=r.p;
device float* out=r.out;
    int c=int(gid);if(c>=nc)return;
    if(p[0]>p[nl-1]) {out[c]=0.f;return;}
    float pt=p1<p2?p1:p2,pb=p1<p2?p2:p1;
    int kt=-1,kb=-1;
    float wtp=0,wtm=0,wbp=0,wbm=0;
    device const float* pc=p+c*nl; device const float* fc=f+c*nl;
    for(int k=0;k<nl-1;k++) {
        if(pc[k]<=pt && pc[k+1]>pt) {
            kt=k;wtp=D(S(pt,pc[k]),S(pc[k+1],pc[k]));
            wtm=D(S(pc[k+1],pt),S(pc[k+1],pc[k]));
        }
        if(pc[k]<=pb && pc[k+1]>pb) {
            kb=k;wbm=D(S(pb,pc[k]),S(pc[k+1],pc[k]));
            wbp=D(S(pc[k+1],pb),S(pc[k+1],pc[k]));
        }
    }
    if(kt==-1 || kb==-1) {out[c]=0;return;}
    float x=A(M(wtm,fc[kt]),M(wtp,fc[kt+1]));
    if(kt==kb) {
        x=A(A(x,M(wbm,fc[kb])),M(wbp,fc[kb+1]));
        out[c]=M(.5f,x);return;
    }
    float total=S(pc[kt+1],pt);
    x=M(M(total,.5f),A(fc[kt+1],x));
    for(int k=kt+1;k<kb;k++) {
        float w=S(pc[k+1],pc[k]); total=A(total,w);
        x=A(x,M(M(w,.5f),A(fc[k],fc[k+1])));
    }
    float w=S(pb,pc[kb]); total=A(total,w);
    float temp=A(M(wbm,fc[kb]),M(wbp,fc[kb+1]));
    x=A(x,M(M(w,.5f),A(fc[kb],temp)));
    out[c]=D(x,total);

}


// mpas_isobaric_diagnostics.F:1094-1227. Inputs are column-major C arrays
// (column,level); p is hPa. A failed column marks the entire output invalid.
kernel void hexv842_deck_slp_columns(constant Hexv842DeckSlpColumnsResources &r [[buffer(0)]],constant Hexv842DeckSlpColumnsParams &abi [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(abi.dispatch_count)) return;
const int nc=abi.nc;
const int nl=abi.nl;
device const float* p=r.p;
device const float* temp=r.temp;
device const float* h=r.h;
device const float* q=r.q;
device float* out=r.out;
device int* failed=r.failed;
    int c=int(gid);if(c>=nc)return;
    device const float* pc=p+c*nl;device const float* tc=temp+c*nl;
    device const float* hc=h+c*(nl+1);device const float* qc=q+c*nl;
    int level=-1;
    for(int k=0;k<nl;k++)if(pc[k]<S(pc[0],100.f)){level=k;break;}
    if(level==-1){atomic_exchange_explicit((device atomic_int*)failed,1,memory_order_relaxed);return;}
    int lo=max(level-1,0),hi=min(lo+1,nl-2);
    if(lo==hi){atomic_exchange_explicit((device atomic_int*)failed,1,memory_order_relaxed);return;}
    float plo=pc[lo],phi=pc[hi];
    float tlo=M(tc[lo],A(1.f,M(.608f,qc[lo])));
    float thi=M(tc[hi],A(1.f,M(.608f,qc[hi])));
    float zlo=M(.5f,A(hc[lo],hc[lo+1]));
    float zhi=M(.5f,A(hc[hi],hc[hi+1]));
    float pp=S(pc[0],100.f);
    float lp=gfk_log(D(pp,phi)),ll=gfk_log(D(plo,phi));
    float tp=S(thi,M(M(S(thi,tlo),lp),ll));
    float zp=S(zhi,M(M(S(zhi,zlo),lp),ll));
    float ts=M(tp,gfk_pow(D(pc[0],pp),D(M(.0065f,287.f),9.80616f)));
    float tm=A(tp,M(.0065f,zp));
    const float limit=290.66f;
    if(ts<=limit && tm>=limit) tm=limit;
    else {float delta=S(ts,limit);tm=S(limit,M(.005f,M(delta,delta)));}
    float zhalf=M(.5f,A(hc[0],hc[1]));
    out[c]=M(pc[0],gfk_exp_fma(D(M(19.61232f,zhalf),M(287.f,A(tm,ts)))));

}


kernel void hexv842_deck_slp_finish(constant Hexv842DeckSlpFinishResources &r [[buffer(0)]],constant Hexv842DeckSlpFinishParams &abi [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(abi.dispatch_count)) return;
const int nc=abi.nc;
device float* out=r.out;
device const int* failed=r.failed;
const float scale=abi.scale;
    int c=int(gid);
    if(c<nc)out[c]=*failed?0.f:M(out[c],scale);

}


// mpas_isobaric_diagnostics.F:583-594 and 643-658.
kernel void hexv842_deck_prepare_mass(constant Hexv842DeckPrepareMassResources &r [[buffer(0)]],constant Hexv842DeckPrepareMassParams &abi [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(abi.dispatch_count)) return;
const int nc=abi.nc;
const int nl=abi.nl;
const int stride=abi.stride;
const int thermo=abi.thermo;
device const float* pp=r.pp;
device const float* pb=r.pb;
device const float* theta=r.theta;
device const float* exner=r.exner;
device const float* q=r.q;
device float* p=r.p;
device float* temp=r.temp;
device float* dew=r.dew;
    int x=int(gid);if(x>=nc*nl)return;
    int c=x%nc,k=x/nc,s=k*stride+c;
    p[x]=D(A(pp[s],pb[s]),100.f);
    if(thermo && c<nc-1) {
        // rvord is the native constant folded from 461.6_RKIND / 287_RKIND.
        temp[k*(nc-1)+c]=M(D(theta[s],A(1.f,M(0x1.9bbda4p+0f,q[s]))),exner[s]);
        float evp=D(M(p[x],q[s]),A(q[s],.622f));evp=max(evp,1.e-8f);
        float l=gfk_log(D(evp,6.112f));
        dew[k*(nc-1)+c]=A(D(M(243.5f,l),S(17.67f,l)),273.15f);
    }

}


// mpas_isobaric_diagnostics.F:596-628. Heights are (level,cell).
kernel void hexv842_deck_prepare_interfaces(constant Hexv842DeckPrepareInterfacesResources &r [[buffer(0)]],constant Hexv842DeckPrepareInterfacesParams &abi [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(abi.dispatch_count)) return;
const int nc=abi.nc;
const int nl=abi.nl;
const int hs=abi.hs;
device const float* p=r.p;
device const float* h=r.h;
device float* pi=r.pi;
    int x=int(gid);if(x>=nc*(nl+1))return;
    int c=x%nc,k=x/nc;float w1,w2,pa,pb;
    if(k==nl || k==0) {
        int a=k==nl?nl-1:0,b=k==nl?nl-2:1;
        float z0=h[k*hs+c];
        float z1=M(.5f,A(h[k*hs+c],h[(k==nl?k-1:k+1)*hs+c]));
        float z2=M(.5f,A(h[(k==nl?k-1:k+1)*hs+c],h[(k==nl?k-2:k+2)*hs+c]));
        w1=D(S(z0,z2),S(z1,z2));w2=S(1.f,w1);
        pa=p[a*(nc+1)+c];pb=p[b*(nc+1)+c];
    } else {
        float den=S(h[(k+1)*hs+c],h[(k-1)*hs+c]);
        w1=D(S(h[k*hs+c],h[(k-1)*hs+c]),den);
        w2=D(S(h[(k+1)*hs+c],h[k*hs+c]),den);
        pa=p[k*(nc+1)+c];pb=p[(k-1)*(nc+1)+c];
    }
    pi[x]=gfk_exp_fma(A(M(w1,gfk_log(pa)),M(w2,gfk_log(pb))));

}


// mpas_isobaric_diagnostics.F:630-641. Native 1-based connectivity includes
// its nCells+1 sentinel; do not discard that pressure contribution.
kernel void hexv842_deck_prepare_vertices(constant Hexv842DeckPrepareVerticesResources &r [[buffer(0)]],constant Hexv842DeckPrepareVerticesParams &abi [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(abi.dispatch_count)) return;
const int nc=abi.nc;
const int nv=abi.nv;
const int nl=abi.nl;
const int degree=abi.degree;
device const int* cov=r.cov;
device const float* kite=r.kite;
device const float* area=r.area;
device const float* p=r.p;
device float* pv=r.pv;
    int x=int(gid);if(x>=nv*nl)return;
    int v=x%nv,k=x/nv;float sum=0.f;
    for(int d=0;d<degree;d++)sum=A(sum,M(kite[d*nv+v],p[k*(nc+1)+cov[d*nv+v]-1]));
    pv[x]=D(sum,area[v]);

}


// mpas_atm_core.F:916-965. Driver calls after advancing the clock and
// refreshing state level 1, before diagnostics manager compute and stream write.



struct R {
    float v;
    R() {}
    R(float x): v(x) {}
    thread R& operator+=(R x) { v=__fadd_rn(v,x.v); return *this; }
};
R operator+(R a,R b) {return __fadd_rn(a.v,b.v);}
R operator-(R a,R b) {return __fsub_rn(a.v,b.v);}
R operator*(R a,R b) {return __fmul_rn(a.v,b.v);}
R operator/(R a,R b) {return __fdiv_rn(a.v,b.v);}
R operator-(R a) {return -a.v;}
bool operator<(R a,R b) {return a.v<b.v;}
bool operator>(R a,R b) {return a.v>b.v;}
bool operator<=(R a,R b) {return a.v<=b.v;}
bool operator>=(R a,R b) {return a.v>=b.v;}
R rmax(R a,R b) {return a.v>b.v?a:b;}
R rmin(R a,R b) {return a.v<b.v?a:b;}
R rabs(R a) {return metal::abs(a.v);}
R rlog(R a) {return gfk_log(a.v);}
R rexp(R a) {return gfk_exp_fma(a.v);}
R rpow(R a,R b) {return gfk_pow(a.v,b.v);}
R rsqrt(R a) {return __fsqrt_rn(a.v);}

// Column access in the pools' native (level, cell) layout: element k of a cell's column is p[k*s].
struct S {
    device const float* p; long s;
    float operator[](int k) const {return p[(long)k*s];}
};
// A writable level-interleaved per-cell work column.
struct W {
    device float* p; long s;
    device float& operator[](int k) const {return p[(long)k*s];}
};

// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1127-1143
template<class X,class Z> R column_height_value(X x,Z z,R zi,int n) {
    int kz=0;
    for(int k=0;k<n;k++) if(R(z[k])<=zi) kz=k;
    kz=kz<n-1?kz:n-2;
    R wz=(R(z[kz+1])-zi)/(R(z[kz+1])-R(z[kz]));
    R wzp1=R(1.0f)-wz;
    return wz*R(x[kz])+wzp1*R(x[kz+1]);
}
// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1168-1184
template<class X,class Z> R integral_zstaggered(X x,Z z,R bot,R top,int n) {
    R result=0.0f;
    for(int k=0;k<n;k++) {
        R zb=rmax(z[k],bot),zt=rmin(z[k+1],top);
        result+=R(x[k])*rmax(0.0f,zt-zb);
    }
    return result;
}
// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1189-1209
template<class X,class Z> R integral_zpoint(X x,Z z,R bot,R top,int n) {
    R result=0.0f;
    for(int k=0;k<n-1;k++) {
        R zb=rmax(z[k],bot),zt=rmin(z[k+1],top);
        R dz=rmax(0.0f,zt-zb);
        R mid=(R(0.5f)*(zt+zb)-R(z[k]))/(R(z[k+1])-R(z[k]));
        R val=R(x[k])+(R(x[k+1])-R(x[k]))*mid;
        result+=dz*val;
    }
    return result;
}
// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1688-1699
R getqvs(R p,R t) {
    const R eps=287.04f/461.5f;
    R es=R(611.2f)*rexp(R(17.67f)*(t-R(273.15f))/(t-R(29.65f)));
    return eps*es/(p-es);
}
// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1705-1716
R getqvi(R p,R t) {
    const R eps=287.04f/461.5f;
    R es=R(611.2f)*rexp(R(21.8745584f)*(t-R(273.15f))/(t-R(7.66f)));
    return eps*es/(p-es);
}
// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1722-1738
R getthe(R p,R t,R td,R q) {
    R tlcl;
    if((td-t)>=R(-0.1f)) tlcl=t;
    else tlcl=R(56.0f)+R(1.0f)/(R(1.0f)/(td-R(56.0f))+R(0.00125f)*rlog(t/td));
    return t*rpow(R(100000.0f)/p,R(0.2854f)*(R(1.0f)-R(0.28f)*q))
        *rexp((R(3376.0f)/tlcl-R(2.54f))*q*(R(1.0f)+R(0.81f)*q));
}
// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1240-1682
// source=2 (most unstable), adiabat=1 (pseudoadiabatic, liquid only). One thread per column; the column's work
// arrays are level-interleaved accessors (W). Same operations in the same order as the native routine, with three
// changes of schedule that leave every word as it was:
//  * The pressure path of the ascent (p2 -= dp per substep) depends only on the level pressures, so pi2 =
//    (p2/p00)^(rd/cp) and log(p2/p1) of every substep are formed first (prep[s], prep[cap+s]) and read back by
//    the ascent; log(p2/p1) is the loop-invariant of the native iteration, evaluated once per substeabi.
//  * The ascent is one flat loop of saturation iterations (a lane finishing a substep or level moves on while
//    its neighbours keep iterating), so the 32 columns of a warp no longer serialise on each other's loop exits.
//  * The ice-saturation term enters only as 0*qvi. When qvi is certainly finite (es_ice < 0.999 p2 by a 1e-5
//    accurate bound, p2 > 0), 0*qvi is a signed zero and qvs (>= +0 there) + that zero is qvs, so qvi is not
//    formed; otherwise it is formed and added exactly as native does (an infinite or NaN qvi still poisons qv2).
template<class A> void getcape_cols(int nk,A pin,A tin,A tdin,A p,A t,A td,A pi,A q,A th,A thv,
                                               A prep,int cap,thread float& cape_out,thread float& cin_out) {
    const R rd=287.04f,rv=461.5f,cp=1005.7f,g=9.81f,cpv=1875.0f,cpl=4190.0f,cpi=2118.636f;
    const R rp00=1.0f/100000.0f,reps=461.5f/287.04f,rddcp=287.04f/1005.7f,cpdg=1005.7f/9.81f;
    const R lv1=2501000.0f+(4190.0f-1875.0f)*273.15f,lv2=4190.0f-1875.0f;
    const R ls1=2836017.0f+(2118.636f-1875.0f)*273.15f,ls2=2118.636f-1875.0f;
    for(int k=0;k<nk;k++) {
        p[k]=(R(100.0f)*R(pin[k])).v;
        t[k]=(R(273.15f)+R(tin[k])).v;
        td[k]=(R(273.15f)+R(tdin[k])).v;
        pi[k]=rpow(R(p[k])*rp00,rddcp).v;
        q[k]=getqvs(p[k],td[k]).v;
        th[k]=(R(t[k])/R(pi[k])).v;
        thv[k]=(R(th[k])*(R(1.0f)+reps*R(q[k]))/(R(1.0f)+R(q[k]))).v;
    }
    int kmax=0;
    if(R(p[0])>=R(50000.0f)) {
        R maxthe=0.0f;
        for(int k=0;k<nk;k++) if(R(p[k])>=R(50000.0f)) {
            R the=getthe(p[k],t[k],td[k],q[k]);
            if(the>maxthe) {maxthe=the;kmax=k;}
        }
    }
    // Pressure path of every substep the ascent can take (it may stop earlier), up to cap substeps.
    int prepared=0;
    {
        R p2=p[kmax];int s=0;
        for(int k=kmax+1;k<nk&&s<cap;k++) {
            R dp=R(p[k-1])-R(p[k]);
            int nloop=1;
            if(dp>=R(100.0f)) {nloop=1+int((dp/R(100.0f)).v);dp=dp/R(float(nloop));}
            for(int n=0;n<nloop&&s<cap;n++,s++) {
                R p1=p2;p2=p2-dp;
                prep[s]=rpow(p2*rp00,rddcp).v;prep[cap+s]=rlog(p2/p1).v;
            }
        }
        prepared=s;
    }
    R narea=0.0f,cape=0.0f,cin=0.0f;
    R th2=th[kmax],pi2=pi[kmax],p2=p[kmax],t2=t[kmax],thv2=thv[kmax],qv2=q[kmax],b2=0.0f;
    R ql2=0.0f,qi2=0.0f,qt=qv2;
    int k=kmax+1;
    if(k>=nk) {cape_out=cape.v;cin_out=cin.v;return;}
    // Level k and its first substeabi.
    R b1=b2,dp=R(p[k-1])-R(p[k]);
    int nloop=1,n=0,s=0,i=0;
    if(dp>=R(100.0f)) {nloop=1+int((dp/R(100.0f)).v);dp=dp/R(float(nloop));}
    R p1=p2,t1=t2,th1=th2,qv1=qv2,ql1=ql2,qi1=qi2,lg;
    p2=p2-dp;
    if(s<cap) {pi2=prep[s];lg=prep[cap+s];} else {pi2=rpow(p2*rp00,rddcp);lg=rlog(p2/p1);}
    R thlast=th1;
    // Keep this level's environmental words in registers throughout its ascent.
    // Read the following substep before the saturation chain starts; volatile loads
    // keep the compiler from sinking them back into the converged branch.
    R env_thv=thv[k],prev_thv=thv[k-1],env_pi=pi[k],prev_pi=pi[k-1],env_p=p[k];
    R next_pi=0.0f,next_lg=0.0f;
    while(true) {
        if(i==0 && s+1<prepared) {
            next_pi=prep[s+1];
            next_lg=prep[cap+s+1];
        }
        i++;
        t2=thlast*pi2;
        const R qvs=getqvs(p2,t2);
        const float targ=(R(21.8745584f)*(t2-R(273.15f))/(t2-R(7.66f))).v;
        if(p2.v>0.0f&&611.2f*metal::exp(targ)<0.999f*p2.v) qv2=rmin(qt,R(1.0f)*qvs);
        else qv2=rmin(qt,R(1.0f)*qvs+R(0.0f)*getqvi(p2,t2));
        qi2=rmax(R(0.0f)*(qt-qv2),0.0f);
        ql2=rmax(qt-qv2-qi2,0.0f);
        R tbar=R(0.5f)*(t1+t2),qvbar=R(0.5f)*(qv1+qv2);
        R qlbar=R(0.5f)*(ql1+ql2),qibar=R(0.5f)*(qi1+qi2);
        R lhv=lv1-lv2*tbar,lhs=ls1-ls2*tbar;
        R rm=rd+rv*qvbar,cpm=cp+cpv*qvbar+cpl*qlbar+cpi*qibar;
        th2=th1*rexp(lhv*(ql2-ql1)/(cpm*tbar)+lhs*(qi2-qi1)/(cpm*tbar)
                     +(rm/cpm-rd/cp)*lg);
        // Native returns the accumulated CAPE/CIN on nonconvergence.
        if(i>100) {cape_out=cape.v;cin_out=cin.v;return;}
        if(rabs(th2-thlast)>R(0.0002f)) {thlast=thlast+R(0.3f)*(th2-thlast);continue;}
        // The substep converged.
        qt=qv2;ql2=0.0f;qi2=0.0f;
        n++;s++;
        if(n>=nloop) {
            // The level is done.
            thv2=th2*(R(1.0f)+reps*qv2)/(R(1.0f)+qv2+ql2+qi2);
            b2=g*(thv2-env_thv)/env_thv;
            R dz=-cpdg*R(0.5f)*(env_thv+prev_thv)*(env_pi-prev_pi);
            R parea;
            if(b2>=R(0.0f)&&b1<R(0.0f)) {
                R frac=b2/(b2-b1);
                parea=R(0.5f)*b2*dz*frac;
                narea=narea-R(0.5f)*b1*dz*(R(1.0f)-frac);
                cin+=narea;narea=0.0f;
            } else if(b2<R(0.0f)&&b1>R(0.0f)) {
                R frac=b1/(b1-b2);
                parea=R(0.5f)*b1*dz*frac;
                narea=-R(0.5f)*b2*dz*(R(1.0f)-frac);
            } else if(b2<R(0.0f)) {
                parea=0.0f;narea=narea-R(0.5f)*dz*(b1+b2);
            } else {parea=R(0.5f)*dz*(b1+b2);narea=0.0f;}
            cape+=rmax(0.0f,parea);
            if(env_p<=R(10000.0f)&&b2<R(0.0f)) break;
            if(++k>=nk) break;
            prev_thv=env_thv;prev_pi=env_pi;
            env_thv=thv[k];env_pi=pi[k];
            R prev_p=env_p;env_p=p[k];
            b1=b2;dp=prev_p-env_p;
            nloop=1;n=0;
            if(dp>=R(100.0f)) {nloop=1+int((dp/R(100.0f)).v);dp=dp/R(float(nloop));}
        }
        // Next substeabi.
        p1=p2;t1=t2;th1=th2;qv1=qv2;ql1=ql2;qi1=qi2;
        p2=p2-dp;
        if(s<cap) {pi2=next_pi;lg=next_lg;} else {pi2=rpow(p2*rp00,rddcp);lg=rlog(p2/p1);}
        thlast=th1;i=0;
    }
    cape_out=cape.v;cin_out=cin.v;
}
// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:710-970
// One thread per column of this launch's slice [c0, c0+n) of the solve cells, 32-thread blocks (one warp per SM:
// the parcel ascent is a long serial chain through FP64 libm, so it is spread over as many SMs as there are
// warps). Inputs are the pools' (level, cell) words read in place (row stride s*); outs[j] is output row j's own
// allocation, written in place. work is level-interleaved per slice column (work[r*n+i]): 17*nl+3 rows, then the
// ascent's substep pressure path (2*cap rows).
kernel void hexv842_deck_compute_diag_cols(constant Hexv842DeckComputeDiagColsResources &r [[buffer(0)]],constant Hexv842DeckComputeDiagColsParams &abi [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(abi.dispatch_count)) return;
const int c0=abi.c0;
const int n=abi.n;
const int nl=abi.nl;
const int cap=abi.cap;
const ulong mask=ulong(uint(abi.mask_lo))|(ulong(uint(abi.mask_hi))<<32);
const float rvord=abi.rvord;
device const float* height=r.height;
const int sz=abi.sz;
device const float* u=r.u;
const int su=abi.su;
device const float* v=r.v;
const int sv=abi.sv;
device const float* theta=r.theta;
const int st=abi.st;
device const float* qv=r.qv;
const int sq=abi.sq;
device const float* exner=r.exner;
const int se=abi.se;
device const float* pb=r.pb;
const int sb=abi.sb;
device const float* pp=r.pp;
const int sp=abi.sp;
const int no=abi.no;
device float* outs=r.outs;
device float* work=r.work;
    const int i=int(gid);
    if(i>=n) return;
    const int c=c0+i;
    const S z{height+c,sz},uc{u+c,su},vc{v+c,sv};
    #define ROW(r) W{work+(long)(r)*n+i,n}
    const W temp=ROW(0),dew=ROW(nl),zp=ROW(2*nl),zr=ROW(3*nl),dudz=ROW(4*nl+1),dvdz=ROW(5*nl+1);
    const W srh=ROW(6*nl+1),pin=ROW(7*nl+2),tin=ROW(8*nl+2),tdin=ROW(9*nl+2);
    const W cp_=ROW(10*nl+2),ct=ROW(11*nl+2),ctd=ROW(12*nl+2),cpi=ROW(13*nl+2),cq=ROW(14*nl+2),cth=ROW(15*nl+2);
    const W cthv=ROW(16*nl+2),prep=ROW(17*nl+3);
    #undef ROW
    for(int k=0;k<nl;k++) {
        const float thk=theta[(long)k*st+c],qk=qv[(long)k*sq+c],exk=exner[(long)k*se+c];
        const float pbk=pb[(long)k*sb+c],ppk=pp[(long)k*sp+c];
        temp[k]=(R(thk)/(R(1.0f)+R(rvord)*R(qk))*R(exk)).v;
        R evp=R(0.01f)*(R(pbk)+R(ppk))*R(qk)/(R(qk)+R(0.622f));
        evp=rmax(evp,1.0e-8f);
        R lg=rlog(evp/R(6.112f));
        dew[k]=(R(243.5f)*lg/(R(17.67f)-lg)+R(273.15f)).v;
        zp[k]=(R(0.5f)*(R(z[k])+R(z[k+1]))-R(z[0])).v;
    }
    for(int k=0;k<=nl;k++) zr[k]=(R(z[k])-R(z[0])).v;
    outs[6*no+c]=uc[0];outs[9*no+c]=vc[0];outs[12*no+c]=temp[0];outs[13*no+c]=dew[0];
    if(mask&(1u<<7)) outs[7*no+c]=column_height_value(uc,zp,1000.0f,nl).v;
    if(mask&(1u<<8)) outs[8*no+c]=column_height_value(uc,zp,6000.0f,nl).v;
    if(mask&(1u<<10)) outs[10*no+c]=column_height_value(vc,zp,1000.0f,nl).v;
    if(mask&(1u<<11)) outs[11*no+c]=column_height_value(vc,zp,6000.0f,nl).v;
    if(mask&((1u<<4)|(1u<<5))) {
        R ub=uc[0],vb=vc[0];
        if(R(0.0f)>R(zp[0])) {ub=column_height_value(uc,zp,0.0f,nl);vb=column_height_value(vc,zp,0.0f,nl);}
        R ut=column_height_value(uc,zp,6000.0f,nl),vt=column_height_value(vc,zp,6000.0f,nl);
        R us=ut-ub,vs=vt-vb;
        R um=integral_zstaggered(uc,zr,0.0f,6000.0f,nl)/R(6000.0f);
        R vm=integral_zstaggered(vc,zr,0.0f,6000.0f,nl)/R(6000.0f);
        R sm=rmax(0.0001f,rsqrt(us*us+vs*vs));
        R ustorm=um+R(7.5f)*vs/sm,vstorm=vm-R(7.5f)*us/sm;
        for(int k=1;k<nl-1;k++) {
            R dz=R(0.5f)*(R(z[k+1])-R(z[k-1]));
            dudz[k]=((R(uc[k])-R(uc[k-1]))/dz).v;
            dvdz[k]=((R(vc[k])-R(vc[k-1]))/dz).v;
        }
        dudz[0]=dudz[1];dvdz[0]=dvdz[1];dudz[nl-1]=dudz[nl-2];dvdz[nl-1]=dvdz[nl-2];
        for(int k=1;k<nl;k++) srh[k]=(-(R(0.5f)*(R(uc[k])+R(uc[k-1]))-ustorm)*R(dvdz[k])
                                  +(R(0.5f)*(R(vc[k])+R(vc[k-1]))-vstorm)*R(dudz[k])).v;
        srh[0]=(-(R(uc[0])-ustorm)*R(dvdz[0])+(R(vc[0])-vstorm)*R(dudz[0])).v;
        srh[nl]=srh[nl-1];
        for(int k=0;k<=nl;k++) srh[k]=rmax(0.0f,srh[k]).v;
        if(mask&(1u<<4)) outs[4*no+c]=integral_zpoint(srh,zr,0.0f,1000.0f,nl+1).v;
        if(mask&(1u<<5)) outs[5*no+c]=integral_zpoint(srh,zr,0.0f,3000.0f,nl+1).v;
    }
    if(mask&3u) {
        for(int k=0;k<nl;k++) {
            const float pbk=pb[(long)k*sb+c],ppk=pp[(long)k*sp+c];
            pin[k]=((R(ppk)+R(pbk))/R(100.0f)).v;
            tin[k]=(R(temp[k])-R(273.15f)).v;tdin[k]=(R(dew[k])-R(273.15f)).v;
        }
        float cape,cin;
        getcape_cols(nl,pin,tin,tdin,cp_,ct,ctd,cpi,cq,cth,cthv,prep,cap,cape,cin);
        outs[0*no+c]=cape;outs[1*no+c]=cin;
    }

}

// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681
// Native vertex order is retained, including the nCells+1 sentinel.

// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681
// The same scatter as a gather: one thread per (cell,level) adds its slots in
// native scatter order from the same zero word (convective_v842.helicity_plan).

// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681
// vertex_helicity_gather reading vorticity in the pool's own (level, vertex) layout (row stride sv).
kernel void hexv842_deck_vertex_helicity_gather_pool(constant Hexv842DeckVertexHelicityGatherPoolResources &r [[buffer(0)]],constant Hexv842DeckVertexHelicityGatherPoolParams &abi [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(abi.dispatch_count)) return;
const int nc=abi.nc;
const int nl=abi.nl;
const int degree=abi.degree;
device const int* offsets=r.offsets;
device const int* entries=r.entries;
device const float* kite=r.kite;
device const float* vort=r.vort;
const int sv=abi.sv;
device float* uh=r.uh;
    long t=(long)int(gid);
    if(t>=(long)(nc+1)*nl) return;
    int c=(int)(t/nl),k=(int)(t%nl);
    R s=0.0f;
    for(int e=offsets[c];e<offsets[c+1];e++) {
        int ix=entries[e];
        s=s+R(kite[ix])*R(vort[(long)k*sv+ix/degree]);
    }
    uh[(long)c*nl+k]=s.v;

}

// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681
// w in the pool's own (level, cell) layout (row stride sw).
kernel void hexv842_deck_split_helicity_pool(constant Hexv842DeckSplitHelicityPoolResources &r [[buffer(0)]],constant Hexv842DeckSplitHelicityPoolParams &abi [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(abi.dispatch_count)) return;
const int nc=abi.nc;
const int nl=abi.nl;
device const float* w=r.w;
const int sw=abi.sw;
device const float* area=r.area;
device float* positive=r.positive;
device float* negative=r.negative;
    int c=int(gid);
    if(c>nc) return;
    for(int k=0;k<nl;k++) {
        R h=positive[c*nl+k];
        if(c<nc) {
            R wm=R(0.5f)*(R(w[(long)k*sw+c])+R(w[(long)(k+1)*sw+c]));
            positive[c*nl+k]=(rmax(wm,0.0f)*rmax(h/R(area[c]),0.0f)).v;
            // The host constant-zero min retains the sign of a zero input.
            R hn=wm*h/R(area[c]);
            negative[c*nl+k]=hn.v<=0.0f?hn.v:0.0f;
        } else negative[c*nl+k]=h.v;
    }

}

// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681
kernel void hexv842_deck_smooth_helicity(constant Hexv842DeckSmoothHelicityResources &r [[buffer(0)]],constant Hexv842DeckSmoothHelicityParams &abi [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(abi.dispatch_count)) return;
const int ns=abi.ns;
const int nl=abi.nl;
const int maxedges=abi.maxedges;
device const int* bdy=r.bdy;
device const int* nedges=r.nedges;
device const int* edges=r.edges;
device const int* cells=r.cells;
device const float* dv=r.dv;
device const float* dc=r.dc;
device const float* positive=r.positive;
device const float* negative=r.negative;
device float* spos=r.spos;
device float* sneg=r.sneg;
    int c=int(gid);
    if(c>=ns) return;
    for(int k=0;k<nl;k++) {
        R sp=0.0f,sn=0.0f;
        if(bdy[c]==0) for(int j=0;j<nedges[c];j++) {
            int ix=c*maxedges+j,e=edges[ix]-1,other=cells[ix]-1;
            R weight=R(dv[e])/R(dc[e]);
            sp+=weight*(R(positive[other*nl+k])-R(positive[c*nl+k]));
            sn+=weight*(R(negative[other*nl+k])-R(negative[c*nl+k]));
        }
        spos[c*nl+k]=(R(positive[c*nl+k])+R(0.1f)*sp).v;
        sneg[c*nl+k]=(R(negative[c*nl+k])+R(0.1f)*sn).v;
    }

}

// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1146-1165
template<class X,class Z> R col_depth_max(X values,Z z,R bot,R top,int n) {
    int kb=0,kt=0;
    for(int k=0;k<n;k++) {if(R(z[k])<=bot) kb=k;if(R(z[k])>=top) kt=k;}
    kb=kb<n-1?kb:n-2;kt=kt<n-1?kt:n-2;
    R result=-as_type<float>(0x7f800000);
    for(int k=kb;k<=kt;k++) result=rmax(values[k],result);
    return result;
}
// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:1213-1233
template<class X,class P> R mean_pressure(X values,P p,R bot,R top,int n) {
    R result=0.0f;int count=0;
    for(int k=0;k<n;k++) if(R(p[k])<=bot&&R(p[k])>=top) {result+=values[k];count++;}
    // The strict GNU 0/0 result is x86's negative indefinite NaN, not
    // CUDA's 0x7fffffff canonical NaN (measured empty-window replay).
    if(count==0) return as_type<float>(0xffc00000);
    return result/R(float(count));
}
// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:297-681
// One thread per cell. Column inputs are the pools' (level, cell) words read in place (row stride s*); outs[j] is
// output row j's own allocation, updated in place. The work area is level-interleaved (work[k*ns+c]): zg is rows
// 0..nl, pres nl+1..2nl, speed 2nl+1..3nl. zg[nl] is the native unassigned wind-integration word: tail[c] when the
// caller restores a measured native word, else zero.
kernel void hexv842_deck_update_diag_pool(constant Hexv842DeckUpdateDiagPoolResources &r [[buffer(0)]],constant Hexv842DeckUpdateDiagPoolParams &abi [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(abi.dispatch_count)) return;
const int ns=abi.ns;
const int nl=abi.nl;
const ulong mask=ulong(uint(abi.mask_lo))|(ulong(uint(abi.mask_hi))<<32);
const float steps=abi.steps;
const int convection=abi.convection;
device const float* z=r.z;
const int sz=abi.sz;
device const float* w=r.w;
const int sw=abi.sw;
device const float* u=r.u;
const int su=abi.su;
device const float* v=r.v;
const int sv=abi.sv;
device const float* pb=r.pb;
const int sb=abi.sb;
device const float* pp=r.pp;
const int spr=abi.spr;
device const float* refl=r.refl;
const int sr=abi.sr;
device const float* u10=r.u10;
device const float* v10=r.v10;
device const float* t2=r.t2;
device const float* rainc=r.rainc;
device const float* rainnc=r.rainnc;
device const float* snow=r.snow;
device const float* uh=r.uh;
device const float* neg=r.neg;
device const float* tail=r.tail;
const int no=abi.no;
device float* outs=r.outs;
device float* work=r.work;
    int c=int(gid);if(c>=ns) return;
    const S zc{z+c,sz},wc{w+c,sw},uc{u+c,su},vc{v+c,sv},pbc{pb+c,sb},ppc{pp+c,spr},rc{refl+c,sr};
    const W zg{work+c,ns},pres{work+(long)(nl+1)*ns+c,ns},speed{work+(long)(2*nl+1)*ns+c,ns};
    zg[nl]=tail?tail[c]:0.0f;
    #define NEED(j) (mask&(1ull<<(j)))
    #define OUT(j) outs[(j)*no+c]
    if(NEED(28)) {R wm=wc[0];for(int k=1;k<nl;k++) wm=rmax(wc[k],wm);OUT(28)=rmax(wm,OUT(28)).v;}
    if(NEED(29)) {R wm=wc[0];for(int k=1;k<nl;k++) wm=rmin(wc[k],wm);OUT(29)=rmin(wm,OUT(29)).v;}
    if(NEED(30)) {
        R span=R(pbc[0])+R(ppc[0])-R(pbc[nl-1])-R(ppc[nl-1]);
        for(int k=0;k<nl;k++) pres[k]=(R(pbc[k])+R(ppc[k])).v;
        R mean=mean_pressure(wc,pres,R(0.8f)*span,R(0.5f)*span,nl);
        // Preserve GNU's quiet NaN word through multiplication/addition;
        // CUDA otherwise canonicalizes it on each arithmetic instruction.
        if(OUT(30)!=OUT(30)) {} // GNU propagates the existing accumulator.
        else if(mean.v!=mean.v) OUT(30)=mean.v;
        else OUT(30)=(R(OUT(30))+(R(1.0f)/R(steps))*mean).v;
    }
    if(NEED(31)) {R a=uc[0],b=vc[0];OUT(31)=rmax(rsqrt(a*a+b*b),OUT(31)).v;}
    if(NEED(32)||NEED(33)) {
        // The fork initializes only the first nl entries (zg[nl] is the word set above).
        for(int k=0;k<nl;k++) zg[k]=(R(0.5f)*(R(zc[k])+R(zc[k+1]))-R(zc[0])).v;
        if(column_height_value(rc,zg,1000.0f,nl)>R(20.0f)) {
            if(NEED(32)) {
                for(int k=0;k<nl;k++) {R a=uc[k],b=vc[k];speed[k]=rsqrt(a*a+b*b).v;}
                OUT(32)=rmax(col_depth_max(speed,zg,0.0f,2000.0f,nl),OUT(32)).v;
            }
            if(NEED(33)) {
                R a=integral_zstaggered(uc,zg,0.0f,2000.0f,nl);
                OUT(33)=rmax(rsqrt(a*a+a*a),OUT(33)).v;
            }
        }
    }
    if(NEED(34)||NEED(35)) {R a=u10[c],b=v10[c],speed10=rsqrt(a*a+b*b);
        if(NEED(34)) OUT(34)=rmax(speed10,OUT(34)).v;
        if(NEED(35)) OUT(35)=(R(OUT(35))+speed10*(R(1.0f)/R(steps))).v;
    }
    if(NEED(36)) OUT(36)=(R(OUT(36))+R(u10[c])*(R(1.0f)/R(steps))).v;
    if(NEED(37)) OUT(37)=(R(OUT(37))+R(v10[c])*(R(1.0f)/R(steps))).v;
    if(NEED(38)&&convection) OUT(38)=rmax(R(OUT(38))+R(rainc[c]),0.0f).v;
    if(NEED(39)) OUT(39)=rmax(R(OUT(39))+R(rainnc[c]),0.0f).v;
    if(NEED(40)) {
        OUT(40)=(R(OUT(40))+R(snow[c])).v;
        if(R(t2[c])<R(273.15f)) {
            if(convection) OUT(40)=(R(OUT(40))+R(rainc[c])).v;
            OUT(40)=rmax(OUT(40),0.0f).v;
        }
    }
    if(mask&(((1ull<<14)-1)<<14)) {
        for(int k=0;k<=nl;k++) zg[k]=(R(zc[k])-R(zc[0])).v;
        const float bot[7]={2000,0,1000,0,500,0,0},top[7]={5000,1000,6000,1500,1500,2000,3000};
        for(int j=0;j<14;j++) if(NEED(14+j)) {
            R h=integral_zstaggered((j<7?uh:neg)+(long)c*nl,zg,bot[j%7],top[j%7],nl);
            OUT(14+j)=(j<7?rmax(h,OUT(14+j)):rmin(h,OUT(14+j))).v;
        }
    }
    #undef NEED
    #undef OUT

}

// RRFS-MPAS src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F:984-1124
kernel void hexv842_deck_reset_diag_pool(constant Hexv842DeckResetDiagPoolResources &r [[buffer(0)]],constant Hexv842DeckResetDiagPoolParams &abi [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
if(gid>=uint(abi.dispatch_count)) return;
const int nc=abi.nc;
const ulong mask=ulong(uint(abi.mask_lo))|(ulong(uint(abi.mask_hi))<<32);
const int no=abi.no;
device float* outs=r.outs;
    int c=int(gid);if(c>=nc) return;
    for(int j=14;j<41;j++) if(mask&(1ull<<j)) outs[j*no+c]=0.0f;

}

// RRFS-MPAS src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:1350-1381

// RRFS-MPAS src/core_atmosphere/mpas_atm_core.F:972-1003


kernel void hexv842_deck_scale(constant Hexv842DeckScaleResources &r [[buffer(0)]], constant Hexv842DeckScaleParams &abi [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if(gid>=uint(abi.n)) return;
    r.dst[gid]=r.src[gid]*abi.factor;
}
