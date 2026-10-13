#include <metal_stdlib>
#include "../generated/ysu_column_bindings.metalh"
#include "../generated/ysu_validate_outputs_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_binary64.metalh"
using namespace metal;

// Ordinary WRF v4.6.1 YSU column, with the topo_wind arm (p.topo = 1:
// ysu_topo.cuh / bl_ysu.F90:1254-1314 and :1402-1408). The BEP variant is
// absent from this entry point and is rejected by the Python dispatcher.
// Source operation order follows woof/core/kernels/ysu.cu / bl_ysu.F90.
// The ordinary CUDA YSU kernel permits multiply-add contraction (the
// recorded NVRTC options do not disable fmad). Preserve that policy here.
#pragma clang fp contract(fast)

constant float YSU_G=9.81f, YSU_RD=287.0f, YSU_RV=461.6f;
constant float YSU_CP=1004.5f, YSU_XLV=2.5e6f;
constant float YSU_RCP=0.2857142984867096f;
// Match CUDA EP2 (RD/RV), including cloudy top-down mixing.
constant float YSU_EP2=YSU_RD/YSU_RV;
constant uint YSU_SLOTS=18u, YSU_LANES=32u;

struct YsuCol { device float *p; };
static inline float yget(YsuCol a, int k) { return a.p[k * int(YSU_LANES)]; }
static inline void yset(YsuCol a, int k, float x) { a.p[k * int(YSU_LANES)] = x; }
static inline float yrndiv(float a,float b) {
    return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)));
}
static inline float ymin(float a, float b) { return isnan(a) ? a : (a < b ? a : b); }
static inline float ymax(float a, float b) { return isnan(a) ? a : (a > b ? a : b); }
static inline float yclip(float x, float lo, float hi) { return ymin(ymax(x, lo), hi); }
struct YsuU128 { ulong hi; ulong lo; };

// Exact 32x32x32 unsigned product, returned as a 128-bit two-limb value.
// Call sites use at most 25-bit operands, so the intermediate 64-bit square
// and the split 64x32 product fit without losing any carry bits.
static inline YsuU128 ysu_cube_u32(uint x) {
    ulong square=ulong(x)*ulong(x);
    ulong low_product=(square & 0xfffffffful)*ulong(x);
    ulong high_product=(square >> 32)*ulong(x);
    ulong shifted=high_product << 32;
    ulong lo=low_product+shifted;
    ulong carry=lo<low_product ? 1ul : 0ul;
    return YsuU128{(high_product >> 32)+carry,lo};
}
static inline bool ysu_u128_le(YsuU128 a,YsuU128 b) {
    return a.hi<b.hi || (a.hi==b.hi && a.lo<=b.lo);
}
static inline bool ysu_u128_eq(YsuU128 a,YsuU128 b) {
    return a.hi==b.hi && a.lo==b.lo;
}
static inline YsuU128 ysu_scaled_u24(uint significand,uint shift) {
    ulong value=ulong(significand);
    return YsuU128{value >> (64u-shift),value << shift};
}

// Correctly rounded RN-even real cube root for finite float32 values. The
// input is normalized to x = M * 2^(E-23), with a 24-bit M. For E=3e+r,
// r in [0,2], compare candidate significand cubes against M*2^(r+46).
// The midpoint between adjacent candidates has odd significand 2*m+1 and
// is compared exactly against M*2^(r+49); equality selects even output bits.
// This guarantees the real RN-even result; it does not claim CUDA cbrtf bit
// identity, since CUDA's libdevice implementation may use a different path.
static inline float ysu_cbrt_rne(float x) {
    const uint original=as_type<uint>(x), sign=original & 0x80000000u;
    const uint magnitude=original & 0x7fffffffu;
    const uint exponent_bits=(magnitude >> 23) & 0xffu;
    uint significand=magnitude & 0x007fffffu;
    if(magnitude==0u) return as_type<float>(original);
    if(exponent_bits==0xffu){
        if(significand==0u) return as_type<float>(original);
        return as_type<float>(original | 0x00400000u);
    }
    int exponent;
    if(exponent_bits!=0u){
        significand|=0x00800000u;
        exponent=int(exponent_bits)-127;
    } else {
        uint msb=0u, probe=significand;
        while(probe>1u){probe >>= 1u;++msb;}
        significand <<= (23u-msb);
        exponent=int(msb)-149;
    }
    int root_exponent=exponent/3, remainder=exponent%3;
    if(remainder<0){remainder+=3;--root_exponent;}
    YsuU128 target=ysu_scaled_u24(significand,uint(remainder+46));
    uint lo=0x00800000u, hi=0x01000000u;
    while(lo+1u<hi){
        uint mid=lo+((hi-lo)>>1u);
        if(ysu_u128_le(ysu_cube_u32(mid),target)) lo=mid;
        else hi=mid;
    }
    const uint odd_midpoint=(lo<<1u)+1u;
    const YsuU128 midpoint_cube=ysu_cube_u32(odd_midpoint);
    const YsuU128 midpoint_target=ysu_scaled_u24(significand,uint(remainder+49));
    uint rounded=lo;
    if(ysu_u128_le(midpoint_cube,midpoint_target)){
        if(!ysu_u128_eq(midpoint_cube,midpoint_target) || (lo&1u)!=0u)
            rounded=lo+1u;
    }
    if(rounded==0x01000000u){++root_exponent;rounded=0x00800000u;}
    const uint result=sign | (uint(root_exponent+127)<<23)
                    | (rounded & 0x007fffffu);
    return as_type<float>(result);
}
static inline float yhypot(float x, float y) {
    if (isinf(x) || isinf(y)) return INFINITY;
    if (isnan(x) || isnan(y)) return NAN;
    float a=fabs(x), b=fabs(y), hi=max(a,b), lo=min(a,b);
    if (hi == 0.0f) return 0.0f;
    float q=lo/hi; return hi*sqrt(1.0f+q*q);
}
// CUDA's exceptional prfac2 rescue is deliberately evaluated in binary64.
// Metal has no native FP64 arithmetic on this target, so use the shared
// integer-bit implementation for that rare branch rather than changing the
// float32 path or dropping the source's rescue.
static inline float yprfac2_rescue(float us,float wstar3,float wstar3_2,float karman){
    const ulong c159=0x402fcccccccccccdul,c4=0x4010000000000000ul;
    const ulong one=0x3ff0000000000000ul;
    ulong u=soft64_from_f32_bits(as_type<uint>(us));
    ulong u3=soft64_mul(soft64_mul(u,u),u);
    ulong w=soft64_add(soft64_from_f32_bits(as_type<uint>(wstar3)),
                       soft64_from_f32_bits(as_type<uint>(wstar3_2)));
    if(soft64_greater(u3,0ul)){
        ulong k=soft64_from_f32_bits(as_type<uint>(karman));
        ulong numer=soft64_div(soft64_mul(c159,w),u3);
        ulong denom=soft64_add(one,soft64_div(soft64_mul(soft64_mul(c4,k),w),u3));
        return as_type<float>(soft64_to_f32_bits(soft64_div(numer,denom)));
    }
    if(soft64_greater(w,0ul)){
        ulong k=soft64_from_f32_bits(as_type<uint>(karman));
        return as_type<float>(soft64_to_f32_bits(soft64_div(c159,soft64_mul(c4,k))));
    }
    return 0.0f;
}

static inline void ydiagnose(device const float *u, device const float *v,
        YsuCol thv, YsuCol za, float thermal, float br0, float brcrit,
        int nz, int st, int col, thread float &hpbl, thread int &kpbl,
        thread float &brdn_out, thread float &brup_out) {
    float brup=br0, brdn=brup; int kp=1; bool crossed=false;
    for (int k=1;k<nz;++k) if (!crossed) {
        brdn=brup; int q=k*st+col;
        float spdk2=ymax(u[q]*u[q]+v[q]*v[q],1.0f);
        brup=(yget(thv,k)-thermal)*(YSU_G*yget(za,k)/yget(thv,0))/spdk2;
        kp=k+1; crossed=brup>brcrit;
    }
    float frac;
    if (brdn>=brcrit) frac=0.0f;
    else if (brup<=brcrit) frac=1.0f;
    else frac=(brcrit-brdn)/(brup-brdn);
    int kh=kp-1;
    hpbl=yget(za,kh-1)+frac*(yget(za,kh)-yget(za,kh-1));
    kpbl=kp; brdn_out=brdn; brup_out=brup;
}

static inline void ythomas(YsuCol lower,YsuCol diag,YsuCol upper,
                            YsuCol rhs,YsuCol gamma,int nz) {
    float inv=1.0f/yget(diag,0); yset(gamma,0,yget(upper,0)*inv);
    yset(rhs,0,yget(rhs,0)*inv);
    for(int k=1;k<nz-1;++k){
        inv=1.0f/(yget(diag,k)-yget(lower,k)*yget(gamma,k-1));
        yset(gamma,k,yget(upper,k)*inv);
        yset(rhs,k,(yget(rhs,k)-yget(lower,k)*yget(rhs,k-1))*inv);
    }
    inv=1.0f/(yget(diag,nz-1)-yget(lower,nz-1)*yget(gamma,nz-2));
    yset(rhs,nz-1,(yget(rhs,nz-1)-yget(lower,nz-1)*yget(rhs,nz-2))*inv);
    for(int k=nz-2;k>=0;--k) yset(rhs,k,yget(rhs,k)-yget(gamma,k)*yget(rhs,k+1));
}

// bl_ysu.F90 get_pblh (ysu_topo.cuh ysu_get_pblh), on the column workspace.
static inline float ysu_get_pblh(YsuCol thv, YsuCol tke, YsuCol zq, int nz, float landsea) {
    const float sbl_lim=200.0f, sbl_damp=400.0f;
    const int kte=nz, kts=1;
    int k=kts+1, kthv=1, ktke=1;
    float maxqke=0.0f, minthv=9.e9f;
    while(k<=kte && yget(zq,k-1)<=500.0f){
        float qtke=fmax(yget(tke,k-1),0.0f);
        if(maxqke<qtke){maxqke=qtke;ktke=k;}
        if(minthv>yget(thv,k-1)){minthv=yget(thv,k-1);kthv=k;}
        k=k+1;
    }
    float tkeeps=maxqke/40.0f; tkeeps=fmax(tkeeps,0.025f); tkeeps=fmin(tkeeps,0.25f);
    float delt_thv=(landsea-1.5f>=0.0f)?0.75f:1.5f;
    float zi=0.0f; k=kthv+1; float thr=minthv+delt_thv;
    while(zi==0.0f && k<=kte){
        if(yget(thv,k-1)>=thr){
            float frac=fmin((yget(thv,k-1)-thr)/fmax(yget(thv,k-1)-yget(thv,k-2),1e-6f),1.0f);
            zi=yget(zq,k-1)-(yget(zq,k-1)-yget(zq,k-2))*frac;
        }
        k=k+1; if(k==kte-1) zi=yget(zq,kts);
    }
    float pblh_tke=0.0f; k=ktke+1;
    while(pblh_tke==0.0f && k<=kte){
        float qtke=fmax(yget(tke,k-1)/2.0f,0.0f), qtkem1=fmax(yget(tke,k-2)/2.0f,0.0f);
        if(qtke<=tkeeps){
            float frac=fmin((tkeeps-qtke)/fmax(qtkem1-qtke,1e-6f),1.0f);
            pblh_tke=yget(zq,k-1)-(yget(zq,k-1)-yget(zq,k-2))*frac;
            pblh_tke=fmax(pblh_tke,yget(zq,kts));
        }
        k=k+1; if(k==kte-1) pblh_tke=yget(zq,kts);
    }
    float wt=0.5f*precise::tanh((zi-sbl_lim)/sbl_damp)+0.5f;
    return pblh_tke*(1.0f-wt)+zi*wt;
}
static inline void ysu_topo_blend_u10(constant YsuColumnResources &r, float u1, float v1, int col) {
    float c2=r.ctopo2[col];
    r.u10o[col]=c2*r.u10[col]+(1.0f-c2)*u1;
    r.v10o[col]=c2*r.v10[col]+(1.0f-c2)*v1;
}

kernel void ysu_column(
    constant YsuColumnResources &r [[buffer(0)]],
    constant YsuColumnParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nz=p.nz, ny=p.ny, nx=p.nx, st=ny*nx;
    const uint local=tid;
    const uint required_ws=((uint(p.tilecol)+YSU_LANES-1u)/YSU_LANES)
                         *YSU_SLOTS*uint(p.wskp)*YSU_LANES;
    if (local>=uint(p.tilecol) || p.ncol!=st || nz<4 || nz>128 ||
        p.wskp<nz+1 || p.ws_count<=0 || uint(p.ws_count)<required_ws) return;
    const int col=p.col0+int(local);
    if(col>=st) return;
    const uint block=local/YSU_LANES, lane=local%YSU_LANES;
    device float *base=r.ws + ulong(block)*YSU_SLOTS*uint(p.wskp)*YSU_LANES;
    YsuCol thv{base+(0u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol thli{base+(1u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol zq{base+(2u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol za{base+(3u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol dza{base+(4u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol delp{base+(5u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol xkzm{base+(6u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol xkzh{base+(7u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol xkzq{base+(8u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol xkzml{base+(9u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol xkzhl{base+(10u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol zfacent{base+(11u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol entfac{base+(12u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol lower{base+(13u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol diag{base+(14u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol upper{base+(15u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol rhs{base+(16u*uint(p.wskp))*YSU_LANES+lane};
    YsuCol gamma{base+(17u*uint(p.wskp))*YSU_LANES+lane};
    const int ts=p.ysu_topdown_pblmix;

    float us=r.ust[col], hf=r.hfx[col], qf=r.qfx[col];
    if(us==0.0f&&hf==0.0f&&qf==0.0f){
        for(int k=0;k<nz;++k){int q=k*st+col;r.du[q]=r.dv[q]=r.dtheta[q]=r.dqv[q]=r.dqc[q]=r.dqi[q]=0.0f;r.exch_h[q]=r.exch_m[q]=0.0f;}
        r.hpbl[col]=r.dz[col];r.kpbl[col]=1;r.wstar[col]=r.delta[col]=0.0f;
        r.topdown_radsum[col]=r.wstar3_2[col]=0.0f;r.cloudflg[col]=0;if(p.topo)ysu_topo_blend_u10(r,r.u[col],r.v[col],col);return;
    }
    const float xkzminm=.1f,xkzminh=.01f,xkzmax=1000.0f,rimin=-100.0f;
    const float rlam=30.0f,prmin=.25f,prmax=4.0f,brcr_ub=0.0f,brcr_sb=.25f,cori=1.0e-4f;
    const float afac=6.8f,bfac=6.8f,phifac=8.0f,sfcfrac=.1f;
    const float d1=.02f,d2=.05f,d3=.001f,h2=.666666666667f,zfmin=1.0e-8f;
    const float aphi5=5.0f,aphi16=16.0f,tmin=.01f,gamcrt=3.0f,gamcrq=.002f;
    const float ep1=YSU_RV/YSU_RD-1.0f,karman=.4f;
    float dt2=2.0f*p.dt,rdt=1.0f/dt2;
    yset(zq,0,0.0f);
    for(int k=0;k<nz;++k){int q=k*st+col;
        yset(thv,k,r.theta[q]*(1.0f+ep1*r.qv[q]));
        yset(thli,k,(r.theta[q]*r.exner[q]-yrndiv(YSU_XLV*r.qc[q],YSU_CP)-yrndiv(2.834e6f*r.qi[q],YSU_CP))/r.exner[q]);
        yset(zq,k+1,yget(zq,k)+r.dz[q]);
        yset(delp,k,r.p_interface[k*st+col]-r.p_interface[(k+1)*st+col]);
    }
    for(int k=0;k<nz;++k)yset(za,k,.5f*(yget(zq,k)+yget(zq,k+1)));
    yset(dza,0,yget(za,0));for(int k=1;k<nz;++k)yset(dza,k,yget(za,k)-yget(za,k-1));
    float theta0=r.theta[col],qv0=r.qv[col],temp0=theta0*r.exner[col];
    float rho=r.psfc[col]/(YSU_RD*temp0*(1.0f+ep1*qv0)),govrth=YSU_G/theta0;
    float u0=r.u[col],v0=r.v[col],wspd1=sqrt(u0*u0+v0*v0)+1.0e-9f;
    float sflux=yrndiv(hf/rho,YSU_CP)+qf/rho*ep1*theta0;
    float thermal=yget(thv,0),thermalli=yget(thli,0),hpbl;int kpbl;float brdn,brup;
    ydiagnose(r.u,r.v,thv,za,thermal,r.br[col],brcr_ub,nz,st,col,hpbl,kpbl,brdn,brup);
    if(hpbl<yget(zq,1))kpbl=1;bool pblflg=kpbl>1,sfcflg=r.br[col]<=0.0f;
    float zol1=ymax(r.br[col]*r.psim[col]*r.psim[col]/r.psih[col],rimin);
    zol1=sfcflg?ymin(zol1,-zfmin):ymax(zol1,zfmin);
    float hol1=zol1*hpbl/yget(za,0)*sfcfrac,phim,phih,wstar3,wstar;
    if(sfcflg){phim=pow(1.0f-16.0f*hol1,-.25f);phih=pow(1.0f-16.0f*hol1,-.5f);wstar3=govrth*ymax(sflux,0.0f)*hpbl;wstar=ysu_cbrt_rne(ymax(wstar3,0.0f));}
    else{phim=phih=1.0f+aphi5*hol1;wstar3=wstar=0.0f;}
    float ust3=us*us*us,wscale=ysu_cbrt_rne(ymax(ust3+phifac*karman*wstar3*.5f,0.0f));
    wscale=ymin(wscale,us*aphi16);wscale=ymax(wscale,yrndiv(us,aphi5));
    float hgamt=0,hgamq=0,hgamu=0,hgamv=0;
    if(sfcflg&&sflux>0.0f){float gamfac=bfac/rho/wscale;hgamt=ymin(yrndiv(gamfac*hf,YSU_CP),gamcrt);hgamq=ymin(gamfac*qf,gamcrq);
        float vpert=yrndiv((hgamt+ep1*theta0*hgamq),bfac)*afac;
        thermal+=ymax(vpert,0.0f)*ymin(yget(za,0)/(sfcfrac*hpbl),1.0f);
        thermalli+=ymax(vpert,0.0f)*ymin(yget(za,0)/(sfcfrac*hpbl),1.0f);
        hgamt=ymax(hgamt,0.0f);hgamq=ymax(hgamq,0.0f);
        float cg=-15.9f*us*us/ymax(r.wspd[col],1.0e-9f)*wstar3/ymax(wscale*wscale*wscale*wscale,1.0e-20f);
        hgamu=cg*u0;hgamv=cg*v0;
        if(pblflg)ydiagnose(r.u,r.v,thv,za,thermal,r.br[col],brcr_ub,nz,st,col,hpbl,kpbl,brdn,brup);
    } else pblflg=false;

    if(ts){bool definebrup=false;int kpblold=kpbl;
        for(int fk=kpblold;fk<=nz-1;++fk){int k=fk-1,q=k*st+col;
            float spdk2=ymax(r.u[q]*r.u[q]+r.v[q]*r.v[q],1.0f);
            float bt=(yget(thli,k)-thermalli)*(YSU_G*yget(za,k)/yget(thli,0))/spdk2;
            bool stable_li=bt>=brcr_ub;if(definebrup){kpbl=fk;brup=bt;definebrup=false;}
            if(!stable_li){brdn=bt;definebrup=true;pblflg=true;}}
    }
    if(pblflg){int kh=kpbl-1;float frac;
        if(brdn>=brcr_ub)frac=0.0f;else if(brup<=brcr_ub)frac=1.0f;else frac=(brcr_ub-brdn)/(brup-brdn);
        hpbl=yget(za,kh-1)+frac*(yget(za,kh)-yget(za,kh-1));if(hpbl<yget(zq,1))kpbl=1;if(kpbl<=1)pblflg=false;}
    if(!sfcflg&&hpbl<yget(zq,1)){float brcrit=brcr_sb;
        if(r.xland[col]>=1.5f){float ross=ymax(yhypot(r.u10[col],r.v10[col]),1.0e-9f)/(cori*r.znt[col]);brcrit=ymin(.16f*pow(1.0e-7f*ross,-.18f),.3f);}
        ydiagnose(r.u,r.v,thv,za,thermal,r.br[col],brcrit,nz,st,col,hpbl,kpbl,brdn,brup);if(hpbl<yget(zq,1))kpbl=1;
    }
    float wm2=0,we=0,hfxpbl=0,qfxpbl=0,ufxpbl=0,vfxpbl=0,delta=0,wstar3_2=0,topdown_radsum=0;bool cloudflg=false;
    if(pblflg){int kt=kpbl-2;float wm3=wstar3+5.0f*ust3;wm2=pow(ymax(wm3,0.0f),h2);
        float bfxpbl=yrndiv(-.15f*yget(thv,0),YSU_G)*wm3/hpbl,dthv=ymax(yget(thv,kt+1)-yget(thv,kt),tmin);
        we=ymax(bfxpbl/dthv,-sqrt(wm2));
        if(ts&&kpbl<nz&&r.qc[kt*st+col]+r.qi[kt*st+col]>1.0e-5f){cloudflg=true;
            float ptop=r.p_interface[(kt+1)*st+col],templ=yget(thli,kt)*pow(yrndiv(ptop,100000.0f),YSU_RCP);
            float rvls=100.0f*6.112f*exp(17.67f*(templ-273.16f)/(templ-29.65f))*(YSU_EP2/ptop);
            float qsum=r.qv[kt*st+col]+r.qc[kt*st+col];
            float temps=templ+(qsum-rvls)/(YSU_CP/YSU_XLV+YSU_EP2*YSU_XLV*rvls/(YSU_RD*templ*templ));
            rvls=100.0f*6.112f*exp(17.67f*(temps-273.15f)/(temps-29.65f))*(YSU_EP2/ptop);
            float rcldb=ymax(qsum-rvls,0.0f);int kabove=kt+2;
            float dthvc=(yget(thli,kabove)+r.theta[kabove*st+col]*ep1*(r.qv[kabove*st+col]+r.qc[kabove*st+col]))
                       -(yget(thli,kt)+r.theta[kt*st+col]*ep1*qsum);dthvc=ymax(dthvc,.1f);
            float tmp1=YSU_XLV/YSU_CP*rcldb/(r.exner[kt*st+col]*dthvc),ent_eff=.2f*8.0f*tmp1+.2f;
            for(int kk=0;kk<=kt;++kk){float flux=r.rthraten[kk*st+col]*r.exner[kk*st+col];flux*=YSU_CP/YSU_G*(r.p_interface[kk*st+col]-r.p_interface[(kk+1)*st+col]);if(flux<0.0f)topdown_radsum+=fabs(flux);}
            topdown_radsum=ymax(topdown_radsum,0.0f);float tk=r.theta[kt*st+col]*r.exner[kt*st+col];
            float rho2=r.p[kt*st+col]/(YSU_RD*tk*(1.0f+ep1*r.qv[kt*st+col]));
            float bfx0=ymax(sflux,0.0f);wm3=govrth*bfx0*hpbl+5.0f*ust3;wm2=pow(wm3,h2);
            bfxpbl=yrndiv(-.15f*yget(thv,0),YSU_G)*wm3/hpbl;dthv=ymax(yget(thv,kt+1)-yget(thv,kt),tmin);we=ymax(bfxpbl/dthv,-sqrt(wm2));
            bfx0=ymax(yrndiv(topdown_radsum/rho2,YSU_CP),0.0f);float wm3top=YSU_G/yget(thv,kt)*bfx0*hpbl,wm2top=pow(wm3top,h2);
            wm2+=wm2top;bfxpbl=-ent_eff*bfx0;dthv=ymax(yget(thv,kt+1)-yget(thv,kt),.1f);we+=ymax(bfxpbl/dthv,-sqrt(wm2top));
            wstar3_2=YSU_G/yget(thv,kt)*bfx0*hpbl;wscale=ysu_cbrt_rne(ust3+phifac*karman*(wstar3+wstar3_2)*.5f);wscale=ymin(wscale,us*aphi16);wscale=ymax(wscale,yrndiv(us,aphi5));
            float gamfac=bfac/rho/wscale;hgamt=ymin(yrndiv(gamfac*hf,YSU_CP),gamcrt);hgamq=ymin(gamfac*qf,gamcrq);
            gamfac=bfac/rho2/wscale;float hgamt2=ymin(yrndiv(gamfac*topdown_radsum,YSU_CP),gamcrt);hgamt=ymax(hgamt,0.0f)+ymax(hgamt2,0.0f);
            float cg=-15.9f*us*us/ymax(r.wspd[col],1.0e-9f)*(wstar3+wstar3_2)/ymax(wscale*wscale*wscale*wscale,1.0e-20f);hgamu=cg*u0;hgamv=cg*v0;
        }
        float dth=ymax(r.theta[(kt+1)*st+col]-r.theta[kt*st+col],tmin),dqq=ymin(r.qv[(kt+1)*st+col]-r.qv[kt*st+col],0.0f);
        hfxpbl=we*dth;qfxpbl=we*dqq;float dux=r.u[(kt+1)*st+col]-r.u[kt*st+col],dvx=r.v[(kt+1)*st+col]-r.v[kt*st+col];
        if(dux>tmin)ufxpbl=ymax(we*dux,-us*us);else if(dux< -tmin)ufxpbl=ymin(we*dux,us*us);
        if(dvx>tmin)vfxpbl=ymax(we*dvx,-us*us);else if(dvx< -tmin)vfxpbl=ymin(we*dvx,us*us);
        float delb=govrth*d3*hpbl;delta=ymin(d1*hpbl+d2*wm2/delb,100.0f);
    }
    for(int k=0;k<nz;++k){float e=1.0e30f;if(pblflg&&delta>0.0f&&k+1>=kpbl){float x=(yget(zq,k+1)-hpbl)/delta;e=x*x;}yset(entfac,k,e);
        yset(xkzm,k,0.0f);yset(xkzh,k,0.0f);yset(xkzq,k,0.0f);if(k<nz-1){yset(xkzm,k,xkzminm);yset(xkzh,k,xkzminh);yset(xkzq,k,xkzminh);}yset(xkzml,k,0.0f);yset(xkzhl,k,0.0f);yset(zfacent,k,0.0f);}
    for(int k=0;k<nz;++k)if(k+1<kpbl){float zfac=yclip(1.0f-(yget(zq,k+1)-yget(za,0))/(hpbl-yget(za,0)),zfmin,1.0f);yset(zfacent,k,pow(1.0f-zfac,3.0f));
        float wsk=ysu_cbrt_rne(ymax(ust3+phifac*karman*wstar3*(1.0f-zfac),0.0f)),wsk2=ysu_cbrt_rne(ymax(phifac*karman*wstar3_2*zfac,0.0f));
        float prfac,prfac2,prnumfac;if(sfcflg){prfac=bfac*karman*sfcfrac;prfac2=15.9f*(wstar3+wstar3_2)/ust3/(1.0f+4.0f*karman*(wstar3+wstar3_2)/ust3);
            if(!isfinite(prfac2))prfac2=yprfac2_rescue(us,wstar3,wstar3_2,karman);
            float zz=ymax(yget(zq,k+1)-sfcfrac*hpbl,0.0f);prnumfac=-3.0f*zz*zz/(hpbl*hpbl);
        }else{prfac=prfac2=prnumfac=0.0f;wsk=ymax(us/(1.0f+aphi5*zol1*yget(zq,k+1)/yget(za,0)),.001f);}
        float prnum0=yclip(phih/phim+prfac,prmin,prmax);float km=wsk*karman*yget(zq,k+1)*zfac*zfac+wsk2*karman*(hpbl-yget(zq,k+1))*(1.0f-zfac)*(1.0f-zfac);
        if(k==kpbl-2&&cloudflg&&we<0.0f)km=0.0f;float prnum=1.0f+(prnum0-1.0f)*exp(prnumfac);float kqq=km/prnum;prnum0/=1.0f+prfac2*karman*sfcfrac;prnum=1.0f+(prnum0-1.0f)*exp(prnumfac);float kh=km/prnum;
        yset(xkzm,k,ymin(km+xkzminm,xkzmax));yset(xkzh,k,ymin(kh+xkzminh,xkzmax));yset(xkzq,k,ymin(kqq+xkzminh,xkzmax));}
    for(int k=0;k<nz-1;++k)if(k+1>=kpbl){int q0=k*st+col,q1=(k+1)*st+col;float ud=r.u[q1]-r.u[q0],vd=r.v[q1]-r.v[q0];float ss=(ud*ud+vd*vd)/(yget(dza,k+1)*yget(dza,k+1))+1.e-9f;
        float ri=(YSU_G/(.5f*(yget(thv,k+1)+yget(thv,k))))*(yget(thv,k+1)-yget(thv,k))/(ss*yget(dza,k+1));
        if(r.qc[q0]+r.qi[q0]>1.e-5f&&r.qc[q1]+r.qi[q1]>1.e-5f){float qm=.5f*(r.qv[q0]+r.qv[q1]),tm=.5f*(r.theta[q0]*r.exner[q0]+r.theta[q1]*r.exner[q1]);float alph=yrndiv(YSU_XLV*qm,YSU_RD)/tm,chi=yrndiv(yrndiv(YSU_XLV*YSU_XLV*qm,YSU_CP),YSU_RV)/(tm*tm);ri=(1.0f+alph)*(ri-(yrndiv(YSU_G*YSU_G/ss/tm,YSU_CP))*((chi-alph)/(1.0f+chi)));}
        float zk=karman*yget(zq,k+1),rlamdz=ymin(ymax(.1f*yget(dza,k+1),rlam),300.0f);rlamdz=ymin(yget(dza,k+1),rlamdz);float rl=zk*rlamdz/(rlamdz+zk),dk=rl*rl*sqrt(ss),km,kh;
        if(ri<0.0f){ri=ymax(ri,rimin);float sri=sqrt(-ri);km=dk*(1.0f+8.0f*(-ri)/(1.0f+1.746f*sri));kh=dk*(1.0f+8.0f*(-ri)/(1.0f+1.286f*sri));}
        else{kh=dk/((1.0f+5.0f*ri)*(1.0f+5.0f*ri));km=kh*ymin(1.0f+2.1f*ri,prmax);}
        yset(xkzm,k,ymin(km+xkzminm,xkzmax));yset(xkzh,k,ymin(kh+xkzminh,xkzmax));yset(xkzml,k,yget(xkzm,k));yset(xkzhl,k,yget(xkzh,k));}
    for(int k=kpbl-1;k<nz-1;++k)yset(xkzq,k,yget(xkzh,k));

    // Heat diffusion/countergradient system and theta tendency.
    for(int k=0;k<nz;++k){yset(lower,k,0.0f);yset(diag,k,0.0f);yset(upper,k,0.0f);yset(rhs,k,0.0f);}
    yset(diag,0,1.0f);yset(rhs,0,theta0-300.0f+yrndiv(hf,(YSU_CP/YSU_G))/yget(delp,0)*dt2);
    for(int k=0;k<nz-1;++k){float dtodsd=dt2/yget(delp,k),dtodsu=dt2/yget(delp,k+1),dsig=r.p[k*st+col]-r.p[(k+1)*st+col],rdz=1.0f/yget(dza,k+1),tem1=dsig*yget(xkzh,k)*rdz;
        if(pblflg&&k+1<kpbl){float flux=tem1*(-hgamt/hpbl-hfxpbl*yget(zfacent,k)/yget(xkzh,k));yset(rhs,k,yget(rhs,k)+dtodsd*flux);yset(rhs,k+1,r.theta[(k+1)*st+col]-300.0f-dtodsu*flux);}
        else if(pblflg&&k+1>=kpbl&&yget(entfac,k)<4.6f){float knew=-we*yget(dza,kpbl-1)*exp(-yget(entfac,k));yset(xkzh,k,yclip(sqrt(ymax(knew*yget(xkzhl,k),0.0f)),xkzminh,xkzmax));yset(rhs,k+1,r.theta[(k+1)*st+col]-300.0f);}
        else yset(rhs,k+1,r.theta[(k+1)*st+col]-300.0f);
        float dsdz2=dsig*yget(xkzh,k)*rdz*rdz;yset(upper,k,-dtodsd*dsdz2);yset(lower,k+1,-dtodsu*dsdz2);yset(diag,k,yget(diag,k)-yget(upper,k));yset(diag,k+1,1.0f-yget(lower,k+1));}
    ythomas(lower,diag,upper,rhs,gamma,nz);
    for(int k=0;k<nz;++k){int q=k*st+col;r.dtheta[q]=(yget(rhs,k)-r.theta[q]+300.0f)*rdt;}

    // Vapor matrix reused for cloud water and ice exactly as WRF's tridi2n.
    for(int k=0;k<nz;++k){yset(lower,k,0.0f);yset(diag,k,0.0f);yset(upper,k,0.0f);yset(rhs,k,0.0f);}yset(diag,0,1.0f);yset(rhs,0,qv0+qf*YSU_G/yget(delp,0)*dt2);
    for(int k=0;k<nz-1;++k){float ds=dt2/yget(delp,k),du=dt2/yget(delp,k+1),dsig=r.p[k*st+col]-r.p[(k+1)*st+col],rdz=1.0f/yget(dza,k+1),tem1=dsig*yget(xkzq,k)*rdz;
        if(pblflg&&k+1<kpbl){float flux=tem1*(-qfxpbl*yget(zfacent,k)/yget(xkzq,k));yset(rhs,k,yget(rhs,k)+ds*flux);yset(rhs,k+1,r.qv[(k+1)*st+col]-du*flux);}
        else if(pblflg&&k+1>=kpbl&&yget(entfac,k)<4.6f){float knew=-we*yget(dza,kpbl-1)*exp(-yget(entfac,k));yset(xkzq,k,yclip(sqrt(ymax(knew*yget(xkzhl,k),0.0f)),xkzminh,xkzmax));yset(rhs,k+1,r.qv[(k+1)*st+col]);}
        else yset(rhs,k+1,r.qv[(k+1)*st+col]);float dsdz2=dsig*yget(xkzq,k)*rdz*rdz;yset(upper,k,-ds*dsdz2);yset(lower,k+1,-du*dsdz2);yset(diag,k,yget(diag,k)-yget(upper,k));yset(diag,k+1,1.0f-yget(lower,k+1));}
    ythomas(lower,diag,upper,rhs,gamma,nz);for(int k=0;k<nz;++k)r.dqv[k*st+col]=(yget(rhs,k)-r.qv[k*st+col])*rdt;
    for(int k=0;k<nz;++k)yset(rhs,k,r.qc[k*st+col]);ythomas(lower,diag,upper,rhs,gamma,nz);for(int k=0;k<nz;++k)r.dqc[k*st+col]=(yget(rhs,k)-r.qc[k*st+col])*rdt;
    for(int k=0;k<nz;++k)yset(rhs,k,r.qi[k*st+col]);ythomas(lower,diag,upper,rhs,gamma,nz);for(int k=0;k<nz;++k)r.dqi[k*st+col]=(yget(rhs,k)-r.qi[k*st+col])*rdt;

    // Momentum diffusion: u and v each get a Thomas solve with the same
    // non-BEP matrix and separately reconstructed countergradient RHS.
    for(int k=0;k<nz;++k){yset(lower,k,0.0f);yset(diag,k,0.0f);yset(upper,k,0.0f);yset(rhs,k,0.0f);}
    float fric=us*us/wspd1*rho*YSU_G/yget(delp,0)*dt2*(wspd1/ymax(r.wspd[col],1.e-9f))*(wspd1/ymax(r.wspd[col],1.e-9f));if(p.topo){
        // bl_ysu.F90:1254-1314: the paj TKE profile, the hybrid PBL height
        // and the convective velocity weight ctopo onto the surface drag.
        YsuCol tke=thli;
        for(int k=0;k<nz-1;++k){int q0=k*st+col,q1=(k+1)*st+col;float rdza=yget(dza,k+1);
            float dudz=(r.u[q1]-r.u[q0])/rdza,dvdz=(r.v[q1]-r.v[q0])/rdza;
            float su=((-(hgamu/hpbl))+dudz)*(r.u[q1]-r.u[q0])/rdza,sv=((-(hgamv/hpbl))+dvdz)*(r.v[q1]-r.v[q0])/rdza;
            float shear=yget(xkzm,k)*(su+sv),dthdz=(r.theta[q1]-r.theta[q0])/rdza;
            float buoy=yget(xkzh,k)*YSU_G*(1.0f/r.theta[q0])*((-(hgamt/hpbl))+dthdz);
            float zk=karman*yget(zq,k+1),rlamdz;
            if(k+1>=kpbl){rlamdz=ymin(ymax(.1f*rdza,rlam),300.0f);rlamdz=ymin(rdza,rlamdz);}else rlamdz=150.0f;
            float el=zk*rlamdz/(rlamdz+zk),t=16.6f*el*(shear-buoy);
            yset(tke,k,t<=0.0f?0.0f:precise::pow(t,0.66f));}
        float pblh_ysu=ysu_get_pblh(thv,tke,zq,nz,r.xland[col]),vconv;
        if(r.xland[col]<1.5f){float fluxc=ymax(sflux,0.0f);vconv=1.0f*precise::pow(YSU_G/yget(thv,0)*pblh_ysu*fluxc,0.33f);}else vconv=0.0f;
        float vconvnew=0.9f*vconv+1.5f*ymax((pblh_ysu-500.0f)/1000.0f,0.0f),vconvlim=ymin(vconvnew,1.0f);
        yset(diag,0,1.0f+fric*vconvlim+r.ctopo[col]*fric*(1.0f-vconvlim));
    } else yset(diag,0,1.0f+fric);
    yset(rhs,0,u0);
    for(int k=0;k<nz-1;++k){float ds=dt2/yget(delp,k),du=dt2/yget(delp,k+1),dsig=r.p[k*st+col]-r.p[(k+1)*st+col],rdz=1.0f/yget(dza,k+1),tem1=dsig*yget(xkzm,k)*rdz;
        if(pblflg&&k+1<kpbl){float flux=tem1*(-hgamu/hpbl-ufxpbl*yget(zfacent,k)/yget(xkzm,k));yset(rhs,k,yget(rhs,k)+ds*flux);yset(rhs,k+1,r.u[(k+1)*st+col]-du*flux);}
        else if(pblflg&&k+1>=kpbl&&yget(entfac,k)<4.6f){yset(xkzm,k,yclip(sqrt(ymax(yget(xkzh,k)*yget(xkzml,k),0.0f)),xkzminm,xkzmax));yset(rhs,k+1,r.u[(k+1)*st+col]);}
        else yset(rhs,k+1,r.u[(k+1)*st+col]);float dsdz2=dsig*yget(xkzm,k)*rdz*rdz;yset(upper,k,-ds*dsdz2);yset(lower,k+1,-du*dsdz2);yset(diag,k,yget(diag,k)-yget(upper,k));yset(diag,k+1,1.0f-yget(lower,k+1));}
    ythomas(lower,diag,upper,rhs,gamma,nz);for(int k=0;k<nz;++k)r.du[k*st+col]=(yget(rhs,k)-r.u[k*st+col])*rdt;
    for(int k=0;k<nz;++k)yset(rhs,k,r.v[k*st+col]);
    for(int k=0;k<nz-1;++k)if(pblflg&&k+1<kpbl){float ds=dt2/yget(delp,k),du=dt2/yget(delp,k+1),dsig=r.p[k*st+col]-r.p[(k+1)*st+col],rdz=1.0f/yget(dza,k+1),tem1=dsig*yget(xkzm,k)*rdz;float flux=tem1*(-hgamv/hpbl-vfxpbl*yget(zfacent,k)/yget(xkzm,k));yset(rhs,k,yget(rhs,k)+ds*flux);yset(rhs,k+1,r.v[(k+1)*st+col]-du*flux);}
    ythomas(lower,diag,upper,rhs,gamma,nz);for(int k=0;k<nz;++k)r.dv[k*st+col]=(yget(rhs,k)-r.v[k*st+col])*rdt;
    r.exch_h[col]=r.exch_m[col]=0.0f;for(int k=1;k<nz;++k){r.exch_h[k*st+col]=yget(xkzh,k-1);r.exch_m[k*st+col]=yget(xkzm,k-1);}
    r.hpbl[col]=hpbl;r.kpbl[col]=kpbl;r.wstar[col]=wstar;r.delta[col]=delta;r.topdown_radsum[col]=topdown_radsum;r.wstar3_2[col]=wstar3_2;r.cloudflg[col]=cloudflg?1:0;
    if(p.topo)ysu_topo_blend_u10(r,u0,v0,col);
}

// Deterministic first-field finiteness reduction in the public result order.
// The host seeds status with UINT32_MAX; a clean result leaves that sentinel.
kernel void ysu_validate_outputs(
    constant YsuValidateOutputsResources &r [[buffer(0)]],
    constant YsuValidateOutputsParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.nfloat)) return;
    device atomic_uint *status = reinterpret_cast<device atomic_uint *>(r.status);
    const uint i=tid, ncol=uint(p.ncol), nfloat=uint(p.nfloat);
    if(!isfinite(r.du[i])) atomic_fetch_min_explicit(status,0u,memory_order_relaxed);
    if(!isfinite(r.dv[i])) atomic_fetch_min_explicit(status,1u,memory_order_relaxed);
    if(!isfinite(r.dtheta[i])) atomic_fetch_min_explicit(status,2u,memory_order_relaxed);
    if(!isfinite(r.dqv[i])) atomic_fetch_min_explicit(status,3u,memory_order_relaxed);
    if(!isfinite(r.dqc[i])) atomic_fetch_min_explicit(status,4u,memory_order_relaxed);
    if(!isfinite(r.dqi[i])) atomic_fetch_min_explicit(status,5u,memory_order_relaxed);
    if(!isfinite(r.exch_h[i])) atomic_fetch_min_explicit(status,6u,memory_order_relaxed);
    if(!isfinite(r.exch_m[i])) atomic_fetch_min_explicit(status,7u,memory_order_relaxed);
    if(i<ncol){
        if(!isfinite(r.hpbl[i])) atomic_fetch_min_explicit(status,8u,memory_order_relaxed);
        if(!isfinite(r.wstar[i])) atomic_fetch_min_explicit(status,9u,memory_order_relaxed);
        if(!isfinite(r.delta[i])) atomic_fetch_min_explicit(status,10u,memory_order_relaxed);
        if(!isfinite(r.topdown_radsum[i])) atomic_fetch_min_explicit(status,11u,memory_order_relaxed);
        if(!isfinite(r.wstar3_2[i])) atomic_fetch_min_explicit(status,12u,memory_order_relaxed);
    }
}
