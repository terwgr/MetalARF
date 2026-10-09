#include <metal_stdlib>
#include "noahmp_bareflux_family_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

// Source port of the pinned opt_sfc=1 / opt_stc=1 BARE_FLUX call. Native
// Metal log/pow/atan are used here so the public runtime has an executable
// kernel; this entrypoint is a caller slice and does not claim oracle parity.
constant uint ESAT_BITS[28] = {
    0x40C37319u,0x3EE32656u,0x3C6A1E55u,0x398AF867u,0x364B6C50u,0x32AEB9EAu,0x2E86F33Bu,
    0x40C37E63u,0x3F00E367u,0x3C9A8091u,0x39DAF453u,0x36C371F7u,0x334FD334u,0x2F4A2E60u,
    0x3EE33B10u,0x3CEA0BB0u,0x3A501761u,0x374BE117u,0x33DE9991u,0x2FC2326Bu,0xAB479299u,
    0x3F00C69Cu,0x3D1A8D72u,0x3AA632DDu,0x37CFD543u,0x34A15DEFu,0x3114557Bu,0x2CFAE738u
};
constant float B_ZERO=0.0f, B_ONE=1.0f, B_TWO=2.0f, B_VKC=0.4f;
constant float B_TFRZ=273.16f, B_SB=5.67032e-8f, B_GRAV=9.80616f;
constant float B_CPAIR=1004.64f, B_MPE=3.0e-7f;
constant float B_P50=50.0f, B_N50=-50.0f, B_P9=0.9f, B_N5=-5.0f;
constant float B_P622=0.622f, B_P378=0.378f, B_QUARTER=0.25f;

static inline float esat(int base,float t){
    float y=as_type<float>(ESAT_BITS[base+5])+t*as_type<float>(ESAT_BITS[base+6]);
    y=as_type<float>(ESAT_BITS[base+4])+t*y; y=as_type<float>(ESAT_BITS[base+3])+t*y;
    y=as_type<float>(ESAT_BITS[base+2])+t*y; y=as_type<float>(ESAT_BITS[base+1])+t*y;
    return 100.0f*(as_type<float>(ESAT_BITS[base])+t*y);
}
static inline float minf(float a,float b){return a<b?a:b;}
static inline float maxf(float a,float b){return a>b?a:b;}
struct Sfc {float moz,fm,fh,fm2,fh2,fv,cm,ch,ch2;int flips;};
static inline void sfcdif(thread Sfc &s,int it,float ts,float rho,float h,float q,
                          float zl,float zp,float z0m,float z0h,float ur){
    float old=s.moz;
    float cm0=log((zl-zp)/z0m),ch0=log((zl-zp)/z0h);
    float cm2=log((2.0f+z0m)/z0m),ch2=log((2.0f+z0h)/z0h);
    float mol=0.0f,moz2=0.0f;
    if(it==1){s.fv=0.0f;s.moz=0.0f;}else{
        float tv=(1.0f+0.61f*q)*ts;
        float t=(((B_VKC*(B_GRAV/tv))*h)/(rho*B_CPAIR));if(fabs(t)<=B_MPE)t=B_MPE;
        mol=-pow(s.fv,3.0f)/t;s.moz=minf((zl-zp)/mol,1.0f);moz2=minf((2.0f+z0h)/mol,1.0f);
    }
    if(old*s.moz<0.0f)s.flips++;
    if(s.flips>=2){s.moz=s.fm=s.fh=moz2=s.fm2=s.fh2=0.0f;}
    float fmnew,fhnew,fm2new,fh2new;
    if(s.moz<0.0f){
        float a=pow(1.0f-16.0f*s.moz,0.25f),b=log((1.0f+a*a)/2.0f),c=log((1.0f+a)/2.0f);
        fmnew=2.0f*c+b-2.0f*atan(a)+1.5707963267948966f;fhnew=2.0f*b;
        float aa=pow(1.0f-16.0f*moz2,0.25f),bb=log((1.0f+aa*aa)/2.0f),cc=log((1.0f+aa)/2.0f);
        fm2new=2.0f*cc+bb-2.0f*atan(aa)+1.5707963267948966f;fh2new=2.0f*bb;
    }else{fmnew=-5.0f*s.moz;fhnew=fmnew;fm2new=-5.0f*moz2;fh2new=fm2new;}
    if(it==1){s.fm=fmnew;s.fh=fhnew;s.fm2=fm2new;s.fh2=fh2new;}
    else{s.fm=0.5f*(s.fm+fmnew);s.fh=0.5f*(s.fh+fhnew);s.fm2=0.5f*(s.fm2+fm2new);s.fh2=0.5f*(s.fh2+fh2new);}
    s.fh=minf(s.fh,0.9f*ch0);s.fm=minf(s.fm,0.9f*cm0);s.fh2=minf(s.fh2,0.9f*ch2);s.fm2=minf(s.fm2,0.9f*cm2);
    float cmfm=cm0-s.fm,chfh=ch0-s.fh,ch2fh2=ch2-s.fh2;
    if(fabs(cmfm)<=B_MPE)cmfm=B_MPE;if(fabs(chfh)<=B_MPE)chfh=B_MPE;if(fabs(ch2fh2)<=B_MPE)ch2fh2=B_MPE;
    s.cm=(B_VKC*B_VKC)/(cmfm*cmfm);s.ch=(B_VKC*B_VKC)/(cmfm*chfh);
    s.fv=ur*sqrt(s.cm);s.ch2=(B_VKC*s.fv)/ch2fh2;(void)mol;
}

kernel void noahmp_bare_flux(constant NoahmpBareFluxResources &r [[buffer(0)]],
                             constant NoahmpBareFluxParams &q [[buffer(1)]],
                             uint idx [[thread_position_in_grid]]){
    if(idx>=uint(q.n))return;ulong xb=ulong(idx)*53ul,ib=ulong(idx)*5ul,ob=ulong(idx)*13ul;
    int isnow=r.ii[ib],urban=r.ii[ib+4];
    float sag=r.in[xb+1],lwdn=r.in[xb+2],ur=r.in[xb+3],uu=r.in[xb+4],vv=r.in[xb+5];
    float sfctmp=r.in[xb+6],qair=r.in[xb+8],eair=r.in[xb+9],rho=r.in[xb+10];
    float snowh=r.in[xb+11],zl=r.in[xb+12],zp=r.in[xb+13],z0m=r.in[xb+14];
    float emg=r.in[xb+16],rsurf=r.in[xb+17],lathea=r.in[xb+18],gamma=r.in[xb+19];
    float rhsur=r.in[xb+20],pahb=r.in[xb+22],psfc=r.in[xb+26],tgb=r.in[xb+28],qsfc=r.in[xb+31];
    ulong lay=xb+32ul;int kk=isnow+3;
    float cgh=(2.0f*r.in[lay+14ul+ulong(kk)]) / r.in[lay+ulong(kk)];
    float cir=emg*B_SB,h=0.0f,csh=0.0f,cev=0.0f,estg=0.0f,ehb=0.0f;
    float irb=0.0f,shb=0.0f,evb=0.0f,ghb=0.0f,z0h=z0m;
    Sfc s={0.0f,0.0f,0.0f,0.0f,0.0f,0.1f,0.0f,0.0f,0.0f,0};
    for(int it=1;it<=5;++it){
        z0h=z0m;sfcdif(s,it,sfctmp,rho,h,qair,zl,zp,z0m,z0h,ur);
        float rah=maxf(1.0f,1.0f/(s.ch*ur)),raw=rah;ehb=1.0f/rah;
        float t=minf(B_P50,maxf(B_N50,tgb-B_TFRZ)),destg;
        if(t>0.0f){estg=esat(0,t);destg=esat(14,t);}else{estg=esat(7,t);destg=esat(21,t);}
        csh=(rho*B_CPAIR)/rah;cev=((rho*B_CPAIR)/gamma)/(rsurf+raw);
        irb=cir*pow(tgb,4.0f)-emg*lwdn;shb=csh*(tgb-sfctmp);evb=cev*(estg*rhsur-eair);
        ghb=cgh*(tgb-r.in[lay+7ul+ulong(kk)]);
        float b=sag-irb-shb-evb-ghb+pahb,cir4t3=4.0f*cir*pow(tgb,3.0f);
        float a=cir4t3+csh+cev*destg+cgh,dtg=b/a;
        irb+=cir4t3*dtg;shb+=csh*dtg;evb+=cev*destg*dtg;ghb+=cgh*dtg;
        tgb+=dtg;h=csh*(tgb-sfctmp);
        t=minf(B_P50,maxf(B_N50,tgb-B_TFRZ));estg=t>0.0f?esat(0,t):esat(7,t);
        float er=estg*rhsur;qsfc=(B_P622*er)/(psfc-B_P378*er);
    }
    if(snowh>0.05f&&tgb>B_TFRZ){tgb=B_TFRZ;irb=cir*pow(tgb,4.0f)-emg*lwdn;shb=csh*(tgb-sfctmp);evb=cev*(estg*rhsur-eair);ghb=sag+pahb-(irb+shb+evb);}
    float taux=-((rho*s.cm*ur)*uu),tauy=-((rho*s.cm*ur)*vv);
    float ehb2=(s.fv*B_VKC)/(log((2.0f+z0h)/z0h)-s.fh2),cq=ehb2,t2m,q2;
    if(ehb2<1.0e-5f){t2m=tgb;q2=qsfc;}else{t2m=tgb-((shb/(rho*B_CPAIR))/ehb2);q2=qsfc-((evb/(lathea*rho))*(1.0f/cq+rsurf));}
    if(urban)q2=qsfc;
    r.out[ob]=tgb;r.out[ob+1]=s.cm;r.out[ob+2]=ehb;r.out[ob+3]=qsfc;
    r.out[ob+4]=taux;r.out[ob+5]=tauy;r.out[ob+6]=irb;r.out[ob+7]=shb;r.out[ob+8]=evb;r.out[ob+9]=ghb;
    r.out[ob+10]=t2m;r.out[ob+11]=q2;r.out[ob+12]=ehb2;
}
