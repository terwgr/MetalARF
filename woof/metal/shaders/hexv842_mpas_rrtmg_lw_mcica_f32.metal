#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "soft_gfk_exp.metalh"
#include "hexv842_mpas_rrtmg_lw_mcica_f32_bindings.metalh"
#pragma clang fp contract(off)
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#define rlw_exp(x) as_type<float>(soft_gfk_exp(as_type<uint>(x)))
#define M(f,k) m[(f)*nm+(k)]
#define S(f) surface[(f)]
#define A(f,k) a[(f)*nl+(k)]
#define P(k) lev[(k)]
#define T(k) lev[nl+1+(k)]

static inline float mpas_kiss(thread uint* seed){
    // MPAS: module_ra_rrtmg_lw.F:2705-2737. Unsigned implements wrap/ISHFT.
    seed[0]=69069u*seed[0]+1327217885u;
    seed[1]^=seed[1]<<13;seed[1]^=seed[1]>>17;seed[1]^=seed[1]<<5;
    seed[2]=18000u*(seed[2]&65535u)+(seed[2]>>16);
    seed[3]=30903u*(seed[3]&65535u)+(seed[3]>>16);
    int kiss=as_type<int>(seed[0]+seed[1]+(seed[2]<<16)+seed[3]);
    return RLW_AD(RLW_MU((float)kiss,2.328306e-10f),0.5f);
}
static inline void mpas_mcica_make(int nl,int overlap,int idcor,int julday,device const float* surface,device const float* a,
    device float* cloud,device float* liquid,device float* ice,device float* snow){
    // MPAS: module_ra_rrtmg_lw.F:2099-2737. One MPAS wrapper column.
    if(overlap==0)return;
    unsigned int seed[4];
    {
        // MPAS: module_ra_rrtmg_lw.F:2432-2440. Short-column native OOB seed
        // words measured as zero in the pristine McICA boundary matrix; full
        // wrapper short-column run-to-run variability remains unqualified.
        for(int k=0;k<4;++k){seed[k]=0u;if(k<nl){float p=RLW_MU(A(0,k),100.0f);seed[k]=(unsigned int)(int)RLW_MU(RLW_SU(p,(float)(int)p),1.e9f);}}
        for(int j=0;j<150;++j)mpas_kiss(seed);
    }
    float length=2500.0f;
    if(idcor==1){
        float am3=RLW_MU(julday>181?(-28.0f/365.0f):(28.0f/365.0f),(float)(julday-(julday>181?272:91)));
        float delta=RLW_SU(surface[5],am3);
        length=RLW_MU(RLW_AD(1.4315f,RLW_MU(2.1219f,rlw_exp(RLW_DV(-RLW_MU(delta,delta),(-25.584f*-25.584f))))),1000.0f);
    }
    float inverse=RLW_DV(1.0f,length);
    for(int ig=0;ig<140;++ig){
        float prev=0.0f,prevcf=0.0f,shared=0.0f;
        if(overlap==3)shared=mpas_kiss(seed);
        for(int k=0;k<nl;++k){
            float cf=A(5,k);if(cf<1.e-20f)cf=0.0f;
            float cdf=overlap==3?shared:(mpas_kiss(seed));
            if(overlap>=4){float cdf2=mpas_kiss(seed);
                float alpha=k?rlw_exp(RLW_MU(-RLW_SU(A(2,k),A(2,k-1)),inverse)):0.0f;
                if(overlap==5&&cf==0.0f&&prevcf>0.0f)alpha=0.0f;
                if(k&&cdf2<alpha)cdf=prev;
            }else if(overlap==2&&k){float threshold=RLW_SU(1.0f,prevcf);cdf=prev>threshold?prev:RLW_MU(cdf,threshold);}
            bool cloudy=cdf>=RLW_SU(1.0f,cf);int idx=k*140+ig;
            cloud[idx]=cloudy?1.0f:0.0f;liquid[idx]=cloudy?A(6,k):0.0f;ice[idx]=cloudy?A(7,k):0.0f;snow[idx]=cloudy?A(8,k):0.0f;
            prev=cdf;prevcf=cf;
        }
    }
}
kernel void hexv842_mpas_rrtmg_lw_mcica_f32(constant Hexv842MpasRrtmgLwMcicaF32Resources &r [[buffer(0)]], constant Hexv842MpasRrtmgLwMcicaF32Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
if(c>=uint(p.nc))return;
mpas_mcica_make(p.nl,p.overlap,p.idcor,p.julday,r.surface+ulong(c)*6,r.a+ulong(c)*12*p.nl,r.cloud+ulong(c)*p.nl*140,r.liquid+ulong(c)*p.nl*140,r.ice+ulong(c)*p.nl*140,r.snow+ulong(c)*p.nl*140);
}
