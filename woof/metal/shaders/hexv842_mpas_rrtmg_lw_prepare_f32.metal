#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#include "hexv842_mpas_rrtmg_lw_prepare_f32_bindings.metalh"
constant float mpas_retab[95] = {5.92779f,6.26422f,6.61973f,6.99539f,7.39234f,7.81177f,8.25496f,8.72323f,9.21800f,9.74075f,10.2930f,10.8765f,11.4929f,12.1440f,12.8317f,13.5581f,14.2319f,15.0351f,15.8799f,16.7674f,17.6986f,18.6744f,19.6955f,20.7623f,21.8757f,23.0364f,24.2452f,25.5034f,26.8125f,27.7895f,28.6450f,29.4167f,30.1088f,30.7306f,31.2943f,31.8151f,32.3077f,32.7870f,33.2657f,33.7540f,34.2601f,34.7892f,35.3442f,35.9255f,36.5316f,37.1602f,37.8078f,38.4720f,39.1508f,39.8442f,40.5552f,41.2912f,42.0635f,42.8876f,43.7863f,44.7853f,45.9170f,47.2165f,48.7221f,50.4710f,52.4980f,54.8315f,57.4898f,60.4785f,63.7898f,65.5604f,71.2885f,75.4113f,79.7368f,84.2351f,88.8833f,93.6658f,98.5739f,103.603f,108.752f,114.025f,119.424f,124.954f,130.630f,136.457f,142.446f,148.608f,154.956f,161.503f,168.262f,175.248f,182.473f,189.952f,197.699f,205.728f,214.055f,222.694f,231.661f,240.971f,250.639f};
constant float mpas_pprof[60] = {1000.00f,855.47f,731.82f,626.05f,535.57f,458.16f,391.94f,335.29f,286.83f,245.38f,209.91f,179.57f,153.62f,131.41f,112.42f,96.17f,82.27f,70.38f,60.21f,51.51f,44.06f,37.69f,32.25f,27.59f,23.60f,20.19f,17.27f,14.77f,12.64f,10.81f,9.25f,7.91f,6.77f,5.79f,4.95f,4.24f,3.63f,3.10f,2.65f,2.27f,1.94f,1.66f,1.42f,1.22f,1.04f,0.89f,0.76f,0.65f,0.56f,0.48f,0.41f,0.35f,0.30f,0.26f,0.22f,0.19f,0.16f,0.14f,0.12f,0.10f};
constant float mpas_tprof[60] = {286.96f,281.07f,275.16f,268.11f,260.56f,253.02f,245.62f,238.41f,231.57f,225.91f,221.72f,217.79f,215.06f,212.74f,210.25f,210.16f,210.69f,212.14f,213.74f,215.37f,216.82f,217.94f,219.03f,220.18f,221.37f,222.64f,224.16f,225.88f,227.63f,229.51f,231.50f,233.73f,236.18f,238.78f,241.60f,244.44f,247.35f,250.33f,253.32f,256.30f,259.22f,262.12f,264.80f,266.50f,267.59f,268.44f,268.69f,267.76f,266.13f,263.96f,261.54f,258.93f,256.15f,253.23f,249.89f,246.67f,243.48f,240.25f,236.66f,233.86f};
constant float mpas_o3sum[31] = {5.297E-8f,5.852E-8f,6.579E-8f,7.505E-8f,8.577E-8f,9.895E-8f,1.175E-7f,1.399E-7f,1.677E-7f,2.003E-7f,2.571E-7f,3.325E-7f,4.438E-7f,6.255E-7f,8.168E-7f,1.036E-6f,1.366E-6f,1.855E-6f,2.514E-6f,3.240E-6f,4.033E-6f,4.854E-6f,5.517E-6f,6.089E-6f,6.689E-6f,1.106E-5f,1.462E-5f,1.321E-5f,9.856E-6f,5.960E-6f,5.960E-6f};
constant float mpas_ppsum[31] = {955.890f,850.532f,754.599f,667.742f,589.841f,519.421f,455.480f,398.085f,347.171f,301.735f,261.310f,225.360f,193.419f,165.490f,141.032f,120.125f,102.689f,87.829f,75.123f,64.306f,55.086f,47.209f,40.535f,34.795f,29.865f,19.122f,9.277f,4.660f,2.421f,1.294f,0.647f};
constant float mpas_o3win[31] = {4.629E-8f,4.686E-8f,5.017E-8f,5.613E-8f,6.871E-8f,8.751E-8f,1.138E-7f,1.516E-7f,2.161E-7f,3.264E-7f,4.968E-7f,7.338E-7f,1.017E-6f,1.308E-6f,1.625E-6f,2.011E-6f,2.516E-6f,3.130E-6f,3.840E-6f,4.703E-6f,5.486E-6f,6.289E-6f,6.993E-6f,7.494E-6f,8.197E-6f,9.632E-6f,1.113E-5f,1.146E-5f,9.389E-6f,6.135E-6f,6.135E-6f};
constant float mpas_ppwin[31] = {955.747f,841.783f,740.199f,649.538f,568.404f,495.815f,431.069f,373.464f,322.354f,277.190f,237.635f,203.433f,174.070f,148.949f,127.408f,108.915f,93.114f,79.551f,67.940f,58.072f,49.593f,42.318f,36.138f,30.907f,26.362f,16.423f,7.583f,3.620f,1.807f,0.938f,0.469f};
#define M(f,k) m[(f)*nm+(k)]
#define S(f) surface[(f)]
#define A(f,k) a[(f)*nl+(k)]
#define P(k) lev[(k)]
#define T(k) lev[nl+1+(k)]

static inline float mpas_profile(float p) {
    // MPAS: module_ra_rrtmg_lw.F:11954-11983.
    int j=59;
    if (mpas_pprof[59]<p) {
        for(int kk=1;kk<60;++kk) if(mpas_pprof[kk]<p){j=kk-1;break;}
    }
    float weight=j==59?0.0f:RLW_DV(RLW_SU(p,mpas_pprof[j]),RLW_SU(mpas_pprof[j+1],mpas_pprof[j]));
    float next=mpas_tprof[j==59?j:j+1];
    return RLW_AD(RLW_MU(weight,RLW_SU(next,mpas_tprof[j])),mpas_tprof[j]);
}
static inline float mpas_ice_radius(float t,int cap){
    // MPAS: module_ra_rrtmg_lw.F:12050-12071,12508-12535.
    int j=min(max((int)RLW_SU(t,179.0f),1),cap)-1;
    float corr=RLW_SU(t,(float)(int)t);
    return RLW_AD(RLW_MU(mpas_retab[j],RLW_SU(1.0f,corr)),RLW_MU(mpas_retab[j+1],corr));
}
static inline float mpas_liquid_radius(float t,float land,float snow,float ice){
    // MPAS: module_ra_rrtmg_lw.F:12447-12506.
    float r=RLW_AD(8.0f,RLW_MU(6.0f,mpas_min(1.0f,mpas_max(0.0f,RLW_MU(RLW_SU(273.16f,t),0.05f)))));
    r=RLW_AD(r,RLW_MU(RLW_SU(14.0f,r),mpas_min(1.0f,mpas_max(0.0f,RLW_MU(RLW_MU(0.001f,snow),10.0f)))));
    r=RLW_AD(r,RLW_MU(RLW_SU(14.0f,r),mpas_min(1.0f,mpas_max(0.0f,RLW_SU(1.0f,RLW_SU(2.0f,land))))));
    return RLW_AD(r,RLW_MU(RLW_SU(14.0f,r),mpas_min(1.0f,mpas_max(0.0f,ice))));
}
static inline float mpas_vinterp_one(float p,device const float* pin,device const float* ozone,int nz,thread int* upper){
    // MPAS: module_ra_rrtmg_vinterp.F:21-179. Called top to bottom.
    for(int kk=*upper;kk<nz-1;++kk) {
        if(pin[kk]<p && p<=pin[kk+1]){*upper=kk;break;}
    }
    if(p<pin[0]) return RLW_DV(RLW_MU(ozone[0],p),pin[0]);
    if(p>pin[nz-1]) return ozone[nz-1];
    float lower_pin=pin[*upper+1];
    float lower_ozone=ozone[*upper+1];
    float dpu=RLW_SU(p,pin[*upper]),dpl=RLW_SU(lower_pin,p);
    return RLW_DV(RLW_AD(RLW_MU(ozone[*upper],dpl),RLW_MU(lower_ozone,dpu)),RLW_AD(dpl,dpu));
}
static inline void mpas_lw_prepare_one(int nm,int nl,int icloud,int reqc,int reqi,int reqs,int o3input,int nz,int presence,
 device const float* m,device const float* interfaces,device const float* surface,device const float* pin,device const float* ozone,device float* a,device float* lev){
    float dzsum=0.0f;
    for(int k=0;k<=nm;++k){P(k)=RLW_DV(interfaces[k],100.0f);T(k)=interfaces[nm+1+k];}
    for(int k=0;k<nl;++k){
        float dz=M(2,min(k,nm-1));
        A(2,k)=RLW_AD(dzsum,RLW_MU(0.5f,dz));dzsum=RLW_AD(dzsum,dz);
        A(3,k)=RLW_MU(mpas_max(M(3,min(k,nm-1)),1.e-12f),1.607793f);
        if(k<nm){A(0,k)=RLW_DV(M(0,k),100.0f);A(1,k)=M(1,k);}
        else {P(k+1)=mpas_max(RLW_SU(P(k),4.0f),0.0f);A(0,k)=RLW_MU(0.5f,RLW_AD(P(k),P(k+1)));}
    }
    P(nl)=0.0f;A(0,nl-1)=RLW_MU(0.5f,RLW_AD(P(nl-1),P(nl)));
    float anchor=RLW_SU(T(nm-1),mpas_profile(P(nm-1)));
    for(int k=nm;k<=nl;++k){T(k)=RLW_AD(mpas_profile(P(k)),anchor);A(1,k-1)=RLW_MU(0.5f,RLW_AD(T(k),T(k-1)));}
    // MPAS: module_ra_rrtmg_lw.F:12308-12444 (INIRAD/O3DATA).
    float annual[31],phalf[32];phalf[0]=1100.0f;phalf[31]=0.0f;
    annual[0]=RLW_MU(0.5f,RLW_AD(mpas_o3sum[0],mpas_o3win[0]));
    for(int j=1;j<31;++j){
        float interp=RLW_AD(mpas_o3win[j-1],RLW_MU(RLW_DV(RLW_SU(mpas_o3win[j],mpas_o3win[j-1]),RLW_SU(mpas_ppwin[j],mpas_ppwin[j-1])),RLW_SU(mpas_ppsum[j],mpas_ppwin[j-1])));
        annual[j]=RLW_MU(0.5f,RLW_AD(interp,mpas_o3sum[j]));
        phalf[j]=RLW_DV(RLW_AD(mpas_ppsum[j],mpas_ppsum[j-1]),2.0f);
    }
    for(int k=0;k<nl;++k){
        float o3=0.0f;
        for(int j=0;j<31;++j){
            float pb1=mpas_max(RLW_SU(P(k),phalf[j]),0.0f),pb2=mpas_max(RLW_SU(P(k),phalf[j+1]),0.0f);
            float pt1=mpas_max(RLW_SU(P(k+1),phalf[j]),0.0f),pt2=mpas_max(RLW_SU(P(k+1),phalf[j+1]),0.0f);
            o3=RLW_AD(o3,RLW_MU(RLW_AD(RLW_SU(RLW_SU(pb2,pb1),pt2),pt1),annual[j]));
        }
        A(4,k)=RLW_MU(RLW_DV(o3,RLW_SU(P(k),P(k+1))),0.603461f);
    }
    if(o3input==2){int upper=0;for(int k=nl-1;k>=0;--k) A(4,k)=mpas_vinterp_one(A(0,k),pin,ozone,nz,&upper);}
    // MPAS: module_ra_rrtmg_lw.F:12016-12164.
    for(int k=0;k<nl;++k){
        // MPAS: module_ra_rrtmg_lw.F:11801-11806. PRESENT is independent of mass.
        float cf=k<nm&&icloud>0&&(presence&7)&&(presence&8)?M(7,k):0.0f;A(5,k)=cf;
        float rel=10.0f,rei=10.0f,res=10.0f,liq=0.0f,ice=0.0f,snow=0.0f;
        if(k<nm){
            if(reqc){rel=mpas_max(2.5f,RLW_MU(M(8,k),1.e6f));if(rel<=2.5f&&soft32_greater(as_type<uint>(cf),0u)){if(S(2)>1.5f)rel=10.5f;else if(S(2)<1.5f)rel=7.5f;}}
            else rel=mpas_liquid_radius(A(1,k),S(2),S(4),S(3));
            if(reqi){rei=mpas_max(5.0f,RLW_MU(M(9,k),1.e6f));if(rei<=5.0f&&soft32_greater(as_type<uint>(cf),0u))rei=mpas_max(5.0f,mpas_ice_radius(M(1,k),75));}
            else{rei=mpas_ice_radius(A(1,k),94);rei=mpas_min(140.0f,RLW_MU(rei,1.0315f));}
            if(reqs)res=mpas_max(10.0f,RLW_MU(M(10,k),1.e6f));
            float dp=RLW_SU(P(k),P(k+1)),denom=mpas_max(0.01f,cf);
            float qc=icloud>0&&(presence&1)?mpas_max(M(4,k),0.0f):0.0f;
            float qi=icloud>0&&(presence&2)?mpas_max(M(5,k),0.0f):0.0f;
            float qs=icloud>0&&(presence&4)?mpas_max(M(6,k),0.0f):0.0f;
            liq=RLW_DV(RLW_MU(RLW_DV(RLW_MU(RLW_MU(qc,dp),100.0f),9.80616f),1000.0f),denom);
            float qice=reqi||reqs?qi:RLW_AD(qi,qs);
            ice=RLW_DV(RLW_MU(RLW_DV(RLW_MU(RLW_MU(qice,dp),100.0f),9.80616f),1000.0f),denom);
            if(reqs){float factor=1.0f;if(res>130.0f){float ratio=RLW_DV(130.0f,res);factor=RLW_MU(ratio,ratio);res=130.0f;}
                snow=RLW_DV(RLW_MU(factor,RLW_MU(RLW_DV(RLW_MU(RLW_MU(qs,dp),100.0f),9.80616f),1000.0f)),denom);}
        }
        A(6,k)=liq;A(7,k)=ice;A(8,k)=snow;A(9,k)=rel;A(10,k)=rei;A(11,k)=res;
    }
}
kernel void hexv842_mpas_rrtmg_lw_prepare_f32(constant Hexv842MpasRrtmgLwPrepareF32Resources &r [[buffer(0)]], constant Hexv842MpasRrtmgLwPrepareF32Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
if(c>=uint(p.nc))return;
mpas_lw_prepare_one(p.nm,p.nl,p.icloud,p.reqc,p.reqi,p.reqs,p.o3input,p.nz,p.presence,r.m+ulong(c)*12*p.nm,r.interfaces+ulong(c)*2*(p.nm+1),r.surface+ulong(c)*6,r.pin,r.ozone+(p.o3input==2?ulong(c)*p.nz:0),r.a+ulong(c)*12*p.nl,r.lev+ulong(c)*2*(p.nl+1));
}
