#include <metal_stdlib>
#include "rrtmg_aer3_sw_optics_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
using namespace metal;
#define AER3_NB 14
#define AER3_NRH 8
#define FADD(a,b) as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)))
#define FSUB(a,b) as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)))
#define FMUL(a,b) as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)))
#define FDIV(a,b) as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)))
constant float AER3_GT_RH_ARR[AER3_NRH] = {
    10.f, 60.f, 70.f, 80.f, 85.f, 90.f, 95.f, 99.8f};
constant float AER3_GT_LOOKUP[AER3_NRH][4][2] = {
    {{5.73936E-15f, 2.63577E-12f}, {5.73936E-15f, 2.63577E-12f},
     {5.73936E-15f, 2.63577E-12f}, {5.73936E-15f, 2.63577E-12f}},
    {{6.93515E-15f, 2.72095E-12f}, {6.93168E-15f, 2.72092E-12f},
     {6.92570E-15f, 2.72091E-12f}, {6.91833E-15f, 2.72087E-12f}},
    {{7.24707E-15f, 2.77219E-12f}, {7.23809E-15f, 2.77222E-12f},
     {7.23108E-15f, 2.77201E-12f}, {7.21800E-15f, 2.77111E-12f}},
    {{8.95130E-15f, 2.87263E-12f}, {9.01582E-15f, 2.87252E-12f},
     {9.13216E-15f, 2.87241E-12f}, {9.16219E-15f, 2.87211E-12f}},
    {{1.06695E-14f, 2.96752E-12f}, {1.06370E-14f, 2.96726E-12f},
     {1.05999E-14f, 2.96702E-12f}, {1.05443E-14f, 2.96603E-12f}},
    {{1.37908E-14f, 3.15081E-12f}, {1.37172E-14f, 3.15020E-12f},
     {1.36362E-14f, 3.14927E-12f}, {1.35287E-14f, 3.14817E-12f}},
    {{2.26019E-14f, 3.66798E-12f}, {2.24435E-14f, 3.66540E-12f},
     {2.23254E-14f, 3.66173E-12f}, {2.20496E-14f, 3.65796E-12f}},
    {{4.41983E-13f, 7.50091E-11f}, {3.93335E-13f, 6.79097E-11f},
     {3.45569E-13f, 6.07845E-11f}, {2.96971E-13f, 5.36085E-11f}}};

// calc_spectral_* RH nodes and the aer_type = 1 (rural) rows.
constant float AER3_RHS[AER3_NRH] = {
    0.f, 50.f, 70.f, 80.f, 90.f, 95.f, 98.f, 99.f};
constant float AER3_RAOD[AER3_NRH][AER3_NB] = {
    {0.0735f, 0.0997f, 0.1281f, 0.1529f, 0.1882f, 0.2512f, 0.3010f, 0.4550f,
     0.7159f, 1.0357f, 1.3582f, 1.6760f, 2.2523f, 0.0582f},
    {0.0741f, 0.1004f, 0.1289f, 0.1537f, 0.1891f, 0.2522f, 0.3021f, 0.4560f,
     0.7166f, 1.0351f, 1.3547f, 1.6687f, 2.2371f, 0.0587f},
    {0.0752f, 0.1017f, 0.1304f, 0.1554f, 0.1909f, 0.2542f, 0.3042f, 0.4580f,
     0.7179f, 1.0342f, 1.3485f, 1.6559f, 2.2102f, 0.0596f},
    {0.0766f, 0.1034f, 0.1323f, 0.1575f, 0.1932f, 0.2567f, 0.3068f, 0.4605f,
     0.7196f, 1.0332f, 1.3411f, 1.6407f, 2.1785f, 0.0608f},
    {0.0807f, 0.1083f, 0.1379f, 0.1635f, 0.1998f, 0.2639f, 0.3143f, 0.4677f,
     0.7244f, 1.0305f, 1.3227f, 1.6031f, 2.1006f, 0.0644f},
    {0.0884f, 0.1174f, 0.1482f, 0.1746f, 0.2118f, 0.2769f, 0.3277f, 0.4805f,
     0.7328f, 1.0272f, 1.2977f, 1.5525f, 1.9976f, 0.0712f},
    {0.1072f, 0.1391f, 0.1724f, 0.2006f, 0.2396f, 0.3066f, 0.3581f, 0.5087f,
     0.7510f, 1.0231f, 1.2622f, 1.4818f, 1.8565f, 0.0878f},
    {0.1286f, 0.1635f, 0.1991f, 0.2288f, 0.2693f, 0.3377f, 0.3895f, 0.5372f,
     0.7686f, 1.0213f, 1.2407f, 1.4394f, 1.7739f, 0.1072f}};
constant float AER3_SSA[AER3_NRH][AER3_NB] = {
    {0.8730f, 0.6695f, 0.8530f, 0.8601f, 0.8365f, 0.7949f, 0.8113f, 0.8810f,
     0.9305f, 0.9436f, 0.9532f, 0.9395f, 0.8007f, 0.8634f},
    {0.8428f, 0.6395f, 0.8571f, 0.8645f, 0.8408f, 0.8007f, 0.8167f, 0.8845f,
     0.9326f, 0.9454f, 0.9545f, 0.9416f, 0.8070f, 0.8589f},
    {0.8000f, 0.6025f, 0.8668f, 0.8740f, 0.8503f, 0.8140f, 0.8309f, 0.8943f,
     0.9370f, 0.9489f, 0.9577f, 0.9451f, 0.8146f, 0.8548f},
    {0.7298f, 0.5666f, 0.9030f, 0.9049f, 0.8863f, 0.8591f, 0.8701f, 0.9178f,
     0.9524f, 0.9612f, 0.9677f, 0.9576f, 0.8476f, 0.8578f},
    {0.7010f, 0.5606f, 0.9312f, 0.9288f, 0.9183f, 0.9031f, 0.9112f, 0.9439f,
     0.9677f, 0.9733f, 0.9772f, 0.9699f, 0.8829f, 0.8590f},
    {0.6933f, 0.5620f, 0.9465f, 0.9393f, 0.9346f, 0.9290f, 0.9332f, 0.9549f,
     0.9738f, 0.9782f, 0.9813f, 0.9750f, 0.8980f, 0.8594f},
    {0.6842f, 0.5843f, 0.9597f, 0.9488f, 0.9462f, 0.9470f, 0.9518f, 0.9679f,
     0.9808f, 0.9839f, 0.9864f, 0.9794f, 0.9113f, 0.8648f},
    {0.6786f, 0.5897f, 0.9658f, 0.9522f, 0.9530f, 0.9610f, 0.9651f, 0.9757f,
     0.9852f, 0.9871f, 0.9883f, 0.9835f, 0.9236f, 0.8618f}};
constant float AER3_ASY[AER3_NRH][AER3_NB] = {
    {0.7444f, 0.7711f, 0.7306f, 0.7103f, 0.6693f, 0.6267f, 0.6169f, 0.6207f,
     0.6341f, 0.6497f, 0.6630f, 0.6748f, 0.7208f, 0.7419f},
    {0.7444f, 0.7747f, 0.7314f, 0.7110f, 0.6711f, 0.6301f, 0.6210f, 0.6251f,
     0.6392f, 0.6551f, 0.6680f, 0.6799f, 0.7244f, 0.7436f},
    {0.7438f, 0.7845f, 0.7341f, 0.7137f, 0.6760f, 0.6381f, 0.6298f, 0.6350f,
     0.6497f, 0.6657f, 0.6790f, 0.6896f, 0.7300f, 0.7477f},
    {0.7336f, 0.7934f, 0.7425f, 0.7217f, 0.6925f, 0.6665f, 0.6616f, 0.6693f,
     0.6857f, 0.7016f, 0.7139f, 0.7218f, 0.7495f, 0.7574f},
    {0.7111f, 0.7865f, 0.7384f, 0.7198f, 0.6995f, 0.6864f, 0.6864f, 0.6987f,
     0.7176f, 0.7326f, 0.7427f, 0.7489f, 0.7644f, 0.7547f},
    {0.7009f, 0.7828f, 0.7366f, 0.7196f, 0.7034f, 0.6958f, 0.6979f, 0.7118f,
     0.7310f, 0.7452f, 0.7542f, 0.7593f, 0.7692f, 0.7522f},
    {0.7226f, 0.8127f, 0.7621f, 0.7434f, 0.7271f, 0.7231f, 0.7248f, 0.7351f,
     0.7506f, 0.7622f, 0.7688f, 0.7719f, 0.7756f, 0.7706f},
    {0.7296f, 0.8219f, 0.7651f, 0.7513f, 0.7404f, 0.7369f, 0.7386f, 0.7485f,
     0.7626f, 0.7724f, 0.7771f, 0.7789f, 0.7790f, 0.7760f}};

static inline float aer3_exp(float x) { return as_type<float>(soft_gfk_exp(as_type<uint>(x))); }
static inline int aer3_nint(float x) { return int(floor(x+(x<0.0f?-0.5f:0.5f))); }
static inline float aer3_rslf(float p,float t) {
 const float c0=.611583699E03f,c1=.444606896E02f,c2=.143177157E01f,c3=.264224321E-1f,c4=.299291081E-3f,c5=.203154182E-5f,c6=.702620698E-8f,c7=.379534310E-11f,c8=-.321582393E-13f;
 float x=max(-80.0f,FSUB(t,273.16f)),e=FADD(c7,FMUL(x,c8));
 e=FADD(c6,FMUL(x,e));e=FADD(c5,FMUL(x,e));e=FADD(c4,FMUL(x,e));e=FADD(c3,FMUL(x,e));e=FADD(c2,FMUL(x,e));e=FADD(c1,FMUL(x,e));e=FADD(c0,FMUL(x,e));e=min(e,FMUL(p,0.15f));
 return FDIV(FMUL(0.622f,e),FSUB(p,e));
}
static inline float aer3_gt_aod(float p,float dz,float t,float qv,float nwfa,float nifa) {
 int ti=max(1,min(aer3_nint(FSUB(10.999f,FMUL(0.0333f,t))),4));
 float rh=min(98.0f,max(10.1f,FMUL(FDIV(qv,aer3_rslf(p,t)),100.0f)));int i1,i2;
 if(rh<60.0f){i1=1;i2=2;}else{int idx=rh<80.0f?aer3_nint(FADD(FMUL(.1f,rh),-4.0f)):min(AER3_NRH,aer3_nint(FADD(FMUL(.2f,rh),-12.0f)));float d=FSUB(rh,AER3_GT_RH_ARR[idx-1]);if(d<0){i1=idx-1;i2=idx;}else{i1=idx;i2=idx+1;if(i2>AER3_NRH){i1=AER3_NRH-1;i2=AER3_NRH;}}}
 float a1=AER3_GT_RH_ARR[i1-1],a2=AER3_GT_RH_ARR[i2-1];float b=FDIV(a1,FSUB(100.0f,a1));
 float rf=max(0.0f,min(1.0f,FDIV(FSUB(FDIV(rh,FSUB(100.0f,rh)),b),FSUB(FDIV(a2,FSUB(100.0f,a2)),b))));
 float l1w=AER3_GT_LOOKUP[i1-1][ti-1][0],l2w=AER3_GT_LOOKUP[i2-1][ti-1][0],l1i=AER3_GT_LOOKUP[i1-1][ti-1][1],l2i=AER3_GT_LOOKUP[i2-1][ti-1][1];
 float rho=FDIV(p,FMUL(287.0f,t)),bw=FADD(l1w,FMUL(FSUB(l2w,l1w),rf)),bi=FADD(l1i,FMUL(FSUB(l2i,l1i),rf));float nw=max(1.0f,min(99999.E6f,nwfa)),ni=max(.01f,min(9999.E6f,nifa));
 return FADD(FMUL(FMUL(FMUL(bw,nw),dz),rho),FMUL(FMUL(FMUL(bi,ni),dz),rho));
}
static inline float aer3_rh(float p,float t,float qv){float tc=FSUB(t,273.15f),rv=max(0.0f,qv);float es=FMUL(6.112f,aer3_exp(FDIV(FMUL(17.6f,tc),FADD(tc,243.5f))));float e=FDIV(FMUL(FMUL(.01f,rv),p),FADD(rv,.62197f));return min(99.0f,max(0.0f,FDIV(FMUL(100.0f,e),es)));}
kernel void rrtmg_aer3_sw_optics(constant RrtmgAer3SwOpticsResources &r [[buffer(0)]],constant RrtmgAer3SwOpticsParams &q [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 long idx=long(tid),total=long(q.nc)*long(q.nlayers);if(idx>=total)return;int c=int(idx/q.nlayers),k=int(idx-long(c)*q.nlayers);ulong base=ulong(c)*14ul*ulong(q.nlayers)+ulong(k);
 if(k>=q.nz){for(int b=0;b<14;++b){r.ztaua[base+ulong(b)*ulong(q.nlayers)]=0;r.zasya[base+ulong(b)*ulong(q.nlayers)]=0;r.zomga[base+ulong(b)*ulong(q.nlayers)]=1;}return;}
 ulong ck=ulong(c)*ulong(q.nz)+ulong(k);float p=r.p3d[ck],t=r.t3d[ck],qv=r.qv3d[ck];float aod=aer3_gt_aod(p,r.dz8w[ck],t,qv,r.nwfa[ck],r.nifa[ck]);if(q.smoke_feedback)aod=FADD(aod,min(3.0f,r.smoke_aod[ck]));r.taod[ck]=aod;
 float rh=aer3_rh(p,t,qv);int ii=1;while(ii<=AER3_NRH&&rh>AER3_RHS[ii-1])++ii;int imin=max(1,ii-3),imax=min(AER3_NRH,ii+2);float lj[6];
 for(int jj=imin;jj<=imax;++jj){float l=1;for(int kk=imin;kk<=imax;++kk)if(kk!=jj)l=FDIV(FMUL(l,FSUB(rh,AER3_RHS[kk-1])),FSUB(AER3_RHS[jj-1],AER3_RHS[kk-1]));lj[jj-imin]=l;}
 for(int b=0;b<14;++b){float tau=0,ssa=0,asy=0;for(int jj=imin;jj<=imax;++jj){float l=lj[jj-imin];tau=FADD(tau,FMUL(FMUL(l,AER3_RAOD[jj-1][b]),aod));ssa=FADD(ssa,FMUL(l,AER3_SSA[jj-1][b]));asy=FADD(asy,FMUL(l,AER3_ASY[jj-1][b]));}r.ztaua[base+ulong(b)*ulong(q.nlayers)]=tau;r.zasya[base+ulong(b)*ulong(q.nlayers)]=asy;r.zomga[base+ulong(b)*ulong(q.nlayers)]=ssa;}
}
