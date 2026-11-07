#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_edmf_condensation_f32_bindings.metalh"
#pragma clang fp contract(off)
static inline float mynn_result(float a,float b,float value) {
    uint x=as_type<uint>(a),y=as_type<uint>(b),r=as_type<uint>(value);
    if ((r&0x7fffffffu)<=0x7f800000u) return value;
    if ((x&0x7fffffffu)>0x7f800000u) return as_type<float>(x|0x00400000u);
    if ((y&0x7fffffffu)>0x7f800000u) return as_type<float>(y|0x00400000u);
    return as_type<float>(0xffc00000u);
}
static inline float mynn_add(float a,float b){return mynn_result(a,b,mpas_add(a,b));}
static inline float mynn_sub(float a,float b){return mynn_result(a,b,mpas_sub(a,b));}
static inline float mynn_mul(float a,float b){return mynn_result(a,b,mpas_mul(a,b));}
static inline float mynn_div(float a,float b){return mynn_result(a,b,cu_div(a,b));}
#define mpas_add mynn_add
#define mpas_sub mynn_sub
#define mpas_mul mynn_mul
#define mpas_div mynn_div

static inline float mynn_neg(float a){return as_type<float>(as_type<uint>(a)^0x80000000u);}
static inline float mynn_min(float a,float b){return soft32_less(as_type<uint>(a),as_type<uint>(b))?a:b;}
static inline float mynn_max(float a,float b){return soft32_greater(as_type<uint>(a),as_type<uint>(b))?a:b;}
static inline float mynn_sqrt(float a){return mynn_result(a,a,cu_sqrt(a));}
static inline float mynn_square(float a){return mynn_mul(a,a);}


#include "soft_urban_trig.metalh"
static inline float mynn_pow(float a,float b){
    uint x=as_type<uint>(a),y=as_type<uint>(b);
    if ((x&0x7fffffffu)>0x7f800000u && (y&0x7fffffffu)==0x3f800000u)
        return as_type<float>((x&0x7fffffffu)|0x00400000u);
    bool sx=(x&0x7fc00000u)==0x7f800000u && (x&0x003fffffu)!=0u;
    bool sy=(y&0x7fc00000u)==0x7f800000u && (y&0x003fffffu)!=0u;
    if(sx||sy)return mynn_result(a,b,as_type<float>(0xffc00000u));
    return mynn_result(a,b,gfk_pow(a,b));
}
static inline float mynn_log(float a){return mynn_result(a,a,gfk_log(a));}
static inline float mynn_atan(float a){return mynn_result(a,a,as_type<float>(soft_urban_atanf(as_type<uint>(a))));}
constant float cphm_unst=16.f,cphh_unst=16.f;

#include "hexv842_mpas_bl_mynn_tanh.metalh"
static inline float mynn_tanh(float x){return mynn_result(x,x,as_type<float>(mynn_tanh_bits(as_type<uint>(x))));}

#include "hexv842_mpas_bl_mynn_log10.metalh"
static inline int mynn_max(int a,int b){return a>b?a:b;}
static inline ulong mynn_real64(float a){return soft64_from_f32_bits(as_type<uint>(a));}
static inline float mynn_real32(ulong a){return as_type<float>(soft64_to_f32_bits(a));}
static inline ulong mynn_sqrt(ulong a){ulong r=soft64_sqrt(a);return soft64_is_nan(r)&&!soft64_is_nan(a)?0xfff8000000000000ul:r;}
static inline float mynn_exp(float a){
uint bits=as_type<uint>(a),abstop=(bits>>20u)&0x7ffu;
if(abstop>=0x42bu){if(bits==0xff800000u)return 0.f;if(abstop>=0x7f8u)return mynn_result(a,a,as_type<float>(soft32_add(bits,bits)));if((bits&0x80000000u)==0u&&bits>0x42b17218u)return as_type<float>(0x7f800000u);if((bits&0x80000000u)!=0u&&(bits&0x7fffffffu)>0x42cff1b4u)return 0.f;}
ulong xd=soft64_from_f32_bits(bits),inv=0x40471547652b82feul,shift=0x4338000000000000ul;
ulong kd=soft64_sub(soft64_fma(inv,xd,shift),shift),r=soft64_fma(inv,xd,soft64_negate_bits(kd));
ulong scale=soft_gfk_exp2_core(kd,shift,0ul,0ul,0ul,0ul),z=soft64_fma(0x3ebc6af84b912394ul,r,0x3f2ebfce50fac4f3ul),r2=soft64_mul(r,r);
ulong y=soft64_fma(0x3f962e42ff0c52d6ul,r,0x3ff0000000000000ul);y=soft64_mul(soft64_fma(z,r2,y),scale);
return as_type<float>(soft64_to_f32_bits(y));}
static inline float mynn_log10(float a){return mynn_result(a,a,as_type<float>(mynn_log10_bits(as_type<uint>(a))));}
constant float j0 = .611583699e03f;
constant float j1 = .444606896e02f;
constant float j2 = .143177157e01f;
constant float j3 = .264224321e-1f;
constant float j4 = .299291081e-3f;
constant float j5 = .203154182e-5f;
constant float j6 = .702620698e-8f;
constant float j7 = .379534310e-11f;
constant float j8 = (-(.321582393e-13f));
constant float k0 = .609868993e03f;
constant float k1 = .499320233e02f;
constant float k2 = .184672631e01f;
constant float k3 = .402737184e-1f;
constant float k4 = .565392987e-3f;
constant float k5 = .521693933e-5f;
constant float k6 = .307839583e-7f;
constant float k7 = .105785160e-9f;
constant float k8 = .161444444e-12f;
static inline float f_qsat_blend(float t, float t0c, float tice, float p, int waterice) {
float qsat_blend;
float xc;
float esl;
float esi;
float rslf;
float rsif;
float chi;
// mynn_shared.F90:80

// mynn_shared.F90:86
xc = mynn_max(((float)(mynn_neg(80.f))), ((float)(mpas_sub(t, t0c))));
// mynn_shared.F90:88
if (soft32_greater_equal(as_type<uint>(t),as_type<uint>(t0c))) {
// mynn_shared.F90:89
esl = mpas_add(j0, mpas_mul(xc, mpas_add(j1, mpas_mul(xc, mpas_add(j2, mpas_mul(xc, mpas_add(j3, mpas_mul(xc, mpas_add(j4, mpas_mul(xc, mpas_add(j5, mpas_mul(xc, mpas_add(j6, mpas_mul(xc, mpas_add(j7, mpas_mul(xc, j8))))))))))))))));
// mynn_shared.F90:90
qsat_blend = mpas_div(mpas_mul(0.622f, esl), mpas_sub(p, esl));
// mynn_shared.F90:91
} else if (soft32_less_equal(as_type<uint>(t),as_type<uint>(tice))) {
// mynn_shared.F90:92
esi = mpas_add(k0, mpas_mul(xc, mpas_add(k1, mpas_mul(xc, mpas_add(k2, mpas_mul(xc, mpas_add(k3, mpas_mul(xc, mpas_add(k4, mpas_mul(xc, mpas_add(k5, mpas_mul(xc, mpas_add(k6, mpas_mul(xc, mpas_add(k7, mpas_mul(xc, k8))))))))))))))));
// mynn_shared.F90:93
qsat_blend = mpas_div(mpas_mul(0.622f, esi), mpas_sub(p, esi));
// mynn_shared.F90:94
} else {
// mynn_shared.F90:95
esl = mpas_add(j0, mpas_mul(xc, mpas_add(j1, mpas_mul(xc, mpas_add(j2, mpas_mul(xc, mpas_add(j3, mpas_mul(xc, mpas_add(j4, mpas_mul(xc, mpas_add(j5, mpas_mul(xc, mpas_add(j6, mpas_mul(xc, mpas_add(j7, mpas_mul(xc, j8))))))))))))))));
// mynn_shared.F90:96
esi = mpas_add(k0, mpas_mul(xc, mpas_add(k1, mpas_mul(xc, mpas_add(k2, mpas_mul(xc, mpas_add(k3, mpas_mul(xc, mpas_add(k4, mpas_mul(xc, mpas_add(k5, mpas_mul(xc, mpas_add(k6, mpas_mul(xc, mpas_add(k7, mpas_mul(xc, k8))))))))))))))));
// mynn_shared.F90:97
rslf = mpas_div(mpas_mul(0.622f, esl), mpas_sub(p, esl));
// mynn_shared.F90:98
rsif = mpas_div(mpas_mul(0.622f, esi), mpas_sub(p, esi));
// mynn_shared.F90:99
chi = mpas_div(mpas_sub(t0c, t), mpas_sub(t0c, tice));
// mynn_shared.F90:100
qsat_blend = mpas_add(mpas_mul(mpas_sub(1.f, chi), rslf), mpas_mul(chi, rsif));
// mynn_shared.F90:101
}
return qsat_blend;
}
static inline float f_xl_blend(float t, float t0c, float tice, float cice, float cliq, float cpv, float xls, float xlv) {
float xl_blend;
float xlvt;
float xlst;
float chi;
// mynn_shared.F90:118
if (soft32_greater_equal(as_type<uint>(t),as_type<uint>(t0c))) {
// mynn_shared.F90:119
xl_blend = mpas_add(xlv, mpas_mul(mpas_sub(cpv, cliq), mpas_sub(t, t0c)));
// mynn_shared.F90:120
} else if (soft32_less_equal(as_type<uint>(t),as_type<uint>(tice))) {
// mynn_shared.F90:121
xl_blend = mpas_add(xls, mpas_mul(mpas_sub(cpv, cice), mpas_sub(t, t0c)));
// mynn_shared.F90:122
} else {
// mynn_shared.F90:123
xlvt = mpas_add(xlv, mpas_mul(mpas_sub(cpv, cliq), mpas_sub(t, t0c)));
// mynn_shared.F90:124
xlst = mpas_add(xls, mpas_mul(mpas_sub(cpv, cice), mpas_sub(t, t0c)));
// mynn_shared.F90:125
chi = mpas_div(mpas_sub(t0c, t), mpas_sub(t0c, tice));
// mynn_shared.F90:126
xl_blend = mpas_add(mpas_mul(mpas_sub(1.f, chi), xlvt), mpas_mul(chi, xlst));
// mynn_shared.F90:127
}
return xl_blend;
}

static inline float mynn_abs(float a){return as_type<float>(as_type<uint>(a)&0x7fffffffu); }

kernel void hexv842_mpas_bl_mynn_edmf_condensation_f32(constant Hexv842MpasBlMynnEdmfCondensationF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnEdmfCondensationF32Params &cfg [[buffer(1)]],uint tid [[thread_position_in_grid]]){
if(tid>=uint(cfg.count))return;float qt=r.qt[tid],thl=r.thl[tid],p=r.p[tid],zagl=r.zagl[tid],qc=r.qc[tid],thv=r.thv[tid];float p1000mb=cfg.p1000mb,rcp=cfg.rcp,xlvcp=cfg.xlvcp,t0c=cfg.t0c,rvovrd=cfg.rvovrd;constexpr float tice=240.f;

int niter;
int i;
float diff;
float exn;
float t;
float th;
float qs;
float qcold;
// bl_mynn_subroutines.F90:5808
niter = 50;
// bl_mynn_subroutines.F90:5810
diff = 1.e-6f;
// bl_mynn_subroutines.F90:5812
exn = mynn_pow(((float)(mpas_div(p, p1000mb))), ((float)(rcp)));
// bl_mynn_subroutines.F90:5814
for (i=1; (1) > 0 ? i <= (niter) : i >= (niter); i += (1)) {
// bl_mynn_subroutines.F90:5815
t = mpas_add(mpas_mul(exn, thl), mpas_mul(xlvcp, qc));
// bl_mynn_subroutines.F90:5816
qs = f_qsat_blend(t, t0c, tice, p, 0);
// bl_mynn_subroutines.F90:5817
qcold = qc;
// bl_mynn_subroutines.F90:5818
qc = mpas_add(mpas_mul(0.5f, qc), mpas_mul(0.5f, mynn_max(((float)(mpas_sub(qt, qs))), ((float)(0.f)))));
// bl_mynn_subroutines.F90:5819
if (soft32_less(as_type<uint>(mynn_abs(mpas_sub(qc,qcold))),as_type<uint>(diff))) { break; }
// bl_mynn_subroutines.F90:5820
}
// bl_mynn_subroutines.F90:5822
t = mpas_add(mpas_mul(exn, thl), mpas_mul(xlvcp, qc));
// bl_mynn_subroutines.F90:5823
qs = f_qsat_blend(t, t0c, tice, p, 0);
// bl_mynn_subroutines.F90:5824
qc = mynn_max(((float)(mpas_sub(qt, qs))), ((float)(0.f)));
// bl_mynn_subroutines.F90:5827
if (soft32_less(as_type<uint>(zagl),as_type<uint>(100.f))) { qc = 0.f; }
// bl_mynn_subroutines.F90:5830
thv = mpas_mul(mpas_add(thl, mpas_mul(xlvcp, qc)), mpas_sub(mpas_add(1.f, mpas_mul(qt, mpas_sub(rvovrd, 1.f))), mpas_mul(rvovrd, qc)));

r.qc[tid]=qc;r.thv[tid]=thv;
}
