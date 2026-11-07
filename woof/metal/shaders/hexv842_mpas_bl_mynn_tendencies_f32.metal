#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_tendencies_f32_bindings.metalh"
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
#include "hexv842_mpas_bl_mynn_tendencies_pack0_f32_bindings.metalh"
#include "hexv842_mpas_bl_mynn_tendencies_pack1_f32_bindings.metalh"
#include "hexv842_mpas_bl_mynn_tendencies_scatter_f32_bindings.metalh"

kernel void hexv842_mpas_bl_mynn_tendencies_f32(constant Hexv842MpasBlMynnTendenciesF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnTendenciesF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;constexpr float tice=240.f;
float psfc=r.psfc[col];
float ust=r.ust[col];
float flt=r.flt[col];
float flq=r.flq[col];
float flqv=r.flqv[col];
float flqc=r.flqc[col];
float wspd=r.wspd[col];
float uoce=r.uoce[col];
float voce=r.voce[col];
float delt=cfg.delt;
float r_d=cfg.r_d;
float p608=cfg.p608;
float xlvcp=cfg.xlvcp;
float xlscp=cfg.xlscp;
float t0c=cfg.t0c;
bool flag_qc=cfg.flag_qc!=0;
bool flag_qi=cfg.flag_qi!=0;
bool flag_qnc=cfg.flag_qnc!=0;
bool flag_qni=cfg.flag_qni!=0;
bool flag_qs=cfg.flag_qs!=0;
bool flag_qnwfa=cfg.flag_qnwfa!=0;
bool flag_qnifa=cfg.flag_qnifa!=0;
bool flag_qnbca=cfg.flag_qnbca!=0;
bool flag_ozone=cfg.flag_ozone!=0;
bool bl_mynn_cloudmix=cfg.bl_mynn_cloudmix!=0;
bool bl_mynn_mixqt=cfg.bl_mynn_mixqt!=0;
bool bl_mynn_edmf_mom=cfg.bl_mynn_edmf_mom!=0;
bool bl_mynn_mixscalars=cfg.bl_mynn_mixscalars!=0;
#define dz(i) r.packed[(ulong(0)*cfg.nc+col)*nl+((i)-1)]
#define rho(i) r.packed[(ulong(1)*cfg.nc+col)*nl+((i)-1)]
#define u(i) r.packed[(ulong(2)*cfg.nc+col)*nl+((i)-1)]
#define v(i) r.packed[(ulong(3)*cfg.nc+col)*nl+((i)-1)]
#define th(i) r.packed[(ulong(4)*cfg.nc+col)*nl+((i)-1)]
#define tk(i) r.packed[(ulong(5)*cfg.nc+col)*nl+((i)-1)]
#define qv(i) r.packed[(ulong(6)*cfg.nc+col)*nl+((i)-1)]
#define qc(i) r.packed[(ulong(7)*cfg.nc+col)*nl+((i)-1)]
#define qi(i) r.packed[(ulong(8)*cfg.nc+col)*nl+((i)-1)]
#define qs(i) r.packed[(ulong(9)*cfg.nc+col)*nl+((i)-1)]
#define qnc(i) r.packed[(ulong(10)*cfg.nc+col)*nl+((i)-1)]
#define qni(i) r.packed[(ulong(11)*cfg.nc+col)*nl+((i)-1)]
#define p(i) r.packed[(ulong(12)*cfg.nc+col)*nl+((i)-1)]
#define exner(i) r.packed[(ulong(13)*cfg.nc+col)*nl+((i)-1)]
#define thl(i) r.packed[(ulong(14)*cfg.nc+col)*nl+((i)-1)]
#define sqv(i) r.packed[(ulong(15)*cfg.nc+col)*nl+((i)-1)]
#define sqc(i) r.packed[(ulong(16)*cfg.nc+col)*nl+((i)-1)]
#define sqi(i) r.packed[(ulong(17)*cfg.nc+col)*nl+((i)-1)]
#define sqs(i) r.packed[(ulong(18)*cfg.nc+col)*nl+((i)-1)]
#define sqw(i) r.packed[(ulong(19)*cfg.nc+col)*nl+((i)-1)]
#define qnwfa(i) r.packed[(ulong(20)*cfg.nc+col)*nl+((i)-1)]
#define qnifa(i) r.packed[(ulong(21)*cfg.nc+col)*nl+((i)-1)]
#define qnbca(i) r.packed[(ulong(22)*cfg.nc+col)*nl+((i)-1)]
#define ozone(i) r.packed[(ulong(23)*cfg.nc+col)*nl+((i)-1)]
#define tsq(i) r.packed[(ulong(24)*cfg.nc+col)*nl+((i)-1)]
#define qsq(i) r.packed[(ulong(25)*cfg.nc+col)*nl+((i)-1)]
#define cov(i) r.packed[(ulong(26)*cfg.nc+col)*nl+((i)-1)]
#define tcd(i) r.packed[(ulong(27)*cfg.nc+col)*nl+((i)-1)]
#define qcd(i) r.packed[(ulong(28)*cfg.nc+col)*nl+((i)-1)]
#define dfm(i) r.packed[(ulong(29)*cfg.nc+col)*nl+((i)-1)]
#define dfh(i) r.packed[(ulong(30)*cfg.nc+col)*nl+((i)-1)]
#define dfq(i) r.packed[(ulong(31)*cfg.nc+col)*nl+((i)-1)]
#define du(i) r.packed[(ulong(32)*cfg.nc+col)*nl+((i)-1)]
#define dv(i) r.packed[(ulong(33)*cfg.nc+col)*nl+((i)-1)]
#define dth(i) r.packed[(ulong(34)*cfg.nc+col)*nl+((i)-1)]
#define dqv(i) r.packed[(ulong(35)*cfg.nc+col)*nl+((i)-1)]
#define dqc(i) r.packed[(ulong(36)*cfg.nc+col)*nl+((i)-1)]
#define dqi(i) r.packed[(ulong(37)*cfg.nc+col)*nl+((i)-1)]
#define dqs(i) r.packed[(ulong(38)*cfg.nc+col)*nl+((i)-1)]
#define dqnc(i) r.packed[(ulong(39)*cfg.nc+col)*nl+((i)-1)]
#define dqni(i) r.packed[(ulong(40)*cfg.nc+col)*nl+((i)-1)]
#define dqnwfa(i) r.packed[(ulong(41)*cfg.nc+col)*nl+((i)-1)]
#define dqnifa(i) r.packed[(ulong(42)*cfg.nc+col)*nl+((i)-1)]
#define dqnbca(i) r.packed[(ulong(43)*cfg.nc+col)*nl+((i)-1)]
#define dozone(i) r.packed[(ulong(44)*cfg.nc+col)*nl+((i)-1)]
#define diss_heat(i) r.packed[(ulong(45)*cfg.nc+col)*nl+((i)-1)]
#define s_aw(i) r.packed[(ulong(46)*cfg.nc+col)*nl+((i)-1)]
#define s_awthl(i) r.packed[(ulong(47)*cfg.nc+col)*nl+((i)-1)]
#define s_awqt(i) r.packed[(ulong(48)*cfg.nc+col)*nl+((i)-1)]
#define s_awqv(i) r.packed[(ulong(49)*cfg.nc+col)*nl+((i)-1)]
#define s_awqc(i) r.packed[(ulong(50)*cfg.nc+col)*nl+((i)-1)]
#define s_awu(i) r.packed[(ulong(51)*cfg.nc+col)*nl+((i)-1)]
#define s_awv(i) r.packed[(ulong(52)*cfg.nc+col)*nl+((i)-1)]
#define s_awqnc(i) r.packed[(ulong(53)*cfg.nc+col)*nl+((i)-1)]
#define s_awqni(i) r.packed[(ulong(54)*cfg.nc+col)*nl+((i)-1)]
#define s_awqnwfa(i) r.packed[(ulong(55)*cfg.nc+col)*nl+((i)-1)]
#define s_awqnifa(i) r.packed[(ulong(56)*cfg.nc+col)*nl+((i)-1)]
#define s_awqnbca(i) r.packed[(ulong(57)*cfg.nc+col)*nl+((i)-1)]
#define sd_aw(i) r.packed[(ulong(58)*cfg.nc+col)*nl+((i)-1)]
#define sd_awthl(i) r.packed[(ulong(59)*cfg.nc+col)*nl+((i)-1)]
#define sd_awqt(i) r.packed[(ulong(60)*cfg.nc+col)*nl+((i)-1)]
#define sd_awqv(i) r.packed[(ulong(61)*cfg.nc+col)*nl+((i)-1)]
#define sd_awqc(i) r.packed[(ulong(62)*cfg.nc+col)*nl+((i)-1)]
#define sd_awu(i) r.packed[(ulong(63)*cfg.nc+col)*nl+((i)-1)]
#define sd_awv(i) r.packed[(ulong(64)*cfg.nc+col)*nl+((i)-1)]
#define sub_thl(i) r.packed[(ulong(65)*cfg.nc+col)*nl+((i)-1)]
#define sub_sqv(i) r.packed[(ulong(66)*cfg.nc+col)*nl+((i)-1)]
#define sub_u(i) r.packed[(ulong(67)*cfg.nc+col)*nl+((i)-1)]
#define sub_v(i) r.packed[(ulong(68)*cfg.nc+col)*nl+((i)-1)]
#define det_thl(i) r.packed[(ulong(69)*cfg.nc+col)*nl+((i)-1)]
#define det_sqv(i) r.packed[(ulong(70)*cfg.nc+col)*nl+((i)-1)]
#define det_sqc(i) r.packed[(ulong(71)*cfg.nc+col)*nl+((i)-1)]
#define det_u(i) r.packed[(ulong(72)*cfg.nc+col)*nl+((i)-1)]
#define det_v(i) r.packed[(ulong(73)*cfg.nc+col)*nl+((i)-1)]
#define cldfra_bl1d(i) r.packed[(ulong(74)*cfg.nc+col)*nl+((i)-1)]
#define dtz(i) r.scratch[(ulong(col)*28+0)*(nl+1)+((i)-1)]
#define dfhc(i) r.scratch[(ulong(col)*28+1)*(nl+1)+((i)-1)]
#define dfmc(i) r.scratch[(ulong(col)*28+2)*(nl+1)+((i)-1)]
#define delp(i) r.scratch[(ulong(col)*28+3)*(nl+1)+((i)-1)]
#define sqv2(i) r.scratch[(ulong(col)*28+4)*(nl+1)+((i)-1)]
#define sqc2(i) r.scratch[(ulong(col)*28+5)*(nl+1)+((i)-1)]
#define sqi2(i) r.scratch[(ulong(col)*28+6)*(nl+1)+((i)-1)]
#define sqs2(i) r.scratch[(ulong(col)*28+7)*(nl+1)+((i)-1)]
#define sqw2(i) r.scratch[(ulong(col)*28+8)*(nl+1)+((i)-1)]
#define qni2(i) r.scratch[(ulong(col)*28+9)*(nl+1)+((i)-1)]
#define qnc2(i) r.scratch[(ulong(col)*28+10)*(nl+1)+((i)-1)]
#define qnwfa2(i) r.scratch[(ulong(col)*28+11)*(nl+1)+((i)-1)]
#define qnifa2(i) r.scratch[(ulong(col)*28+12)*(nl+1)+((i)-1)]
#define qnbca2(i) r.scratch[(ulong(col)*28+13)*(nl+1)+((i)-1)]
#define ozone2(i) r.scratch[(ulong(col)*28+14)*(nl+1)+((i)-1)]
#define zfac(i) r.scratch[(ulong(col)*28+15)*(nl+1)+((i)-1)]
#define plumekh(i) r.scratch[(ulong(col)*28+16)*(nl+1)+((i)-1)]
#define rhoinv(i) r.scratch[(ulong(col)*28+17)*(nl+1)+((i)-1)]
#define a(i) r.scratch[(ulong(col)*28+18)*(nl+1)+((i)-1)]
#define b(i) r.scratch[(ulong(col)*28+19)*(nl+1)+((i)-1)]
#define c(i) r.scratch[(ulong(col)*28+20)*(nl+1)+((i)-1)]
#define d(i) r.scratch[(ulong(col)*28+21)*(nl+1)+((i)-1)]
#define x(i) r.scratch[(ulong(col)*28+22)*(nl+1)+((i)-1)]
#define rhoz(i) r.scratch[(ulong(col)*28+23)*(nl+1)+((i)-1)]
#define khdz(i) r.scratch[(ulong(col)*28+24)*(nl+1)+((i)-1)]
#define kmdz(i) r.scratch[(ulong(col)*28+25)*(nl+1)+((i)-1)]
#define cp(i) r.scratch[(ulong(col)*28+26)*(nl+1)+((i)-1)]
#define dp(i) r.scratch[(ulong(col)*28+27)*(nl+1)+((i)-1)]

float wsp;
float wsp2;
float tk2;
float th2;
bool problem;
int kproblem;
float rhs;
float gfluxm;
float gfluxp;
float dztop;
float maxdfh;
float mindfh;
float maxcf;
float maxkh;
float zw;
float t;
float esat;
float qsl;
float onoff;
float kh;
float km;
float dzk;
float rhosfc;
float ustdrag;
float ustdiff;
float qvflux;
float th_new;
float portion_qc;
float portion_qi;
float condensate;
float qsat;
int k;
int kk;
constexpr float nonloc = 1.0f;
// bl_mynn_subroutines.F90:3083
dztop = mpas_mul(.5f, mpas_add(dz(kte), dz(((kte) - (1)))));
// bl_mynn_subroutines.F90:3088
if ((!(bl_mynn_edmf_mom))) {
// bl_mynn_subroutines.F90:3089
onoff = 0.0f;
// bl_mynn_subroutines.F90:3090
} else {
// bl_mynn_subroutines.F90:3091
onoff = 1.0f;
// bl_mynn_subroutines.F90:3092
}
// bl_mynn_subroutines.F90:3096
rhosfc = mpas_div(psfc, mpas_mul(r_d, mpas_add(tk(kts), mpas_mul(p608, qv(kts)))));
// bl_mynn_subroutines.F90:3097
dtz(kts) = mpas_div(delt, dz(kts));
// bl_mynn_subroutines.F90:3098
rhoz(kts) = rho(kts);
// bl_mynn_subroutines.F90:3099
rhoinv(kts) = mpas_div(1.f, rho(kts));
// bl_mynn_subroutines.F90:3100
khdz(kts) = mpas_mul(rhoz(kts), dfh(kts));
// bl_mynn_subroutines.F90:3101
kmdz(kts) = mpas_mul(rhoz(kts), dfm(kts));
// bl_mynn_subroutines.F90:3102
delp(kts) = mpas_sub(psfc, mpas_div(mpas_add(mpas_mul(p(((kts) + (1))), dz(kts)), mpas_mul(p(kts), dz(((kts) + (1))))), mpas_add(dz(kts), dz(((kts) + (1))))));
// bl_mynn_subroutines.F90:3103
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3104
dtz(k) = mpas_div(delt, dz(k));
// bl_mynn_subroutines.F90:3105
rhoz(k) = mpas_div(mpas_add(mpas_mul(rho(k), dz(((k) - (1)))), mpas_mul(rho(((k) - (1))), dz(k))), mpas_add(dz(((k) - (1))), dz(k)));
// bl_mynn_subroutines.F90:3106
rhoz(k) = mynn_max(((float)(rhoz(k))), ((float)(1e-4f)));
// bl_mynn_subroutines.F90:3107
rhoinv(k) = mpas_div(1.f, mynn_max(((float)(rho(k))), ((float)(1e-4f))));
// bl_mynn_subroutines.F90:3108
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:3109
khdz(k) = mpas_mul(rhoz(k), dfh(k));
// bl_mynn_subroutines.F90:3110
kmdz(k) = mpas_mul(rhoz(k), dfm(k));
// bl_mynn_subroutines.F90:3111
}
// bl_mynn_subroutines.F90:3112
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3113
delp(k) = mpas_sub(mpas_div(mpas_add(mpas_mul(p(k), dz(((k) - (1)))), mpas_mul(p(((k) - (1))), dz(k))), mpas_add(dz(k), dz(((k) - (1))))), mpas_div(mpas_add(mpas_mul(p(((k) + (1))), dz(k)), mpas_mul(p(k), dz(((k) + (1))))), mpas_add(dz(k), dz(((k) + (1))))));
// bl_mynn_subroutines.F90:3115
}
// bl_mynn_subroutines.F90:3116
delp(kte) = delp(((kte) - (1)));
// bl_mynn_subroutines.F90:3117
rhoz(((kte) + (1))) = rhoz(kte);
// bl_mynn_subroutines.F90:3118
khdz(((kte) + (1))) = mpas_mul(rhoz(((kte) + (1))), dfh(kte));
// bl_mynn_subroutines.F90:3119
kmdz(((kte) + (1))) = mpas_mul(rhoz(((kte) + (1))), dfm(kte));
// bl_mynn_subroutines.F90:3122
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3123
khdz(k) = mynn_max(((float)(khdz(k))), ((float)(mpas_mul(0.5f, s_aw(k)))));
// bl_mynn_subroutines.F90:3124
khdz(k) = mynn_max(((float)(khdz(k))), ((float)(mpas_mul(mynn_neg(0.5f), mpas_sub(s_aw(k), s_aw(((k) + (1))))))));
// bl_mynn_subroutines.F90:3125
kmdz(k) = mynn_max(((float)(kmdz(k))), ((float)(mpas_mul(0.5f, s_aw(k)))));
// bl_mynn_subroutines.F90:3126
kmdz(k) = mynn_max(((float)(kmdz(k))), ((float)(mpas_mul(mynn_neg(0.5f), mpas_sub(s_aw(k), s_aw(((k) + (1))))))));
// bl_mynn_subroutines.F90:3127
}
// bl_mynn_subroutines.F90:3129
ustdrag = mpas_div(mynn_min(((float)(mpas_mul(ust, ust))), ((float)(0.99f))), wspd);
// bl_mynn_subroutines.F90:3130
ustdiff = mpas_div(mynn_min(((float)(mpas_mul(ust, ust))), ((float)(0.01f))), wspd);
// bl_mynn_subroutines.F90:3131
for(int i=1;i<=nl;i++) { dth(i)=0.0f; }
// bl_mynn_subroutines.F90:3137
k = kts;
// bl_mynn_subroutines.F90:3140
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3141
b(k) = mpas_sub(mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(((k) + (1))), mpas_div(mpas_mul(rhosfc, mynn_square(ust)), wspd))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:3144
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:3147
d(k) = mpas_add(mpas_add(mpas_add(mpas_sub(mpas_add(u(k), mpas_div(mpas_mul(mpas_mul(dtz(k), uoce), mynn_square(ust)), wspd)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awu(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), sd_awu(((k) + (1)))), onoff)), mpas_mul(sub_u(k), delt)), mpas_mul(det_u(k), delt));
// bl_mynn_subroutines.F90:3152
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3153
a(k) = mpas_add(mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(k)), onoff));
// bl_mynn_subroutines.F90:3156
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(sd_aw(k), sd_aw(((k) + (1))))), onoff));
// bl_mynn_subroutines.F90:3159
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:3162
d(k) = mpas_add(mpas_add(mpas_sub(mpas_add(u(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awu(k), s_awu(((k) + (1))))), onoff)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(sd_awu(k), sd_awu(((k) + (1))))), onoff)), mpas_mul(sub_u(k), delt)), mpas_mul(det_u(k), delt));
// bl_mynn_subroutines.F90:3165
}
// bl_mynn_subroutines.F90:3180
a(kte) = 0;
// bl_mynn_subroutines.F90:3181
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3182
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3183
d(kte) = u(kte);
// bl_mynn_subroutines.F90:3186
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3189
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3191
du(k) = mpas_div(mpas_sub(x(k), u(k)), delt);
// bl_mynn_subroutines.F90:3192
}
// bl_mynn_subroutines.F90:3198
k = kts;
// bl_mynn_subroutines.F90:3201
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3202
b(k) = mpas_sub(mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(((k) + (1))), mpas_div(mpas_mul(rhosfc, mynn_square(ust)), wspd))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:3205
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:3208
d(k) = mpas_add(mpas_add(mpas_add(mpas_sub(mpas_add(v(k), mpas_div(mpas_mul(mpas_mul(dtz(k), voce), mynn_square(ust)), wspd)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awv(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), sd_awv(((k) + (1)))), onoff)), mpas_mul(sub_v(k), delt)), mpas_mul(det_v(k), delt));
// bl_mynn_subroutines.F90:3213
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3214
a(k) = mpas_add(mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(k)), onoff));
// bl_mynn_subroutines.F90:3217
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(sd_aw(k), sd_aw(((k) + (1))))), onoff));
// bl_mynn_subroutines.F90:3220
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:3223
d(k) = mpas_add(mpas_add(mpas_sub(mpas_add(v(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awv(k), s_awv(((k) + (1))))), onoff)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(sd_awv(k), sd_awv(((k) + (1))))), onoff)), mpas_mul(sub_v(k), delt)), mpas_mul(det_v(k), delt));
// bl_mynn_subroutines.F90:3226
}
// bl_mynn_subroutines.F90:3241
a(kte) = 0;
// bl_mynn_subroutines.F90:3242
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3243
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3244
d(kte) = v(kte);
// bl_mynn_subroutines.F90:3247
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3250
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3252
dv(k) = mpas_div(mpas_sub(x(k), v(k)), delt);
// bl_mynn_subroutines.F90:3253
}
// bl_mynn_subroutines.F90:3258
k = kts;
// bl_mynn_subroutines.F90:3277
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3278
b(k) = mpas_sub(mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3279
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3280
d(k) = mpas_add(mpas_add(mpas_add(mpas_sub(mpas_sub(mpas_add(mpas_add(thl(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhosfc), flt), rhoinv(k))), mpas_mul(tcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awthl(((k) + (1))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), sd_awthl(((k) + (1))))), mpas_mul(diss_heat(k), delt)), mpas_mul(sub_thl(k), delt)), mpas_mul(det_thl(k), delt));
// bl_mynn_subroutines.F90:3284
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3285
a(k) = mpas_add(mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(k)));
// bl_mynn_subroutines.F90:3286
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1)))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(sd_aw(k), sd_aw(((k) + (1))))));
// bl_mynn_subroutines.F90:3288
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3289
d(k) = mpas_add(mpas_add(mpas_add(mpas_add(mpas_add(mpas_add(thl(k), mpas_mul(tcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awthl(k), s_awthl(((k) + (1)))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(sd_awthl(k), sd_awthl(((k) + (1)))))), mpas_mul(diss_heat(k), delt)), mpas_mul(sub_thl(k), delt)), mpas_mul(det_thl(k), delt));
// bl_mynn_subroutines.F90:3293
}
// bl_mynn_subroutines.F90:3309
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3310
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3311
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3312
d(kte) = thl(kte);
// bl_mynn_subroutines.F90:3315
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3318
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3320
thl(k) = x(k);
// bl_mynn_subroutines.F90:3321
}
// bl_mynn_subroutines.F90:3323
if (bl_mynn_mixqt) {
// bl_mynn_subroutines.F90:3331
k = kts;
// bl_mynn_subroutines.F90:3347
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3348
b(k) = mpas_sub(mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3349
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3350
d(k) = mpas_sub(mpas_sub(mpas_add(mpas_add(sqw(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhosfc), flq), rhoinv(k))), mpas_mul(qcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqt(((k) + (1))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), sd_awqt(((k) + (1)))));
// bl_mynn_subroutines.F90:3352
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3353
a(k) = mpas_add(mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(k)));
// bl_mynn_subroutines.F90:3354
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1)))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(sd_aw(k), sd_aw(((k) + (1))))));
// bl_mynn_subroutines.F90:3356
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3357
d(k) = mpas_add(mpas_add(mpas_add(sqw(k), mpas_mul(qcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqt(k), s_awqt(((k) + (1)))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(sd_awqt(k), sd_awqt(((k) + (1))))));
// bl_mynn_subroutines.F90:3358
}
// bl_mynn_subroutines.F90:3372
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3373
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3374
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3375
d(kte) = sqw(kte);
// bl_mynn_subroutines.F90:3378
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
sqw2(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
sqw2(i) = mpas_sub(dp(i), mpas_mul(cp(i), sqw2(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3384
} else {
// bl_mynn_subroutines.F90:3385
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { sqw2(mynn_element+1) = sqw(mynn_element+1); }
// bl_mynn_subroutines.F90:3386
}
// bl_mynn_subroutines.F90:3388
if ((!(bl_mynn_mixqt))) {
// bl_mynn_subroutines.F90:3393
if (((bl_mynn_cloudmix) && (flag_qc))) {
// bl_mynn_subroutines.F90:3395
k = kts;
// bl_mynn_subroutines.F90:3412
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3413
b(k) = mpas_sub(mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3414
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3415
d(k) = mpas_add(mpas_sub(mpas_sub(mpas_add(mpas_add(sqc(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhosfc), flqc), rhoinv(k))), mpas_mul(qcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqc(((k) + (1))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), sd_awqc(((k) + (1))))), mpas_mul(det_sqc(k), delt));
// bl_mynn_subroutines.F90:3419
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3420
a(k) = mpas_add(mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(k)));
// bl_mynn_subroutines.F90:3421
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1)))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(sd_aw(k), sd_aw(((k) + (1))))));
// bl_mynn_subroutines.F90:3423
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3424
d(k) = mpas_add(mpas_add(mpas_add(mpas_add(sqc(k), mpas_mul(qcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqc(k), s_awqc(((k) + (1)))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(sd_awqc(k), sd_awqc(((k) + (1)))))), mpas_mul(det_sqc(k), delt));
// bl_mynn_subroutines.F90:3426
}
// bl_mynn_subroutines.F90:3429
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3430
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3431
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3432
d(kte) = sqc(kte);
// bl_mynn_subroutines.F90:3435
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
sqc2(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
sqc2(i) = mpas_sub(dp(i), mpas_mul(cp(i), sqc2(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3441
} else {
// bl_mynn_subroutines.F90:3443
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { sqc2(mynn_element+1) = sqc(mynn_element+1); }
// bl_mynn_subroutines.F90:3444
}
// bl_mynn_subroutines.F90:3445
}
// bl_mynn_subroutines.F90:3447
if ((!(bl_mynn_mixqt))) {
// bl_mynn_subroutines.F90:3453
k = kts;
// bl_mynn_subroutines.F90:3470
qvflux = flqv;
// bl_mynn_subroutines.F90:3471
if (soft32_less(as_type<uint>(qvflux),0u)) {
// bl_mynn_subroutines.F90:3473
qvflux = mynn_max(((float)(qvflux)), ((float)(mpas_div(mynn_min(((float)(mpas_sub(mpas_mul(0.9f, sqv(kts)), 1e-8f))), ((float)(0.0f))), dtz(kts)))));
// bl_mynn_subroutines.F90:3474
}
// bl_mynn_subroutines.F90:3477
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3478
b(k) = mpas_sub(mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3479
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3480
d(k) = mpas_add(mpas_add(mpas_sub(mpas_sub(mpas_add(mpas_add(sqv(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhosfc), qvflux), rhoinv(k))), mpas_mul(qcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqv(((k) + (1))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), sd_awqv(((k) + (1))))), mpas_mul(sub_sqv(k), delt)), mpas_mul(det_sqv(k), delt));
// bl_mynn_subroutines.F90:3484
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3485
a(k) = mpas_add(mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(k)));
// bl_mynn_subroutines.F90:3486
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1)))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(sd_aw(k), sd_aw(((k) + (1))))));
// bl_mynn_subroutines.F90:3488
c(k) = mpas_sub(mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), sd_aw(((k) + (1)))));
// bl_mynn_subroutines.F90:3489
d(k) = mpas_add(mpas_add(mpas_add(mpas_add(mpas_add(sqv(k), mpas_mul(qcd(k), delt)), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqv(k), s_awqv(((k) + (1)))))), mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(sd_awqv(k), sd_awqv(((k) + (1)))))), mpas_mul(sub_sqv(k), delt)), mpas_mul(det_sqv(k), delt));
// bl_mynn_subroutines.F90:3491
}
// bl_mynn_subroutines.F90:3507
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3508
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3509
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3510
d(kte) = sqv(kte);
// bl_mynn_subroutines.F90:3513
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
sqv2(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
sqv2(i) = mpas_sub(dp(i), mpas_mul(cp(i), sqv2(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3519
} else {
// bl_mynn_subroutines.F90:3520
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { sqv2(mynn_element+1) = sqv(mynn_element+1); }
// bl_mynn_subroutines.F90:3521
}
// bl_mynn_subroutines.F90:3526
if (((bl_mynn_cloudmix) && (flag_qi))) {
// bl_mynn_subroutines.F90:3528
k = kts;
// bl_mynn_subroutines.F90:3530
a(k) = mynn_neg(mpas_mul(mpas_mul(dtz(k), khdz(k)), rhoinv(k)));
// bl_mynn_subroutines.F90:3531
b(k) = mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k)));
// bl_mynn_subroutines.F90:3532
c(k) = mynn_neg(mpas_mul(mpas_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// bl_mynn_subroutines.F90:3533
d(k) = sqi(k);
// bl_mynn_subroutines.F90:3535
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3536
a(k) = mynn_neg(mpas_mul(mpas_mul(dtz(k), khdz(k)), rhoinv(k)));
// bl_mynn_subroutines.F90:3537
b(k) = mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// bl_mynn_subroutines.F90:3538
c(k) = mynn_neg(mpas_mul(mpas_mul(dtz(k), khdz(((k) + (1)))), rhoinv(k)));
// bl_mynn_subroutines.F90:3539
d(k) = sqi(k);
// bl_mynn_subroutines.F90:3540
}
// bl_mynn_subroutines.F90:3556
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3557
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3558
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3559
d(kte) = sqi(kte);
// bl_mynn_subroutines.F90:3562
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
sqi2(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
sqi2(i) = mpas_sub(dp(i), mpas_mul(cp(i), sqi2(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3568
} else {
// bl_mynn_subroutines.F90:3569
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { sqi2(mynn_element+1) = sqi(mynn_element+1); }
// bl_mynn_subroutines.F90:3570
}
// bl_mynn_subroutines.F90:3576
if (((bl_mynn_cloudmix) && (false))) {
// bl_mynn_subroutines.F90:3578
k = kts;
// bl_mynn_subroutines.F90:3580
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3581
b(k) = mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k)));
// bl_mynn_subroutines.F90:3582
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:3583
d(k) = sqs(k);
// bl_mynn_subroutines.F90:3585
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3586
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3587
b(k) = mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// bl_mynn_subroutines.F90:3588
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:3589
d(k) = sqs(k);
// bl_mynn_subroutines.F90:3590
}
// bl_mynn_subroutines.F90:3593
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3594
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3595
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3596
d(kte) = sqs(kte);
// bl_mynn_subroutines.F90:3599
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
sqs2(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
sqs2(i) = mpas_sub(dp(i), mpas_mul(cp(i), sqs2(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3605
} else {
// bl_mynn_subroutines.F90:3606
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { sqs2(mynn_element+1) = sqs(mynn_element+1); }
// bl_mynn_subroutines.F90:3607
}
// bl_mynn_subroutines.F90:3612
if (((((bl_mynn_cloudmix) && (flag_qni))) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3615
k = kts;
// bl_mynn_subroutines.F90:3617
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3618
b(k) = mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3619
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3620
d(k) = mpas_sub(qni(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqni(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3622
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3623
a(k) = mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), nonloc));
// bl_mynn_subroutines.F90:3624
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3626
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3627
d(k) = mpas_add(qni(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqni(k), s_awqni(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3628
}
// bl_mynn_subroutines.F90:3631
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3632
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3633
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3634
d(kte) = qni(kte);
// bl_mynn_subroutines.F90:3637
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3640
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3642
qni2(k) = x(k);
// bl_mynn_subroutines.F90:3643
}
// bl_mynn_subroutines.F90:3645
} else {
// bl_mynn_subroutines.F90:3646
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { qni2(mynn_element+1) = qni(mynn_element+1); }
// bl_mynn_subroutines.F90:3647
}
// bl_mynn_subroutines.F90:3653
if (((((bl_mynn_cloudmix) && (flag_qnc))) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3656
k = kts;
// bl_mynn_subroutines.F90:3658
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3659
b(k) = mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3660
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3661
d(k) = mpas_sub(qnc(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqnc(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3663
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3664
a(k) = mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), nonloc));
// bl_mynn_subroutines.F90:3665
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3667
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3668
d(k) = mpas_add(qnc(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqnc(k), s_awqnc(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3669
}
// bl_mynn_subroutines.F90:3672
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3673
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3674
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3675
d(kte) = qnc(kte);
// bl_mynn_subroutines.F90:3678
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3681
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3683
qnc2(k) = x(k);
// bl_mynn_subroutines.F90:3684
}
// bl_mynn_subroutines.F90:3686
} else {
// bl_mynn_subroutines.F90:3687
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { qnc2(mynn_element+1) = qnc(mynn_element+1); }
// bl_mynn_subroutines.F90:3688
}
// bl_mynn_subroutines.F90:3693
if (((((bl_mynn_cloudmix) && (flag_qnwfa))) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3696
k = kts;
// bl_mynn_subroutines.F90:3698
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3699
b(k) = mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3701
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3702
d(k) = mpas_sub(qnwfa(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqnwfa(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3704
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3705
a(k) = mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), nonloc));
// bl_mynn_subroutines.F90:3706
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3708
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3709
d(k) = mpas_add(qnwfa(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqnwfa(k), s_awqnwfa(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3710
}
// bl_mynn_subroutines.F90:3713
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3714
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3715
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3716
d(kte) = qnwfa(kte);
// bl_mynn_subroutines.F90:3719
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3722
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3724
qnwfa2(k) = x(k);
// bl_mynn_subroutines.F90:3725
}
// bl_mynn_subroutines.F90:3727
} else {
// bl_mynn_subroutines.F90:3729
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { qnwfa2(mynn_element+1) = qnwfa(mynn_element+1); }
// bl_mynn_subroutines.F90:3730
}
// bl_mynn_subroutines.F90:3735
if (((((bl_mynn_cloudmix) && (flag_qnifa))) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3738
k = kts;
// bl_mynn_subroutines.F90:3740
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3741
b(k) = mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3743
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3744
d(k) = mpas_sub(qnifa(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqnifa(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3746
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3747
a(k) = mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), nonloc));
// bl_mynn_subroutines.F90:3748
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3750
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3751
d(k) = mpas_add(qnifa(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqnifa(k), s_awqnifa(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3752
}
// bl_mynn_subroutines.F90:3755
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3756
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3757
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3758
d(kte) = qnifa(kte);
// bl_mynn_subroutines.F90:3761
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3764
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3766
qnifa2(k) = x(k);
// bl_mynn_subroutines.F90:3767
}
// bl_mynn_subroutines.F90:3769
} else {
// bl_mynn_subroutines.F90:3771
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { qnifa2(mynn_element+1) = qnifa(mynn_element+1); }
// bl_mynn_subroutines.F90:3772
}
// bl_mynn_subroutines.F90:3777
if (((((bl_mynn_cloudmix) && (flag_qnbca))) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3780
k = kts;
// bl_mynn_subroutines.F90:3782
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3783
b(k) = mpas_sub(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3785
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3786
d(k) = mpas_sub(qnbca(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), s_awqnbca(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3788
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3789
a(k) = mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), nonloc));
// bl_mynn_subroutines.F90:3790
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3792
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), nonloc));
// bl_mynn_subroutines.F90:3793
d(k) = mpas_add(qnbca(k), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqnbca(k), s_awqnbca(((k) + (1))))), nonloc));
// bl_mynn_subroutines.F90:3794
}
// bl_mynn_subroutines.F90:3797
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3798
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3799
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3800
d(kte) = qnbca(kte);
// bl_mynn_subroutines.F90:3803
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3806
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3808
qnbca2(k) = x(k);
// bl_mynn_subroutines.F90:3809
}
// bl_mynn_subroutines.F90:3811
} else {
// bl_mynn_subroutines.F90:3813
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { qnbca2(mynn_element+1) = qnbca(mynn_element+1); }
// bl_mynn_subroutines.F90:3814
}
// bl_mynn_subroutines.F90:3819
if (flag_ozone) {
// bl_mynn_subroutines.F90:3820
k = kts;
// bl_mynn_subroutines.F90:3823
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3824
b(k) = mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(((k) + (1))), khdz(k))), rhoinv(k)));
// bl_mynn_subroutines.F90:3825
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:3826
d(k) = ozone(k);
// bl_mynn_subroutines.F90:3828
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:3829
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:3830
b(k) = mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(khdz(k), khdz(((k) + (1))))), rhoinv(k)));
// bl_mynn_subroutines.F90:3831
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), khdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:3832
d(k) = ozone(k);
// bl_mynn_subroutines.F90:3833
}
// bl_mynn_subroutines.F90:3836
a(kte) = 0.f;
// bl_mynn_subroutines.F90:3837
b(kte) = 1.f;
// bl_mynn_subroutines.F90:3838
c(kte) = 0.f;
// bl_mynn_subroutines.F90:3839
d(kte) = ozone(kte);
// bl_mynn_subroutines.F90:3842
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:3845
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3847
dozone(k) = mpas_div(mpas_sub(x(k), ozone(k)), delt);
// bl_mynn_subroutines.F90:3848
}
// bl_mynn_subroutines.F90:3849
} else {
// bl_mynn_subroutines.F90:3850
for(int i=1;i<=nl;i++) { dozone(i)=0.0f; }
// bl_mynn_subroutines.F90:3851
}
// bl_mynn_subroutines.F90:3858
if (bl_mynn_mixqt) {
// bl_mynn_subroutines.F90:3859
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3861
th_new = mpas_add(mpas_add(thl(k), mpas_mul(mpas_div(xlvcp, exner(k)), sqc(k))), mpas_mul(mpas_div(xlscp, exner(k)), sqi(k)));
// bl_mynn_subroutines.F90:3864
t = mpas_mul(th_new, exner(k));
// bl_mynn_subroutines.F90:3865
qsat = f_qsat_blend(t, t0c, tice, p(k), 0);
// bl_mynn_subroutines.F90:3872
if (((soft32_greater(as_type<uint>(sqc(k)),0u)) || (soft32_greater(as_type<uint>(sqi(k)),0u)))) {
// bl_mynn_subroutines.F90:3873
sqv2(k) = mynn_min(((float)(sqw2(k))), ((float)(qsat)));
// bl_mynn_subroutines.F90:3874
portion_qc = mpas_div(sqc(k), mpas_add(sqc(k), sqi(k)));
// bl_mynn_subroutines.F90:3875
portion_qi = mpas_div(sqi(k), mpas_add(sqc(k), sqi(k)));
// bl_mynn_subroutines.F90:3876
condensate = mynn_max(((float)(mpas_sub(sqw2(k), qsat))), ((float)(0.0f)));
// bl_mynn_subroutines.F90:3877
sqc2(k) = mpas_mul(condensate, portion_qc);
// bl_mynn_subroutines.F90:3878
sqi2(k) = mpas_mul(condensate, portion_qi);
// bl_mynn_subroutines.F90:3879
} else {
// bl_mynn_subroutines.F90:3880
sqv2(k) = sqw2(k);
// bl_mynn_subroutines.F90:3881
sqi2(k) = 0.0f;
// bl_mynn_subroutines.F90:3882
sqc2(k) = 0.0f;
// bl_mynn_subroutines.F90:3883
}
// bl_mynn_subroutines.F90:3884
}
// bl_mynn_subroutines.F90:3885
}
// bl_mynn_subroutines.F90:3891
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3892
dqv(k) = mpas_div(mpas_sub(sqv2(k), sqv(k)), delt);
// bl_mynn_subroutines.F90:3894
}
// bl_mynn_subroutines.F90:3896
if (bl_mynn_cloudmix) {
// bl_mynn_subroutines.F90:3901
if (flag_qc) {
// bl_mynn_subroutines.F90:3902
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3903
dqc(k) = mpas_div(mpas_sub(sqc2(k), sqc(k)), delt);
// bl_mynn_subroutines.F90:3905
}
// bl_mynn_subroutines.F90:3906
} else {
// bl_mynn_subroutines.F90:3907
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3908
dqc(k) = 0.f;
// bl_mynn_subroutines.F90:3909
}
// bl_mynn_subroutines.F90:3910
}
// bl_mynn_subroutines.F90:3915
if (((flag_qnc) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3916
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3917
dqnc(k) = mpas_div(mpas_sub(qnc2(k), qnc(k)), delt);
// bl_mynn_subroutines.F90:3919
}
// bl_mynn_subroutines.F90:3920
} else {
// bl_mynn_subroutines.F90:3921
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3922
dqnc(k) = 0.f;
// bl_mynn_subroutines.F90:3923
}
// bl_mynn_subroutines.F90:3924
}
// bl_mynn_subroutines.F90:3929
if (flag_qi) {
// bl_mynn_subroutines.F90:3930
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3931
dqi(k) = mpas_div(mpas_sub(sqi2(k), sqi(k)), delt);
// bl_mynn_subroutines.F90:3933
}
// bl_mynn_subroutines.F90:3934
} else {
// bl_mynn_subroutines.F90:3935
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3936
dqi(k) = 0.f;
// bl_mynn_subroutines.F90:3937
}
// bl_mynn_subroutines.F90:3938
}
// bl_mynn_subroutines.F90:3943
if (false) {
// bl_mynn_subroutines.F90:3944
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3945
dqs(k) = mpas_div(mpas_sub(sqs2(k), sqs(k)), delt);
// bl_mynn_subroutines.F90:3946
}
// bl_mynn_subroutines.F90:3947
} else {
// bl_mynn_subroutines.F90:3948
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3949
dqs(k) = 0.f;
// bl_mynn_subroutines.F90:3950
}
// bl_mynn_subroutines.F90:3951
}
// bl_mynn_subroutines.F90:3956
if (((flag_qni) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:3957
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3958
dqni(k) = mpas_div(mpas_sub(qni2(k), qni(k)), delt);
// bl_mynn_subroutines.F90:3960
}
// bl_mynn_subroutines.F90:3961
} else {
// bl_mynn_subroutines.F90:3962
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3963
dqni(k) = 0.f;
// bl_mynn_subroutines.F90:3964
}
// bl_mynn_subroutines.F90:3965
}
// bl_mynn_subroutines.F90:3966
} else {
// bl_mynn_subroutines.F90:3968
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3969
dqc(k) = 0.f;
// bl_mynn_subroutines.F90:3970
dqnc(k) = 0.f;
// bl_mynn_subroutines.F90:3971
dqi(k) = 0.f;
// bl_mynn_subroutines.F90:3972
dqni(k) = 0.f;
// bl_mynn_subroutines.F90:3973
dqs(k) = 0.f;
// bl_mynn_subroutines.F90:3974
}
// bl_mynn_subroutines.F90:3975
}
// bl_mynn_subroutines.F90:3978
{

int k;
float dqc2;
float dqi2;
float dqs2;
float dqv2;
float sum;
float aa;
float dum;
constexpr float qvmin = 1e-20f;
constexpr float qcmin = 0.0f;
constexpr float qimin = 0.0f;
// bl_mynn_subroutines.F90:4124
for (k=kte; ((-(1))) > 0 ? k <= (1) : k >= (1); k += ((-(1)))) {
// bl_mynn_subroutines.F90:4125
dqc2 = mynn_max(((float)(0.0f)), ((float)(mpas_sub(qcmin, sqc2(k)))));
// bl_mynn_subroutines.F90:4126
dqi2 = mynn_max(((float)(0.0f)), ((float)(mpas_sub(qimin, sqi2(k)))));
// bl_mynn_subroutines.F90:4127
dqs2 = mynn_max(((float)(0.0f)), ((float)(mpas_sub(qimin, sqs2(k)))));
// bl_mynn_subroutines.F90:4130
dqc(k) = mpas_add(dqc(k), mpas_div(dqc2, delt));
// bl_mynn_subroutines.F90:4131
dqi(k) = mpas_add(dqi(k), mpas_div(dqi2, delt));
// bl_mynn_subroutines.F90:4132
dqs(k) = mpas_add(dqs(k), mpas_div(dqs2, delt));
// bl_mynn_subroutines.F90:4133
dqv(k) = mpas_sub(dqv(k), mpas_div(mpas_add(mpas_add(dqc2, dqi2), dqs2), delt));
// bl_mynn_subroutines.F90:4134
dth(k) = mpas_add(mpas_add(dth(k), mpas_mul(mpas_div(xlvcp, exner(k)), mpas_div(dqc2, delt))), mpas_mul(mpas_div(xlscp, exner(k)), mpas_div(mpas_add(dqi2, dqs2), delt)));
// bl_mynn_subroutines.F90:4137
sqc2(k) = mpas_add(sqc2(k), dqc2);
// bl_mynn_subroutines.F90:4138
sqi2(k) = mpas_add(sqi2(k), dqi2);
// bl_mynn_subroutines.F90:4139
sqs2(k) = mpas_add(sqs2(k), dqs2);
// bl_mynn_subroutines.F90:4140
sqv2(k) = mpas_sub(mpas_sub(mpas_sub(sqv2(k), dqc2), dqi2), dqs2);
// bl_mynn_subroutines.F90:4141
thl(k) = mpas_add(mpas_add(thl(k), mpas_mul(mpas_div(xlvcp, exner(k)), dqc2)), mpas_mul(mpas_div(xlscp, exner(k)), mpas_add(dqi2, dqs2)));
// bl_mynn_subroutines.F90:4145
dqv2 = mynn_max(((float)(0.0f)), ((float)(mpas_sub(qvmin, sqv2(k)))));
// bl_mynn_subroutines.F90:4146
dqv(k) = mpas_add(dqv(k), mpas_div(dqv2, delt));
// bl_mynn_subroutines.F90:4147
sqv2(k) = mpas_add(sqv2(k), dqv2);
// bl_mynn_subroutines.F90:4148
if (((k) != (1))) {
// bl_mynn_subroutines.F90:4149
sqv2(((k) - (1))) = mpas_sub(sqv2(((k) - (1))), mpas_div(mpas_mul(dqv2, delp(k)), delp(((k) - (1)))));
// bl_mynn_subroutines.F90:4150
dqv(((k) - (1))) = mpas_sub(dqv(((k) - (1))), mpas_div(mpas_div(mpas_mul(dqv2, delp(k)), delp(((k) - (1)))), delt));
// bl_mynn_subroutines.F90:4151
}
// bl_mynn_subroutines.F90:4152
sqv2(k) = mynn_max(((float)(sqv2(k))), ((float)(qvmin)));
// bl_mynn_subroutines.F90:4153
sqc2(k) = mynn_max(((float)(sqc2(k))), ((float)(qcmin)));
// bl_mynn_subroutines.F90:4154
sqi2(k) = mynn_max(((float)(sqi2(k))), ((float)(qimin)));
// bl_mynn_subroutines.F90:4155
sqs2(k) = mynn_max(((float)(sqs2(k))), ((float)(qimin)));
// bl_mynn_subroutines.F90:4156
}
// bl_mynn_subroutines.F90:4160
if (soft32_greater(as_type<uint>(dqv2),as_type<uint>(1.e-20f))) {
// bl_mynn_subroutines.F90:4161
sum = 0.0f;
// bl_mynn_subroutines.F90:4162
for (k=1; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4163
if (soft32_greater(as_type<uint>(sqv2(k)),as_type<uint>(mpas_mul(2.f,qvmin)))) { sum = mpas_add(sum, mpas_mul(sqv2(k), delp(k))); }
// bl_mynn_subroutines.F90:4164
}
// bl_mynn_subroutines.F90:4165
aa = mpas_div(mpas_mul(dqv2, delp(1)), mynn_max(((float)(1.e-20f)), ((float)(sum))));
// bl_mynn_subroutines.F90:4166
if (soft32_less(as_type<uint>(aa),as_type<uint>(0.5f))) {
// bl_mynn_subroutines.F90:4167
for (k=1; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4168
if (soft32_greater(as_type<uint>(sqv2(k)),as_type<uint>(mpas_mul(2.f,qvmin)))) {
// bl_mynn_subroutines.F90:4169
dum = mpas_mul(aa, sqv2(k));
// bl_mynn_subroutines.F90:4170
sqv2(k) = mpas_sub(sqv2(k), dum);
// bl_mynn_subroutines.F90:4171
dqv(k) = mpas_sub(dqv(k), mpas_div(dum, delt));
// bl_mynn_subroutines.F90:4172
}
// bl_mynn_subroutines.F90:4173
}
// bl_mynn_subroutines.F90:4174
} else {
// bl_mynn_subroutines.F90:4177
}
// bl_mynn_subroutines.F90:4178
}
// bl_mynn_subroutines.F90:4180


}
// bl_mynn_subroutines.F90:3985
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3986
if (soft32_less(as_type<uint>(mpas_add(mpas_mul(dozone(k),delt),ozone(k))),0u)) {
// bl_mynn_subroutines.F90:3987
dozone(k) = mpas_div(mpas_mul(mynn_neg(ozone(k)), 0.99f), delt);
// bl_mynn_subroutines.F90:3988
}
// bl_mynn_subroutines.F90:3989
}
// bl_mynn_subroutines.F90:3994
if (flag_qi) {
// bl_mynn_subroutines.F90:3995
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:3996
dth(k) = mpas_div(mpas_sub(mpas_add(mpas_add(thl(k), mpas_mul(mpas_div(xlvcp, exner(k)), sqc2(k))), mpas_mul(mpas_div(xlscp, exner(k)), mpas_add(sqi2(k), sqs(k)))), th(k)), delt);
// bl_mynn_subroutines.F90:4004
}
// bl_mynn_subroutines.F90:4005
} else {
// bl_mynn_subroutines.F90:4006
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4007
dth(k) = mpas_div(mpas_sub(mpas_add(thl(k), mpas_mul(mpas_div(xlvcp, exner(k)), sqc2(k))), th(k)), delt);
// bl_mynn_subroutines.F90:4012
}
// bl_mynn_subroutines.F90:4013
}
// bl_mynn_subroutines.F90:4018
if (((((flag_qnwfa) && (flag_qnifa))) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:4020
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4024
dqnwfa(k) = mpas_div(mpas_sub(qnwfa2(k), qnwfa(k)), delt);
// bl_mynn_subroutines.F90:4028
dqnifa(k) = mpas_div(mpas_sub(qnifa2(k), qnifa(k)), delt);
// bl_mynn_subroutines.F90:4029
}
// bl_mynn_subroutines.F90:4030
} else {
// bl_mynn_subroutines.F90:4031
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4032
dqnwfa(k) = 0.f;
// bl_mynn_subroutines.F90:4033
dqnifa(k) = 0.f;
// bl_mynn_subroutines.F90:4034
}
// bl_mynn_subroutines.F90:4035
}
// bl_mynn_subroutines.F90:4040
if (((flag_qnbca) && (bl_mynn_mixscalars))) {
// bl_mynn_subroutines.F90:4041
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4042
dqnbca(k) = mpas_div(mpas_sub(qnbca2(k), qnbca(k)), delt);
// bl_mynn_subroutines.F90:4043
}
// bl_mynn_subroutines.F90:4044
} else {
// bl_mynn_subroutines.F90:4045
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4046
dqnbca(k) = 0.f;
// bl_mynn_subroutines.F90:4047
}
// bl_mynn_subroutines.F90:4048
}
// bl_mynn_subroutines.F90:4057


#undef dz
#undef rho
#undef u
#undef v
#undef th
#undef tk
#undef qv
#undef qc
#undef qi
#undef qs
#undef qnc
#undef qni
#undef p
#undef exner
#undef thl
#undef sqv
#undef sqc
#undef sqi
#undef sqs
#undef sqw
#undef qnwfa
#undef qnifa
#undef qnbca
#undef ozone
#undef tsq
#undef qsq
#undef cov
#undef tcd
#undef qcd
#undef dfm
#undef dfh
#undef dfq
#undef du
#undef dv
#undef dth
#undef dqv
#undef dqc
#undef dqi
#undef dqs
#undef dqnc
#undef dqni
#undef dqnwfa
#undef dqnifa
#undef dqnbca
#undef dozone
#undef diss_heat
#undef s_aw
#undef s_awthl
#undef s_awqt
#undef s_awqv
#undef s_awqc
#undef s_awu
#undef s_awv
#undef s_awqnc
#undef s_awqni
#undef s_awqnwfa
#undef s_awqnifa
#undef s_awqnbca
#undef sd_aw
#undef sd_awthl
#undef sd_awqt
#undef sd_awqv
#undef sd_awqc
#undef sd_awu
#undef sd_awv
#undef sub_thl
#undef sub_sqv
#undef sub_u
#undef sub_v
#undef det_thl
#undef det_sqv
#undef det_sqc
#undef det_u
#undef det_v
#undef cldfra_bl1d
#undef dtz
#undef dfhc
#undef dfmc
#undef delp
#undef sqv2
#undef sqc2
#undef sqi2
#undef sqs2
#undef sqw2
#undef qni2
#undef qnc2
#undef qnwfa2
#undef qnifa2
#undef qnbca2
#undef ozone2
#undef zfac
#undef plumekh
#undef rhoinv
#undef a
#undef b
#undef c
#undef d
#undef x
#undef rhoz
#undef khdz
#undef kmdz
#undef cp
#undef dp
}

kernel void hexv842_mpas_bl_mynn_tendencies_pack0_f32(constant Hexv842MpasBlMynnTendenciesPack0F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnTendenciesPack0F32Params &cfg [[buffer(1)]],uint i [[thread_position_in_grid]]){uint n=uint(cfg.nc*cfg.nl);if(i>=n)return;
r.packed[ulong(0)*n+i]=r.dz[i];
r.packed[ulong(1)*n+i]=r.rho[i];
r.packed[ulong(2)*n+i]=r.u[i];
r.packed[ulong(3)*n+i]=r.v[i];
r.packed[ulong(4)*n+i]=r.th[i];
r.packed[ulong(5)*n+i]=r.tk[i];
r.packed[ulong(6)*n+i]=r.qv[i];
r.packed[ulong(7)*n+i]=r.qc[i];
r.packed[ulong(8)*n+i]=r.qi[i];
r.packed[ulong(9)*n+i]=r.qs[i];
r.packed[ulong(10)*n+i]=r.qnc[i];
r.packed[ulong(11)*n+i]=r.qni[i];
r.packed[ulong(12)*n+i]=r.p[i];
r.packed[ulong(13)*n+i]=r.exner[i];
r.packed[ulong(14)*n+i]=r.thl[i];
r.packed[ulong(15)*n+i]=r.sqv[i];
r.packed[ulong(16)*n+i]=r.sqc[i];
r.packed[ulong(17)*n+i]=r.sqi[i];
r.packed[ulong(18)*n+i]=r.sqs[i];
r.packed[ulong(19)*n+i]=r.sqw[i];
r.packed[ulong(20)*n+i]=r.qnwfa[i];
r.packed[ulong(21)*n+i]=r.qnifa[i];
r.packed[ulong(22)*n+i]=r.qnbca[i];
r.packed[ulong(23)*n+i]=r.ozone[i];
r.packed[ulong(24)*n+i]=r.tsq[i];
r.packed[ulong(25)*n+i]=r.qsq[i];
r.packed[ulong(26)*n+i]=r.cov[i];
r.packed[ulong(27)*n+i]=r.tcd[i];
r.packed[ulong(28)*n+i]=r.qcd[i];
r.packed[ulong(29)*n+i]=r.dfm[i];
r.packed[ulong(30)*n+i]=r.dfh[i];
r.packed[ulong(31)*n+i]=r.dfq[i];
r.packed[ulong(32)*n+i]=r.du[i];
r.packed[ulong(33)*n+i]=r.dv[i];
r.packed[ulong(34)*n+i]=r.dth[i];
r.packed[ulong(35)*n+i]=r.dqv[i];
r.packed[ulong(36)*n+i]=r.dqc[i];
r.packed[ulong(37)*n+i]=r.dqi[i];
r.packed[ulong(38)*n+i]=r.dqs[i];
r.packed[ulong(39)*n+i]=r.dqnc[i];
}

kernel void hexv842_mpas_bl_mynn_tendencies_pack1_f32(constant Hexv842MpasBlMynnTendenciesPack1F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnTendenciesPack1F32Params &cfg [[buffer(1)]],uint i [[thread_position_in_grid]]){uint n=uint(cfg.nc*cfg.nl);if(i>=n)return;
r.packed[ulong(40)*n+i]=r.dqni[i];
r.packed[ulong(41)*n+i]=r.dqnwfa[i];
r.packed[ulong(42)*n+i]=r.dqnifa[i];
r.packed[ulong(43)*n+i]=r.dqnbca[i];
r.packed[ulong(44)*n+i]=r.dozone[i];
r.packed[ulong(45)*n+i]=r.diss_heat[i];
r.packed[ulong(46)*n+i]=r.s_aw[i];
r.packed[ulong(47)*n+i]=r.s_awthl[i];
r.packed[ulong(48)*n+i]=r.s_awqt[i];
r.packed[ulong(49)*n+i]=r.s_awqv[i];
r.packed[ulong(50)*n+i]=r.s_awqc[i];
r.packed[ulong(51)*n+i]=r.s_awu[i];
r.packed[ulong(52)*n+i]=r.s_awv[i];
r.packed[ulong(53)*n+i]=r.s_awqnc[i];
r.packed[ulong(54)*n+i]=r.s_awqni[i];
r.packed[ulong(55)*n+i]=r.s_awqnwfa[i];
r.packed[ulong(56)*n+i]=r.s_awqnifa[i];
r.packed[ulong(57)*n+i]=r.s_awqnbca[i];
r.packed[ulong(58)*n+i]=r.sd_aw[i];
r.packed[ulong(59)*n+i]=r.sd_awthl[i];
r.packed[ulong(60)*n+i]=r.sd_awqt[i];
r.packed[ulong(61)*n+i]=r.sd_awqv[i];
r.packed[ulong(62)*n+i]=r.sd_awqc[i];
r.packed[ulong(63)*n+i]=r.sd_awu[i];
r.packed[ulong(64)*n+i]=r.sd_awv[i];
r.packed[ulong(65)*n+i]=r.sub_thl[i];
r.packed[ulong(66)*n+i]=r.sub_sqv[i];
r.packed[ulong(67)*n+i]=r.sub_u[i];
r.packed[ulong(68)*n+i]=r.sub_v[i];
r.packed[ulong(69)*n+i]=r.det_thl[i];
r.packed[ulong(70)*n+i]=r.det_sqv[i];
r.packed[ulong(71)*n+i]=r.det_sqc[i];
r.packed[ulong(72)*n+i]=r.det_u[i];
r.packed[ulong(73)*n+i]=r.det_v[i];
r.packed[ulong(74)*n+i]=r.cldfra_bl1d[i];
}

kernel void hexv842_mpas_bl_mynn_tendencies_scatter_f32(constant Hexv842MpasBlMynnTendenciesScatterF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnTendenciesScatterF32Params &cfg [[buffer(1)]],uint i [[thread_position_in_grid]]){uint n=uint(cfg.nc*cfg.nl);if(i>=n)return;
r.thl[i]=r.packed[ulong(14)*n+i];
r.du[i]=r.packed[ulong(32)*n+i];
r.dv[i]=r.packed[ulong(33)*n+i];
r.dth[i]=r.packed[ulong(34)*n+i];
r.dqv[i]=r.packed[ulong(35)*n+i];
r.dqc[i]=r.packed[ulong(36)*n+i];
r.dqi[i]=r.packed[ulong(37)*n+i];
r.dqs[i]=r.packed[ulong(38)*n+i];
r.dqnc[i]=r.packed[ulong(39)*n+i];
r.dqni[i]=r.packed[ulong(40)*n+i];
r.dqnwfa[i]=r.packed[ulong(41)*n+i];
r.dqnifa[i]=r.packed[ulong(42)*n+i];
r.dqnbca[i]=r.packed[ulong(43)*n+i];
r.dozone[i]=r.packed[ulong(44)*n+i];
}
