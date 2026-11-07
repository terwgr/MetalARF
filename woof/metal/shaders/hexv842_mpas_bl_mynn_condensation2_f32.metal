#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_condensation2_f32_bindings.metalh"
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

#include "hexv842_mpas_bl_mynn_condensation_exp_f32_bindings.metalh"
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

kernel void hexv842_mpas_bl_mynn_condensation2_f32(constant Hexv842MpasBlMynnCondensation2F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnCondensation2F32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl,spp_pbl=0;float xland=r.xland[col],pblh1=r.pblh1[col],hfx1=r.hfx1[col],rmo=r.rmo[col];constexpr float tice=240.f,tliq=269.f;
float cp=cfg.cp;
float cpv=cfg.cpv;
float cice=cfg.cice;
float cliq=cfg.cliq;
float ep_2=cfg.ep_2;
float r_d=cfg.r_d;
float r_v=cfg.r_v;
float xls=cfg.xls;
float xlv=cfg.xlv;
float t0c=cfg.t0c;
float xlvcp=cfg.xlvcp;
float ep1=cfg.ep1;
float tv0=mynn_mul(ep1,300.f);
#define dz(i) r.dz[ulong(col)*nl+((i)-1)]
#define zw(i) r.zw[ulong(col)*nl+((i)-1)]
#define thl(i) r.thl[ulong(col)*nl+((i)-1)]
#define qw(i) r.qw[ulong(col)*nl+((i)-1)]
#define qv(i) r.qv[ulong(col)*nl+((i)-1)]
#define qc(i) r.qc[ulong(col)*nl+((i)-1)]
#define qi(i) r.qi[ulong(col)*nl+((i)-1)]
#define qs(i) r.qs[ulong(col)*nl+((i)-1)]
#define p(i) r.p[ulong(col)*nl+((i)-1)]
#define exner(i) r.exner[ulong(col)*nl+((i)-1)]
#define tsq(i) r.tsq[ulong(col)*nl+((i)-1)]
#define qsq(i) r.qsq[ulong(col)*nl+((i)-1)]
#define cov(i) r.cov[ulong(col)*nl+((i)-1)]
#define sh(i) r.sh[ulong(col)*nl+((i)-1)]
#define el(i) r.el[ulong(col)*nl+((i)-1)]
#define th(i) r.th[ulong(col)*nl+((i)-1)]
#define rstoch_col(i) r.rstoch_col[ulong(col)*nl+((i)-1)]
#define qc_bl1d(i) r.qc_bl1d[ulong(col)*nl+((i)-1)]
#define qi_bl1d(i) r.qi_bl1d[ulong(col)*nl+((i)-1)]
#define cldfra_bl1d(i) r.cldfra_bl1d[ulong(col)*nl+((i)-1)]
#define vt(i) r.vt[ulong(col)*nl+((i)-1)]
#define vq(i) r.vq[ulong(col)*nl+((i)-1)]
#define sgm(i) r.sgm[ulong(col)*nl+((i)-1)]
#define alp(i) r.scratch[(ulong(col)*7+0)*nl+((i)-1)]
#define a(i) r.scratch[(ulong(col)*7+1)*nl+((i)-1)]
#define bet(i) r.scratch[(ulong(col)*7+2)*nl+((i)-1)]
#define b(i) r.scratch[(ulong(col)*7+3)*nl+((i)-1)]
#define ql(i) r.scratch[(ulong(col)*7+4)*nl+((i)-1)]
#define q1(i) r.scratch[(ulong(col)*7+5)*nl+((i)-1)]
#define rh(i) r.scratch[(ulong(col)*7+6)*nl+((i)-1)]

ulong t3sq;
ulong r3sq;
ulong c3sq;
float qsl;
float esat;
float qsat;
float dqsl;
float cld0;
float q1k;
float qlk;
float eq1;
float qll;
float q2p;
float pt;
float rac;
float qt;
float t;
float xl;
float rsl;
float cpm;
float fng;
float qww;
float alpha;
float beta;
float bb;
float ls;
float wt;
float wt2;
float qpct;
float cld_factor;
float fac_damp;
float liq_frac;
float ql_ice;
float ql_water;
float qmq;
float qsat_tk;
float q1_rh;
float rh_hack;
float dzm1;
float zsl;
float maxqc;
constexpr float qpct_sfc = 0.025f;
constexpr float qpct_pbl = 0.030f;
constexpr float qpct_trp = 0.040f;
constexpr float rhcrit = 0.83f;
constexpr float rhmax = 1.02f;
int i;
int j;
int k;
float erf;
float dth;
float dtl;
float dqw;
float dzk;
float els;
float zagl;
float damp;
float pblh2;
float cfmax;
float theta1;
float theta2;
float ht1;
float ht2;
int k_tropo;
float qw_pert;
// bl_mynn_subroutines.F90:2632
for (k=((kte) - (3)); ((-(1))) > 0 ? k <= (kts) : k >= (kts); k += ((-(1)))) {
// bl_mynn_subroutines.F90:2633
theta1 = th(k);
// bl_mynn_subroutines.F90:2634
theta2 = th(((k) + (2)));
// bl_mynn_subroutines.F90:2635
ht1 = mpas_mul(44307.692f, mpas_sub(1.0f, mynn_pow(((float)(mpas_div(p(k), 101325.f))), ((float)(0.190f)))));
// bl_mynn_subroutines.F90:2636
ht2 = mpas_mul(44307.692f, mpas_sub(1.0f, mynn_pow(((float)(mpas_div(p(((k) + (2))), 101325.f))), ((float)(0.190f)))));
// bl_mynn_subroutines.F90:2637
if (((((soft32_less(as_type<uint>(mpas_div(mpas_sub(theta2,theta1),mpas_sub(ht2,ht1))),as_type<uint>(mpas_div(10.f,1500.f)))) && (soft32_less(as_type<uint>(ht1),as_type<uint>(19000.f))))) && (soft32_greater(as_type<uint>(ht1),as_type<uint>(4000.f))))) {
// bl_mynn_subroutines.F90:2639
break;
// bl_mynn_subroutines.F90:2640
}
// bl_mynn_subroutines.F90:2641
}
// bl_mynn_subroutines.F90:2642

// bl_mynn_subroutines.F90:2643
k_tropo = mynn_max(((int)(((kts) + (2)))), ((int)(((k) + (2)))));
// bl_mynn_subroutines.F90:2645
zagl = 0.f;
// bl_mynn_subroutines.F90:2647

// bl_mynn_subroutines.F90:2778
pblh2 = mynn_max(((float)(10.f)), ((float)(pblh1)));
// bl_mynn_subroutines.F90:2779
zagl = 0.f;
// bl_mynn_subroutines.F90:2780
dzm1 = 0.f;
// bl_mynn_subroutines.F90:2781
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2782
zagl = mpas_add(zagl, mpas_mul(0.5f, mpas_add(dz(k), dzm1)));
// bl_mynn_subroutines.F90:2783
dzm1 = dz(k);
// bl_mynn_subroutines.F90:2785
t = mpas_mul(th(k), exner(k));
// bl_mynn_subroutines.F90:2786
xl = f_xl_blend(t, t0c, tice, cice, cliq, cpv, xls, xlv);
// bl_mynn_subroutines.F90:2787
qsat_tk = f_qsat_blend(t, t0c, tice, p(k), 0);
// bl_mynn_subroutines.F90:2788
rh(k) = mynn_max(((float)(mynn_min(((float)(mpas_div(qw(k), mynn_max(((float)(1.e-10f)), ((float)(qsat_tk)))))), ((float)(rhmax))))), ((float)(0.001f)));
// bl_mynn_subroutines.F90:2791
dqsl = mpas_div(mpas_mul(mpas_mul(qsat_tk, ep_2), xlv), mpas_mul(r_d, mynn_square(t)));
// bl_mynn_subroutines.F90:2792
alp(k) = mpas_div(1.0f, mpas_add(1.0f, mpas_mul(dqsl, xlvcp)));
// bl_mynn_subroutines.F90:2793
bet(k) = mpas_mul(dqsl, exner(k));
// bl_mynn_subroutines.F90:2795
rsl = mpas_div(mpas_mul(xl, qsat_tk), mpas_mul(r_v, mynn_square(t)));
// bl_mynn_subroutines.F90:2797
cpm = mpas_add(cp, mpas_mul(qw(k), cpv));
// bl_mynn_subroutines.F90:2798
a(k) = mpas_div(1.f, mpas_add(1.f, mpas_div(mpas_mul(xl, rsl), cpm)));
// bl_mynn_subroutines.F90:2799
b(k) = mpas_mul(a(k), rsl);
// bl_mynn_subroutines.F90:2802
qw_pert = mpas_add(qw(k), mpas_mul(mpas_mul(mpas_mul(qw(k), 0.5f), rstoch_col(k)), ((float)(spp_pbl))));
// bl_mynn_subroutines.F90:2805
qmq = mpas_sub(qw_pert, qsat_tk);
// bl_mynn_subroutines.F90:2809
r3sq = mynn_real64(mynn_max(((float)(qsq(k))), ((float)(0.0f))));
// bl_mynn_subroutines.F90:2811
sgm(k) = mynn_real32(mynn_sqrt(r3sq));
// bl_mynn_subroutines.F90:2813
sgm(k) = mynn_min(((float)(mpas_mul(qsat_tk, 0.666f))), ((float)(sgm(k))));
// bl_mynn_subroutines.F90:2817
wt = mpas_div(mynn_max(((float)(mpas_sub(500.f, mynn_max(((float)(mpas_sub(dz(k), 100.f))), ((float)(0.0f)))))), ((float)(0.0f))), 500.f);
// bl_mynn_subroutines.F90:2818
sgm(k) = mpas_add(sgm(k), mpas_mul(mpas_mul(sgm(k), 0.2f), mpas_sub(1.0f, wt)));
// bl_mynn_subroutines.F90:2821
qpct = mpas_add(mpas_mul(qpct_pbl, wt), mpas_mul(qpct_trp, mpas_sub(1.0f, wt)));
// bl_mynn_subroutines.F90:2822
qpct = mynn_min(((float)(qpct)), ((float)(mynn_max(((float)(qpct_sfc)), ((float)(mpas_div(mpas_mul(qpct_pbl, zagl), 500.f)))))));
// bl_mynn_subroutines.F90:2823
sgm(k) = mynn_max(((float)(mpas_mul(qsat_tk, qpct))), ((float)(sgm(k))));
// bl_mynn_subroutines.F90:2825
q1(k) = mpas_div(qmq, sgm(k));
// bl_mynn_subroutines.F90:2829
rh_hack = rh(k);
// bl_mynn_subroutines.F90:2830
wt2 = mynn_min(((float)(mpas_div(mynn_max(((float)(mpas_sub(zagl, pblh2))), ((float)(0.0f))), 300.f))), ((float)(1.0f)));
// bl_mynn_subroutines.F90:2832
if (((soft32_greater(as_type<uint>(mpas_add(qi(k),qs(k))),as_type<uint>(1.e-9f))) && (soft32_greater(as_type<uint>(zagl),as_type<uint>(pblh2))))) {
// bl_mynn_subroutines.F90:2833
rh_hack = mynn_min(((float)(rhmax)), ((float)(mpas_add(rhcrit, mpas_mul(mpas_mul(wt2, 0.045f), mpas_add(9.0f, mynn_log10(mpas_add(qi(k), qs(k)))))))));
// bl_mynn_subroutines.F90:2834
rh(k) = mynn_max(((float)(rh(k))), ((float)(rh_hack)));
// bl_mynn_subroutines.F90:2836
q1_rh = mpas_add(mynn_neg(3.f), mpas_div(mpas_mul(3.f, mpas_sub(rh(k), rhcrit)), mpas_sub(1.f, rhcrit)));
// bl_mynn_subroutines.F90:2837
q1(k) = mynn_max(((float)(q1_rh)), ((float)(q1(k))));
// bl_mynn_subroutines.F90:2838
}
// bl_mynn_subroutines.F90:2840
if (((soft32_greater(as_type<uint>(qc(k)),as_type<uint>(1.e-6f))) && (soft32_greater(as_type<uint>(zagl),as_type<uint>(pblh2))))) {
// bl_mynn_subroutines.F90:2841
rh_hack = mynn_min(((float)(rhmax)), ((float)(mpas_add(rhcrit, mpas_mul(mpas_mul(wt2, 0.08f), mpas_add(6.0f, mynn_log10(qc(k))))))));
// bl_mynn_subroutines.F90:2842
rh(k) = mynn_max(((float)(rh(k))), ((float)(rh_hack)));
// bl_mynn_subroutines.F90:2844
q1_rh = mpas_add(mynn_neg(3.f), mpas_div(mpas_mul(3.f, mpas_sub(rh(k), rhcrit)), mpas_sub(1.f, rhcrit)));
// bl_mynn_subroutines.F90:2845
q1(k) = mynn_max(((float)(q1_rh)), ((float)(q1(k))));
// bl_mynn_subroutines.F90:2846
}
// bl_mynn_subroutines.F90:2848
q1k = q1(k);
// bl_mynn_subroutines.F90:2856
cldfra_bl1d(k) = mynn_max(((float)(mynn_min(((float)(mpas_add(0.5f, mpas_mul(0.36f, mynn_atan(mpas_mul(1.8f, mpas_add(q1(k), 0.2f))))))), ((float)(1.f))))), ((float)(0.f)));
// bl_mynn_subroutines.F90:2861
maxqc = mynn_max(((float)(mpas_sub(qw(k), qsat_tk))), ((float)(0.0f)));
// bl_mynn_subroutines.F90:2862
if (soft32_less(as_type<uint>(q1k),0u)) {
// bl_mynn_subroutines.F90:2863
ql_water = mpas_mul(sgm(k), mynn_exp(mpas_sub(mpas_mul(1.2f, q1k), 1.f)));
// bl_mynn_subroutines.F90:2864
ql_ice = mpas_mul(sgm(k), mynn_exp(mpas_sub(mpas_mul(1.2f, q1k), 1.f)));
// bl_mynn_subroutines.F90:2865
} else if (soft32_greater(as_type<uint>(q1k),as_type<uint>(2.f))) {
// bl_mynn_subroutines.F90:2866
ql_water = mynn_min(((float)(maxqc)), ((float)(mpas_mul(sgm(k), q1k))));
// bl_mynn_subroutines.F90:2867
ql_ice = mpas_mul(sgm(k), q1k);
// bl_mynn_subroutines.F90:2868
} else {
// bl_mynn_subroutines.F90:2869
ql_water = mynn_min(((float)(maxqc)), ((float)(mpas_mul(sgm(k), mpas_add(mpas_add(mynn_exp(mynn_neg(1.f)), mpas_mul(0.66f, q1k)), mpas_mul(0.086f, mynn_square(q1k)))))));
// bl_mynn_subroutines.F90:2870
ql_ice = mpas_mul(sgm(k), mpas_add(mpas_add(mynn_exp(mynn_neg(1.f)), mpas_mul(0.66f, q1k)), mpas_mul(0.086f, mynn_square(q1k))));
// bl_mynn_subroutines.F90:2871
}
// bl_mynn_subroutines.F90:2878
if (soft32_less(as_type<uint>(cldfra_bl1d(k)),as_type<uint>(0.001f))) {
// bl_mynn_subroutines.F90:2879
ql_ice = 0.0f;
// bl_mynn_subroutines.F90:2880
ql_water = 0.0f;
// bl_mynn_subroutines.F90:2881
cldfra_bl1d(k) = 0.0f;
// bl_mynn_subroutines.F90:2882
}
// bl_mynn_subroutines.F90:2884
liq_frac = mynn_min(((float)(1.0f)), ((float)(mynn_max(((float)(0.0f)), ((float)(mpas_div(mpas_sub(t, tice), mpas_sub(tliq, tice))))))));
// bl_mynn_subroutines.F90:2885
qc_bl1d(k) = mpas_mul(liq_frac, ql_water);
// bl_mynn_subroutines.F90:2886
qi_bl1d(k) = mpas_mul(mpas_sub(1.0f, liq_frac), ql_ice);
// bl_mynn_subroutines.F90:2890
if (((k) >= (k_tropo))) {
// bl_mynn_subroutines.F90:2891
cldfra_bl1d(k) = 0.f;
// bl_mynn_subroutines.F90:2892
qc_bl1d(k) = 0.f;
// bl_mynn_subroutines.F90:2893
qi_bl1d(k) = 0.f;
// bl_mynn_subroutines.F90:2894
}
// bl_mynn_subroutines.F90:2899
if (soft32_greater_equal(as_type<uint>(mpas_sub(xland,1.5f)),0u)) {
// bl_mynn_subroutines.F90:2900
q1k = mynn_max(((float)(q1(k))), ((float)(mynn_neg(2.5f))));
// bl_mynn_subroutines.F90:2901
} else {
// bl_mynn_subroutines.F90:2902
q1k = mynn_max(((float)(q1(k))), ((float)(mynn_neg(2.0f))));
// bl_mynn_subroutines.F90:2903
}
// bl_mynn_subroutines.F90:2916
if (soft32_greater_equal(as_type<uint>(q1k),as_type<uint>(1.f))) {
// bl_mynn_subroutines.F90:2917
fng = 1.0f;
// bl_mynn_subroutines.F90:2918
} else if (((soft32_greater_equal(as_type<uint>(q1k),as_type<uint>(mynn_neg(1.7f)))) && (soft32_less(as_type<uint>(q1k),as_type<uint>(1.f))))) {
// bl_mynn_subroutines.F90:2919
fng = mynn_exp(mpas_mul(mynn_neg(0.4f), mpas_sub(q1k, 1.0f)));
// bl_mynn_subroutines.F90:2920
} else if (((soft32_greater_equal(as_type<uint>(q1k),as_type<uint>(mynn_neg(2.5f)))) && (soft32_less(as_type<uint>(q1k),as_type<uint>(mynn_neg(1.7f)))))) {
// bl_mynn_subroutines.F90:2921
fng = mpas_add(3.0f, mynn_exp(mpas_mul(mynn_neg(3.8f), mpas_add(q1k, 1.7f))));
// bl_mynn_subroutines.F90:2922
} else {
// bl_mynn_subroutines.F90:2923
fng = mynn_min(((float)(mpas_add(23.9f, mynn_exp(mpas_mul(mynn_neg(1.6f), mpas_add(q1k, 2.5f)))))), ((float)(60.f)));
// bl_mynn_subroutines.F90:2924
}
// bl_mynn_subroutines.F90:2926
cfmax = mynn_min(((float)(cldfra_bl1d(k))), ((float)(0.6f)));
// bl_mynn_subroutines.F90:2928
zsl = mynn_min(((float)(mynn_max(((float)(25.f)), ((float)(mpas_mul(0.1f, pblh2)))))), ((float)(100.f)));
// bl_mynn_subroutines.F90:2929
wt = mynn_min(((float)(mpas_div(zagl, zsl))), ((float)(1.0f)));
// bl_mynn_subroutines.F90:2930
cfmax = mpas_mul(cfmax, wt);
// bl_mynn_subroutines.F90:2932
bb = mpas_div(mpas_mul(b(k), t), th(k));
// bl_mynn_subroutines.F90:2938
qww = mpas_add(1.f, mpas_mul(0.61f, qw(k)));
// bl_mynn_subroutines.F90:2939
alpha = mpas_mul(0.61f, th(k));
// bl_mynn_subroutines.F90:2940
beta = mpas_sub(mpas_mul(mpas_div(th(k), t), mpas_div(xl, cp)), mpas_mul(1.61f, th(k)));
// bl_mynn_subroutines.F90:2941
vt(k) = mpas_sub(mpas_sub(qww, mpas_mul(mpas_mul(mpas_mul(cfmax, beta), bb), fng)), 1.f);
// bl_mynn_subroutines.F90:2942
vq(k) = mpas_sub(mpas_add(alpha, mpas_mul(mpas_mul(mpas_mul(cfmax, beta), a(k)), fng)), tv0);
// bl_mynn_subroutines.F90:2950
fac_damp = mynn_min(((float)(mpas_mul(zagl, 0.0025f))), ((float)(1.0f)));
// bl_mynn_subroutines.F90:2953
cld_factor = mpas_add(1.0f, mpas_mul(fac_damp, mynn_min(((float)(mynn_square(mpas_div(mynn_max(((float)(0.0f)), ((float)(mpas_sub(rh(k), 0.92f)))), 0.145f)))), ((float)(0.37f)))));
// bl_mynn_subroutines.F90:2954
cldfra_bl1d(k) = mynn_min(((float)(1.f)), ((float)(mpas_mul(cld_factor, cldfra_bl1d(k)))));
// bl_mynn_subroutines.F90:2955
}
// bl_mynn_subroutines.F90:2957

// bl_mynn_subroutines.F90:2960

// bl_mynn_subroutines.F90:2968
ql(kte) = ql(((kte) - (1)));
// bl_mynn_subroutines.F90:2969
vt(kte) = vt(((kte) - (1)));
// bl_mynn_subroutines.F90:2970
vq(kte) = vq(((kte) - (1)));
// bl_mynn_subroutines.F90:2971
qc_bl1d(kte) = 0.f;
// bl_mynn_subroutines.F90:2972
qi_bl1d(kte) = 0.f;
// bl_mynn_subroutines.F90:2973
cldfra_bl1d(kte) = 0.f;
// bl_mynn_subroutines.F90:2974
return;

#undef dz
#undef zw
#undef thl
#undef qw
#undef qv
#undef qc
#undef qi
#undef qs
#undef p
#undef exner
#undef tsq
#undef qsq
#undef cov
#undef sh
#undef el
#undef th
#undef rstoch_col
#undef qc_bl1d
#undef qi_bl1d
#undef cldfra_bl1d
#undef vt
#undef vq
#undef sgm
#undef alp
#undef a
#undef bet
#undef b
#undef ql
#undef q1
#undef rh
}

kernel void hexv842_mpas_bl_mynn_condensation_exp_f32(constant Hexv842MpasBlMynnCondensationExpF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnCondensationExpF32Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]){if(i<uint(p.count))r.output[i]=mynn_exp(r.input[i]);}
