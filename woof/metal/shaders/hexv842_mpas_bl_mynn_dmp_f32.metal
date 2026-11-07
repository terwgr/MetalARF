#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_dmp_f32_bindings.metalh"
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
#include "hexv842_mpas_bl_mynn_dmp_pack0_f32_bindings.metalh"
#include "hexv842_mpas_bl_mynn_dmp_pack1_f32_bindings.metalh"
#include "hexv842_mpas_bl_mynn_dmp_pack2_f32_bindings.metalh"
#include "hexv842_mpas_bl_mynn_dmp_scatter_f32_bindings.metalh"

static inline int mynn_min(int a,int b){return a<b?a:b;}
static inline float f_esat_blend(float t,float t0c,float tice){

float esat_blend;
float xc;
float esl;
float esi;
float chi;
// mynn_shared.F90:50
xc = mynn_max(((float)(mynn_neg(80.f))), ((float)(mpas_sub(t, t0c))));
// mynn_shared.F90:54
if (soft32_greater_equal(as_type<uint>(t),as_type<uint>(t0c))) {
// mynn_shared.F90:55
esat_blend = mpas_add(j0, mpas_mul(xc, mpas_add(j1, mpas_mul(xc, mpas_add(j2, mpas_mul(xc, mpas_add(j3, mpas_mul(xc, mpas_add(j4, mpas_mul(xc, mpas_add(j5, mpas_mul(xc, mpas_add(j6, mpas_mul(xc, mpas_add(j7, mpas_mul(xc, j8))))))))))))))));
// mynn_shared.F90:56
} else if (soft32_less_equal(as_type<uint>(t),as_type<uint>(tice))) {
// mynn_shared.F90:57
esat_blend = mpas_add(k0, mpas_mul(xc, mpas_add(k1, mpas_mul(xc, mpas_add(k2, mpas_mul(xc, mpas_add(k3, mpas_mul(xc, mpas_add(k4, mpas_mul(xc, mpas_add(k5, mpas_mul(xc, mpas_add(k6, mpas_mul(xc, mpas_add(k7, mpas_mul(xc, k8))))))))))))))));
// mynn_shared.F90:58
} else {
// mynn_shared.F90:59
esl = mpas_add(j0, mpas_mul(xc, mpas_add(j1, mpas_mul(xc, mpas_add(j2, mpas_mul(xc, mpas_add(j3, mpas_mul(xc, mpas_add(j4, mpas_mul(xc, mpas_add(j5, mpas_mul(xc, mpas_add(j6, mpas_mul(xc, mpas_add(j7, mpas_mul(xc, j8))))))))))))))));
// mynn_shared.F90:60
esi = mpas_add(k0, mpas_mul(xc, mpas_add(k1, mpas_mul(xc, mpas_add(k2, mpas_mul(xc, mpas_add(k3, mpas_mul(xc, mpas_add(k4, mpas_mul(xc, mpas_add(k5, mpas_mul(xc, mpas_add(k6, mpas_mul(xc, mpas_add(k7, mpas_mul(xc, k8))))))))))))))));
// mynn_shared.F90:61
chi = mpas_div(mpas_sub(273.16f, t), 20.16f);
// mynn_shared.F90:62
esat_blend = mpas_add(mpas_mul(mpas_sub(1.f, chi), esl), mpas_mul(chi, esi));
// mynn_shared.F90:63
}
return esat_blend;

}
static inline void f_condensation_edmf(float qt,float thl,float p,float zagl,thread float &thv,thread float &qc,float p1000mb,float rcp,float xlvcp,float t0c,float rvovrd){constexpr float tice=240.f,onethird=1.f/3.f;constexpr bool env_subs=false;

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
qs = f_qsat_blend(t,t0c,tice,p,0);
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
qs = f_qsat_blend(t,t0c,tice,p,0);
// bl_mynn_subroutines.F90:5824
qc = mynn_max(((float)(mpas_sub(qt, qs))), ((float)(0.f)));
// bl_mynn_subroutines.F90:5827
if (soft32_less(as_type<uint>(zagl),as_type<uint>(100.f))) { qc = 0.f; }
// bl_mynn_subroutines.F90:5830
thv = mpas_mul(mpas_add(thl, mpas_mul(xlvcp, qc)), mpas_sub(mpas_add(1.f, mpas_mul(qt, mpas_sub(rvovrd, 1.f))), mpas_mul(rvovrd, qc)));

}

kernel void hexv842_mpas_bl_mynn_dmp_f32(constant Hexv842MpasBlMynnDmpF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDmpF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl,ii=int(col)+1,spp_pbl=0,kpbl=r.kpbl[col];constexpr float tice=240.f,onethird=1.f/3.f;constexpr bool env_subs=false;
float ust=r.ust[col];
float flt=r.flt[col];
float fltv=r.fltv[col];
float flq=r.flq[col];
float flqv=r.flqv[col];
float pblh=r.pblh[col];
float dx=r.dx[col];
float landsea=r.landsea[col];
float ts=r.ts[col];
float psig_shcu=r.psig_shcu[col];
float dt=cfg.dt;
float p1000mb=cfg.p1000mb;
float rcp=cfg.rcp;
float xlvcp=cfg.xlvcp;
float t0c=cfg.t0c;
float rvovrd=cfg.rvovrd;
float r_v=cfg.r_v;
float ep_2=cfg.ep_2;
float ep_3=cfg.ep_3;
float cice=cfg.cice;
float cliq=cfg.cliq;
float cpv=cfg.cpv;
float xls=cfg.xls;
float xlv=cfg.xlv;
float cp=cfg.cp;
float p608=cfg.p608;
float gravity=cfg.gravity;
float ep1=cfg.ep1;
bool momentum_opt=cfg.momentum_opt!=0;
bool tke_opt=cfg.tke_opt!=0;
bool scalar_opt=cfg.scalar_opt!=0;
float grav=gravity,gtr=mynn_div(grav,300.f),tv0=mynn_mul(ep1,300.f),maxwidth=r.maxwidth[col],maxmf=r.maxmf[col],ztop=r.ztop[col];int ktop=r.ktop[col];
#define zw(i) r.packed[(ulong(0)*cfg.nc+col)*(nl+1)+((i)-1)]
#define dz(i) r.packed[(ulong(1)*cfg.nc+col)*(nl+1)+((i)-1)]
#define p(i) r.packed[(ulong(2)*cfg.nc+col)*(nl+1)+((i)-1)]
#define rho(i) r.packed[(ulong(3)*cfg.nc+col)*(nl+1)+((i)-1)]
#define u(i) r.packed[(ulong(4)*cfg.nc+col)*(nl+1)+((i)-1)]
#define v(i) r.packed[(ulong(5)*cfg.nc+col)*(nl+1)+((i)-1)]
#define w(i) r.packed[(ulong(6)*cfg.nc+col)*(nl+1)+((i)-1)]
#define th(i) r.packed[(ulong(7)*cfg.nc+col)*(nl+1)+((i)-1)]
#define thl(i) r.packed[(ulong(8)*cfg.nc+col)*(nl+1)+((i)-1)]
#define thv(i) r.packed[(ulong(9)*cfg.nc+col)*(nl+1)+((i)-1)]
#define tk(i) r.packed[(ulong(10)*cfg.nc+col)*(nl+1)+((i)-1)]
#define qt(i) r.packed[(ulong(11)*cfg.nc+col)*(nl+1)+((i)-1)]
#define qv(i) r.packed[(ulong(12)*cfg.nc+col)*(nl+1)+((i)-1)]
#define qc(i) r.packed[(ulong(13)*cfg.nc+col)*(nl+1)+((i)-1)]
#define qke(i) r.packed[(ulong(14)*cfg.nc+col)*(nl+1)+((i)-1)]
#define qnc(i) r.packed[(ulong(15)*cfg.nc+col)*(nl+1)+((i)-1)]
#define qni(i) r.packed[(ulong(16)*cfg.nc+col)*(nl+1)+((i)-1)]
#define qnwfa(i) r.packed[(ulong(17)*cfg.nc+col)*(nl+1)+((i)-1)]
#define qnifa(i) r.packed[(ulong(18)*cfg.nc+col)*(nl+1)+((i)-1)]
#define qnbca(i) r.packed[(ulong(19)*cfg.nc+col)*(nl+1)+((i)-1)]
#define exner(i) r.packed[(ulong(20)*cfg.nc+col)*(nl+1)+((i)-1)]
#define vt(i) r.packed[(ulong(21)*cfg.nc+col)*(nl+1)+((i)-1)]
#define vq(i) r.packed[(ulong(22)*cfg.nc+col)*(nl+1)+((i)-1)]
#define sgm(i) r.packed[(ulong(23)*cfg.nc+col)*(nl+1)+((i)-1)]
#define edmf_a(i) r.packed[(ulong(24)*cfg.nc+col)*(nl+1)+((i)-1)]
#define edmf_w(i) r.packed[(ulong(25)*cfg.nc+col)*(nl+1)+((i)-1)]
#define edmf_qt(i) r.packed[(ulong(26)*cfg.nc+col)*(nl+1)+((i)-1)]
#define edmf_thl(i) r.packed[(ulong(27)*cfg.nc+col)*(nl+1)+((i)-1)]
#define edmf_ent(i) r.packed[(ulong(28)*cfg.nc+col)*(nl+1)+((i)-1)]
#define edmf_qc(i) r.packed[(ulong(29)*cfg.nc+col)*(nl+1)+((i)-1)]
#define s_aw(i) r.packed[(ulong(30)*cfg.nc+col)*(nl+1)+((i)-1)]
#define s_awthl(i) r.packed[(ulong(31)*cfg.nc+col)*(nl+1)+((i)-1)]
#define s_awqt(i) r.packed[(ulong(32)*cfg.nc+col)*(nl+1)+((i)-1)]
#define s_awqv(i) r.packed[(ulong(33)*cfg.nc+col)*(nl+1)+((i)-1)]
#define s_awqc(i) r.packed[(ulong(34)*cfg.nc+col)*(nl+1)+((i)-1)]
#define s_awu(i) r.packed[(ulong(35)*cfg.nc+col)*(nl+1)+((i)-1)]
#define s_awv(i) r.packed[(ulong(36)*cfg.nc+col)*(nl+1)+((i)-1)]
#define s_awqke(i) r.packed[(ulong(37)*cfg.nc+col)*(nl+1)+((i)-1)]
#define s_awqnc(i) r.packed[(ulong(38)*cfg.nc+col)*(nl+1)+((i)-1)]
#define s_awqni(i) r.packed[(ulong(39)*cfg.nc+col)*(nl+1)+((i)-1)]
#define s_awqnwfa(i) r.packed[(ulong(40)*cfg.nc+col)*(nl+1)+((i)-1)]
#define s_awqnifa(i) r.packed[(ulong(41)*cfg.nc+col)*(nl+1)+((i)-1)]
#define s_awqnbca(i) r.packed[(ulong(42)*cfg.nc+col)*(nl+1)+((i)-1)]
#define sub_thl(i) r.packed[(ulong(43)*cfg.nc+col)*(nl+1)+((i)-1)]
#define sub_sqv(i) r.packed[(ulong(44)*cfg.nc+col)*(nl+1)+((i)-1)]
#define sub_u(i) r.packed[(ulong(45)*cfg.nc+col)*(nl+1)+((i)-1)]
#define sub_v(i) r.packed[(ulong(46)*cfg.nc+col)*(nl+1)+((i)-1)]
#define det_thl(i) r.packed[(ulong(47)*cfg.nc+col)*(nl+1)+((i)-1)]
#define det_sqv(i) r.packed[(ulong(48)*cfg.nc+col)*(nl+1)+((i)-1)]
#define det_sqc(i) r.packed[(ulong(49)*cfg.nc+col)*(nl+1)+((i)-1)]
#define det_u(i) r.packed[(ulong(50)*cfg.nc+col)*(nl+1)+((i)-1)]
#define det_v(i) r.packed[(ulong(51)*cfg.nc+col)*(nl+1)+((i)-1)]
#define qc_bl1d(i) r.packed[(ulong(52)*cfg.nc+col)*(nl+1)+((i)-1)]
#define cldfra_bl1d(i) r.packed[(ulong(53)*cfg.nc+col)*(nl+1)+((i)-1)]
#define qc_bl1d_old(i) r.packed[(ulong(54)*cfg.nc+col)*(nl+1)+((i)-1)]
#define cldfra_bl1d_old(i) r.packed[(ulong(55)*cfg.nc+col)*(nl+1)+((i)-1)]
#define rstoch_col(i) r.packed[(ulong(56)*cfg.nc+col)*(nl+1)+((i)-1)]
#define edmf_th(i) r.scratch[(ulong(col)*150+0)*(nl+1)+((i)-1)]
#define s_aw2(i) r.scratch[(ulong(col)*150+1)*(nl+1)+((i)-1)]
#define exneri(i) r.scratch[(ulong(col)*150+2)*(nl+1)+((i)-1)]
#define dzi(i) r.scratch[(ulong(col)*150+3)*(nl+1)+((i)-1)]
#define rhoz(i) r.scratch[(ulong(col)*150+4)*(nl+1)+((i)-1)]
#define envm_a(i) r.scratch[(ulong(col)*150+5)*(nl+1)+((i)-1)]
#define envm_w(i) r.scratch[(ulong(col)*150+6)*(nl+1)+((i)-1)]
#define envm_thl(i) r.scratch[(ulong(col)*150+7)*(nl+1)+((i)-1)]
#define envm_sqv(i) r.scratch[(ulong(col)*150+8)*(nl+1)+((i)-1)]
#define envm_sqc(i) r.scratch[(ulong(col)*150+9)*(nl+1)+((i)-1)]
#define envm_u(i) r.scratch[(ulong(col)*150+10)*(nl+1)+((i)-1)]
#define envm_v(i) r.scratch[(ulong(col)*150+11)*(nl+1)+((i)-1)]
#define envi_a(i) r.scratch[(ulong(col)*150+12)*(nl+1)+((i)-1)]
#define envi_w(i) r.scratch[(ulong(col)*150+13)*(nl+1)+((i)-1)]
#define upw(i,j) r.scratch[(ulong(col)*150+14+((j)-1))*(nl+1)+((i)-1)]
#define upthl(i,j) r.scratch[(ulong(col)*150+22+((j)-1))*(nl+1)+((i)-1)]
#define upqt(i,j) r.scratch[(ulong(col)*150+30+((j)-1))*(nl+1)+((i)-1)]
#define upqc(i,j) r.scratch[(ulong(col)*150+38+((j)-1))*(nl+1)+((i)-1)]
#define upqv(i,j) r.scratch[(ulong(col)*150+46+((j)-1))*(nl+1)+((i)-1)]
#define upa(i,j) r.scratch[(ulong(col)*150+54+((j)-1))*(nl+1)+((i)-1)]
#define upu(i,j) r.scratch[(ulong(col)*150+62+((j)-1))*(nl+1)+((i)-1)]
#define upv(i,j) r.scratch[(ulong(col)*150+70+((j)-1))*(nl+1)+((i)-1)]
#define upthv(i,j) r.scratch[(ulong(col)*150+78+((j)-1))*(nl+1)+((i)-1)]
#define upqke(i,j) r.scratch[(ulong(col)*150+86+((j)-1))*(nl+1)+((i)-1)]
#define upqnc(i,j) r.scratch[(ulong(col)*150+94+((j)-1))*(nl+1)+((i)-1)]
#define upqni(i,j) r.scratch[(ulong(col)*150+102+((j)-1))*(nl+1)+((i)-1)]
#define upqnwfa(i,j) r.scratch[(ulong(col)*150+110+((j)-1))*(nl+1)+((i)-1)]
#define upqnifa(i,j) r.scratch[(ulong(col)*150+118+((j)-1))*(nl+1)+((i)-1)]
#define upqnbca(i,j) r.scratch[(ulong(col)*150+126+((j)-1))*(nl+1)+((i)-1)]
#define ent(i,j) r.scratch[(ulong(col)*150+134+((j)-1))*(nl+1)+((i)-1)]
#define entf(i,j) r.scratch[(ulong(col)*150+142+((j)-1))*(nl+1)+((i)-1)]
#define enti(i,j) r.intscratch[(ulong(col)*8+((j)-1))*(nl+1)+((i)-1)]

constexpr int nup = 8;
constexpr int debug_mf = 0;
float nup2;
int k;
int i;
int k50;
float fltv2;
float wstar;
float qstar;
float thstar;
float sigmaw;
float sigmaqt;
float sigmath;
float z0;
float pwmin;
float pwmax;
float wmin;
float wmax;
float wlv;
float psig_w;
float maxw;
float maxqc;
float wpbl;
float b;
float qtn;
float thln;
float thvn;
float qcn;
float un;
float vn;
float qken;
float qncn;
float qnin;
float qnwfan;
float qnifan;
float qnbcan;
float wn2;
float wn;
float entexp;
float entexm;
float entw;
float bcoeff;
float thvkm1;
float thvk;
float pk;
float rho_int;
constexpr float wa = ((2.f) / (3.f));
constexpr float wb = 0.002f;
constexpr float wc = 1.5f;
constexpr float l0 = 100.f;
constexpr float ent0 = 0.1f;
constexpr float atot = 0.10f;
constexpr float lmax = 1000.f;
constexpr float lmin = 300.f;
constexpr float dlmin = 0.f;
float minwidth;
float dl;
constexpr float dcut = 1.2f;
float d;
float cn;
float c;
float l;
float n;
float an2;
float hux;
float wspd_pbl;
float cloud_base;
float width_flx;
int ic;
float erf;
bool superadiabatic;
float sigq;
float xl;
float rsl;
float cpm;
float a;
float qmq;
float mf_cf;
float aup;
float q1;
float diffqt;
float qsat_tk;
float fng;
float qww;
float alpha;
float beta;
float bb;
float f;
float pt;
float t;
float q2p;
float b9;
float satvp;
float rhgrid;
float ac_mf;
float ac_strat;
float qc_mf;
constexpr float cf_thresh = 0.5f;
float thp;
float qtp;
float qcp;
float qcs;
float esat;
float qsl;
float csigma;
float acfac;
float ac_wsp;
int overshoot;
float bvf;
float frz;
float dzp;
float adjustment;
float flx1;
constexpr float fluxportion = 0.75f;
float temp;
float sublim;
float qc_ent;
float qv_ent;
float qt_ent;
float thl_ent;
float detrate;
float detrateuv;
float oow;
float exc_fac;
float aratio;
float detturb;
float qc_grid;
float qc_sgs;
float qc_plume;
float exc_heat;
float exc_moist;
float tk_int;
float tvs;
constexpr float cdet = ((1.f) / (45.f));
constexpr float dzpmax = 300.f;
constexpr float csub = 0.25f;
constexpr float pgfac = 0.00f;
float uk;
float ukm1;
float vk;
float vkm1;
float dxsa;
// bl_mynn_subroutines.F90:4836
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upw((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4837
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upthl((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4838
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upthv((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4839
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upqt((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4840
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upa((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4841
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upu((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4842
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upv((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4843
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upqc((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4844
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upqv((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4845
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upqke((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4846
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upqnc((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4847
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upqni((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4848
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upqnwfa((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4849
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upqnifa((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4850
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upqnbca((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = 0.f; }
// bl_mynn_subroutines.F90:4851

// bl_mynn_subroutines.F90:4855
for (int mynn_element=0; mynn_element < (8*nl); ++mynn_element) { ent((mynn_element%nl)+1,(mynn_element/nl)+1) = 0.001f; }
// bl_mynn_subroutines.F90:4857
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { edmf_a(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4858
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { edmf_w(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4859
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { edmf_qt(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4860
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { edmf_thl(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4861
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { edmf_ent(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4862
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { edmf_qc(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4863

// bl_mynn_subroutines.F90:4868
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_aw(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4869
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awthl(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4870
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqt(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4871
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqv(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4872
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqc(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4873
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awu(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4874
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awv(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4875
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqke(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4876
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqnc(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4877
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqni(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4878
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqnwfa(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4879
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqnifa(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4880
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqnbca(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4881

// bl_mynn_subroutines.F90:4886
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { sub_thl(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4887
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { sub_sqv(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4888
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { sub_u(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4889
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { sub_v(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4890
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { det_thl(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4891
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { det_sqv(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4892
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { det_sqc(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4893
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { det_u(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4894
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { det_v(mynn_element+1) = 0.f; }
// bl_mynn_subroutines.F90:4895
nup2 = nup;
// bl_mynn_subroutines.F90:4899
maxw = 0.0f;
// bl_mynn_subroutines.F90:4900
cloud_base = 9000.0f;
// bl_mynn_subroutines.F90:4901
for (k=1; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:4902
if (soft32_greater(as_type<uint>(float(zw(k))),as_type<uint>(float(mpas_add(pblh, 500.f))))) { break; }
// bl_mynn_subroutines.F90:4904
wpbl = w(k);
// bl_mynn_subroutines.F90:4905
if (soft32_less(as_type<uint>(float(w(k))),as_type<uint>(float(0.f)))) { wpbl = mpas_mul(2.f, w(k)); }
// bl_mynn_subroutines.F90:4906
maxw = mynn_max(((float)(maxw)), ((float)(mynn_abs(wpbl))));
// bl_mynn_subroutines.F90:4909
if (soft32_less_equal(as_type<uint>(float(zw(k))),as_type<uint>(float(50.f)))) { k50 = k; }
// bl_mynn_subroutines.F90:4912
qc_sgs = mynn_max(((float)(qc(k))), ((float)(qc_bl1d(k))));
// bl_mynn_subroutines.F90:4913
if (((((soft32_greater(as_type<uint>(float(qc_sgs)),as_type<uint>(float(1e-5f)))) && (soft32_greater_equal(as_type<uint>(float(cldfra_bl1d(k))),as_type<uint>(float(0.5f)))))) && (soft32_equal(as_type<uint>(float(cloud_base)),as_type<uint>(float(9000.0f)))))) {
// bl_mynn_subroutines.F90:4914
cloud_base = mpas_mul(0.5f, mpas_add(zw(k), zw(((k) + (1)))));
// bl_mynn_subroutines.F90:4915
}
// bl_mynn_subroutines.F90:4916
}
// bl_mynn_subroutines.F90:4919
maxw = mynn_max(((float)(0.f)), ((float)(mpas_sub(maxw, 1.0f))));
// bl_mynn_subroutines.F90:4920
psig_w = mynn_max(((float)(0.0f)), ((float)(mpas_sub(1.0f, maxw))));
// bl_mynn_subroutines.F90:4921
psig_w = mynn_min(((float)(psig_w)), ((float)(psig_shcu)));
// bl_mynn_subroutines.F90:4924
fltv2 = fltv;
// bl_mynn_subroutines.F90:4925
if (((soft32_equal(as_type<uint>(float(psig_w)),as_type<uint>(float(0.0f)))) && (soft32_greater(as_type<uint>(float(fltv)),as_type<uint>(float(0.0f)))))) { fltv2 = mpas_mul(mynn_neg(1.f), fltv); }
// bl_mynn_subroutines.F90:4929
superadiabatic = false;
// bl_mynn_subroutines.F90:4930
if (soft32_greater_equal(as_type<uint>(float(mpas_sub(landsea, 1.5f))),as_type<uint>(float(0)))) {
// bl_mynn_subroutines.F90:4931
hux = mynn_neg(0.001f);
// bl_mynn_subroutines.F90:4932
} else {
// bl_mynn_subroutines.F90:4933
hux = mynn_neg(0.005f);
// bl_mynn_subroutines.F90:4934
}
// bl_mynn_subroutines.F90:4935
tvs = mpas_mul(ts, mpas_add(1.0f, mpas_mul(p608, qv(kts))));
// bl_mynn_subroutines.F90:4936
for (k=1; (1) > 0 ? k <= (mynn_max(((int)(1)), ((int)(((k50) - (1)))))) : k >= (mynn_max(((int)(1)), ((int)(((k50) - (1)))))); k += (1)) {
// bl_mynn_subroutines.F90:4937
if (((k) == (1))) {
// bl_mynn_subroutines.F90:4938
if (soft32_less(as_type<uint>(float(mpas_div(mpas_sub(thv(k), tvs), mpas_mul(0.5f, dz(k))))),as_type<uint>(float(hux)))) {
// bl_mynn_subroutines.F90:4939
superadiabatic = true;
// bl_mynn_subroutines.F90:4940
} else {
// bl_mynn_subroutines.F90:4941
superadiabatic = false;
// bl_mynn_subroutines.F90:4942
break;
// bl_mynn_subroutines.F90:4943
}
// bl_mynn_subroutines.F90:4944
} else {
// bl_mynn_subroutines.F90:4945
if (soft32_less(as_type<uint>(float(mpas_div(mpas_sub(thv(k), thv(((k) - (1)))), mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))))))),as_type<uint>(float(hux)))) {
// bl_mynn_subroutines.F90:4946
superadiabatic = true;
// bl_mynn_subroutines.F90:4947
} else {
// bl_mynn_subroutines.F90:4948
superadiabatic = false;
// bl_mynn_subroutines.F90:4949
break;
// bl_mynn_subroutines.F90:4950
}
// bl_mynn_subroutines.F90:4951
}
// bl_mynn_subroutines.F90:4952
}
// bl_mynn_subroutines.F90:4963
maxwidth = mynn_min(((float)(mpas_mul(dx, dcut))), ((float)(lmax)));
// bl_mynn_subroutines.F90:4965
maxwidth = mynn_min(((float)(maxwidth)), ((float)(mpas_mul(1.1f, pblh))));
// bl_mynn_subroutines.F90:4967
if (soft32_less(as_type<uint>(float(mpas_sub(landsea, 1.5f))),as_type<uint>(float(0)))) {
// bl_mynn_subroutines.F90:4968
maxwidth = mynn_min(((float)(maxwidth)), ((float)(mpas_mul(0.5f, cloud_base))));
// bl_mynn_subroutines.F90:4969
} else {
// bl_mynn_subroutines.F90:4970
maxwidth = mynn_min(((float)(maxwidth)), ((float)(mpas_mul(0.9f, cloud_base))));
// bl_mynn_subroutines.F90:4971
}
// bl_mynn_subroutines.F90:4973
wspd_pbl = mynn_sqrt(mynn_max(((float)(mpas_add(mynn_square(u(kts)), mynn_square(v(kts))))), ((float)(0.01f))));
// bl_mynn_subroutines.F90:4976
if (soft32_less(as_type<uint>(float(mpas_sub(landsea, 1.5f))),as_type<uint>(float(0)))) {
// bl_mynn_subroutines.F90:4977
width_flx = mynn_max(((float)(mynn_min(((float)(mpas_mul(1000.f, mpas_add(mpas_mul(0.6f, mynn_tanh(mpas_div(mpas_sub(fltv, 0.040f), 0.04f))), .5f)))), ((float)(1000.f))))), ((float)(0.f)));
// bl_mynn_subroutines.F90:4978
} else {
// bl_mynn_subroutines.F90:4979
width_flx = mynn_max(((float)(mynn_min(((float)(mpas_mul(1000.f, mpas_add(mpas_mul(0.6f, mynn_tanh(mpas_div(mpas_sub(fltv, 0.007f), 0.02f))), .5f)))), ((float)(1000.f))))), ((float)(0.f)));
// bl_mynn_subroutines.F90:4980
}
// bl_mynn_subroutines.F90:4981
maxwidth = mynn_min(((float)(maxwidth)), ((float)(width_flx)));
// bl_mynn_subroutines.F90:4982
minwidth = lmin;
// bl_mynn_subroutines.F90:4985
if (((soft32_greater_equal(as_type<uint>(float(maxwidth)),as_type<uint>(float(mpas_sub(lmax, 1.0f))))) && (soft32_greater(as_type<uint>(float(fltv)),as_type<uint>(float(0.2f)))))) { minwidth = mpas_add(lmin, mpas_mul(dlmin, mynn_min(((float)(mpas_div(mpas_sub(fltv, 0.2f), 0.3f))), ((float)(1.f))))); }
// bl_mynn_subroutines.F90:4987
if (soft32_less_equal(as_type<uint>(float(maxwidth)),as_type<uint>(float(minwidth)))) {
// bl_mynn_subroutines.F90:4988
nup2 = 0;
// bl_mynn_subroutines.F90:4989
maxwidth = 0.0f;
// bl_mynn_subroutines.F90:4990
}
// bl_mynn_subroutines.F90:4993
ktop = 0;
// bl_mynn_subroutines.F90:4994
ztop = 0.0f;
// bl_mynn_subroutines.F90:4995
maxmf = 0.0f;
// bl_mynn_subroutines.F90:4998
if (((((soft32_greater(as_type<uint>(float(fltv2)),as_type<uint>(float(0.002f)))) && (soft32_greater(as_type<uint>(float(maxwidth)),as_type<uint>(float(minwidth)))))) && (superadiabatic))) {
// bl_mynn_subroutines.F90:5001
cn = 0.f;
// bl_mynn_subroutines.F90:5002
d = mynn_neg(1.9f);
// bl_mynn_subroutines.F90:5003
dl = mpas_div(mpas_sub(maxwidth, minwidth), ((float)(((nup) - (1)))));
// bl_mynn_subroutines.F90:5004
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5006
l = mpas_add(minwidth, mpas_mul(dl, ((float)(((i) - (1))))));
// bl_mynn_subroutines.F90:5007
cn = mpas_add(cn, mpas_mul(mpas_div(mpas_mul(mynn_pow(((float)(l)), ((float)(d))), mpas_mul(l, l)), mpas_mul(dx, dx)), dl));
// bl_mynn_subroutines.F90:5008
}
// bl_mynn_subroutines.F90:5009
c = mpas_div(atot, cn);
// bl_mynn_subroutines.F90:5012
if (soft32_less(as_type<uint>(float(mpas_sub(landsea, 1.5f))),as_type<uint>(float(0)))) {
// bl_mynn_subroutines.F90:5013
acfac = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(fltv2, 0.02f), 0.05f))), .5f);
// bl_mynn_subroutines.F90:5014
} else {
// bl_mynn_subroutines.F90:5015
acfac = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(fltv2, 0.01f), 0.03f))), .5f);
// bl_mynn_subroutines.F90:5016
}
// bl_mynn_subroutines.F90:5021
if (soft32_less_equal(as_type<uint>(float(wspd_pbl)),as_type<uint>(float(10.f)))) {
// bl_mynn_subroutines.F90:5022
ac_wsp = 1.0f;
// bl_mynn_subroutines.F90:5023
} else {
// bl_mynn_subroutines.F90:5024
ac_wsp = mpas_sub(1.0f, mynn_min(((float)(mpas_div(mpas_sub(wspd_pbl, 10.0f), 15.f))), ((float)(1.0f))));
// bl_mynn_subroutines.F90:5025
}
// bl_mynn_subroutines.F90:5026
acfac = mpas_mul(acfac, ac_wsp);
// bl_mynn_subroutines.F90:5029
an2 = 0.f;
// bl_mynn_subroutines.F90:5030
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5032
l = mpas_add(minwidth, mpas_mul(dl, ((float)(((i) - (1))))));
// bl_mynn_subroutines.F90:5033
n = mpas_mul(c, mynn_pow(((float)(l)), ((float)(d))));
// bl_mynn_subroutines.F90:5034
upa(1, i) = mpas_mul(mpas_div(mpas_mul(mpas_mul(n, l), l), mpas_mul(dx, dx)), dl);
// bl_mynn_subroutines.F90:5036
upa(1, i) = mpas_mul(upa(1, i), acfac);
// bl_mynn_subroutines.F90:5037
an2 = mpas_add(an2, upa(1, i));
// bl_mynn_subroutines.F90:5039
}
// bl_mynn_subroutines.F90:5042
z0 = 50.f;
// bl_mynn_subroutines.F90:5043
pwmin = 0.1f;
// bl_mynn_subroutines.F90:5044
pwmax = 0.4f;
// bl_mynn_subroutines.F90:5046
wstar = mynn_max(((float)(1.e-2f)), ((float)(mynn_pow(((float)(mpas_mul(mpas_mul(gtr, fltv2), pblh))), ((float)(onethird))))));
// bl_mynn_subroutines.F90:5047
qstar = mpas_div(mynn_max(((float)(flq)), ((float)(1.0e-5f))), wstar);
// bl_mynn_subroutines.F90:5048
thstar = mpas_div(flt, wstar);
// bl_mynn_subroutines.F90:5050
if (soft32_greater_equal(as_type<uint>(float(mpas_sub(landsea, 1.5f))),as_type<uint>(float(0)))) {
// bl_mynn_subroutines.F90:5051
csigma = 1.34f;
// bl_mynn_subroutines.F90:5052
} else {
// bl_mynn_subroutines.F90:5053
csigma = 1.34f;
// bl_mynn_subroutines.F90:5054
}
// bl_mynn_subroutines.F90:5056
if (env_subs) {
// bl_mynn_subroutines.F90:5057
exc_fac = 0.0f;
// bl_mynn_subroutines.F90:5058
} else {
// bl_mynn_subroutines.F90:5059
if (soft32_greater_equal(as_type<uint>(float(mpas_sub(landsea, 1.5f))),as_type<uint>(float(0)))) {
// bl_mynn_subroutines.F90:5061
exc_fac = mpas_mul(0.58f, 4.0f);
// bl_mynn_subroutines.F90:5062
} else {
// bl_mynn_subroutines.F90:5064
exc_fac = 0.58f;
// bl_mynn_subroutines.F90:5065
}
// bl_mynn_subroutines.F90:5066
}
// bl_mynn_subroutines.F90:5068
exc_fac = mpas_mul(exc_fac, ac_wsp);
// bl_mynn_subroutines.F90:5071
sigmaw = mpas_mul(mpas_mul(mpas_mul(csigma, wstar), mynn_pow(((float)(mpas_div(z0, pblh))), ((float)(onethird)))), mpas_sub(1, mpas_div(mpas_mul(0.8f, z0), pblh)));
// bl_mynn_subroutines.F90:5072
sigmaqt = mpas_mul(mpas_mul(csigma, qstar), mynn_pow(((float)(mpas_div(z0, pblh))), ((float)(onethird))));
// bl_mynn_subroutines.F90:5073
sigmath = mpas_mul(mpas_mul(csigma, thstar), mynn_pow(((float)(mpas_div(z0, pblh))), ((float)(onethird))));
// bl_mynn_subroutines.F90:5077
wmin = mynn_min(((float)(mpas_mul(sigmaw, pwmin))), ((float)(0.1f)));
// bl_mynn_subroutines.F90:5078
wmax = mynn_min(((float)(mpas_mul(sigmaw, pwmax))), ((float)(0.5f)));
// bl_mynn_subroutines.F90:5081
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5082
wlv = mpas_add(wmin, mpas_mul(mpas_div(mpas_sub(wmax, wmin), nup2), ((i) - (1))));
// bl_mynn_subroutines.F90:5085
upw(1, i) = mpas_add(wmin, mpas_mul(mpas_div(((float)(i)), ((float)(nup))), mpas_sub(wmax, wmin)));
// bl_mynn_subroutines.F90:5086
upu(1, i) = mpas_div(mpas_add(mpas_mul(u(kts), dz(((kts) + (1)))), mpas_mul(u(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5087
upv(1, i) = mpas_div(mpas_add(mpas_mul(v(kts), dz(((kts) + (1)))), mpas_mul(v(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5088
upqc(1, i) = 0.0f;
// bl_mynn_subroutines.F90:5091
exc_heat = mpas_div(mpas_mul(mpas_mul(exc_fac, upw(1, i)), sigmath), sigmaw);
// bl_mynn_subroutines.F90:5092
upthv(1, i) = mpas_add(mpas_div(mpas_add(mpas_mul(thv(kts), dz(((kts) + (1)))), mpas_mul(thv(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1))))), exc_heat);
// bl_mynn_subroutines.F90:5094
upthl(1, i) = mpas_add(mpas_div(mpas_add(mpas_mul(thl(kts), dz(((kts) + (1)))), mpas_mul(thl(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1))))), exc_heat);
// bl_mynn_subroutines.F90:5098
exc_moist = mpas_div(mpas_mul(mpas_mul(exc_fac, upw(1, i)), sigmaqt), sigmaw);
// bl_mynn_subroutines.F90:5099
upqt(1, i) = mpas_add(mpas_div(mpas_add(mpas_mul(qt(kts), dz(((kts) + (1)))), mpas_mul(qt(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1))))), exc_moist);
// bl_mynn_subroutines.F90:5102
upqke(1, i) = mpas_div(mpas_add(mpas_mul(qke(kts), dz(((kts) + (1)))), mpas_mul(qke(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5103
upqnc(1, i) = mpas_div(mpas_add(mpas_mul(qnc(kts), dz(((kts) + (1)))), mpas_mul(qnc(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5104
upqni(1, i) = mpas_div(mpas_add(mpas_mul(qni(kts), dz(((kts) + (1)))), mpas_mul(qni(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5105
upqnwfa(1, i) = mpas_div(mpas_add(mpas_mul(qnwfa(kts), dz(((kts) + (1)))), mpas_mul(qnwfa(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5106
upqnifa(1, i) = mpas_div(mpas_add(mpas_mul(qnifa(kts), dz(((kts) + (1)))), mpas_mul(qnifa(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5107
upqnbca(1, i) = mpas_div(mpas_add(mpas_mul(qnbca(kts), dz(((kts) + (1)))), mpas_mul(qnbca(((kts) + (1))), dz(kts))), mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5108
}
// bl_mynn_subroutines.F90:5110

// bl_mynn_subroutines.F90:5119
for(int mynn_element=1;mynn_element<=nl;++mynn_element) { envm_thl(mynn_element)=thl(mynn_element); }
// bl_mynn_subroutines.F90:5120
for(int mynn_element=1;mynn_element<=nl;++mynn_element) { envm_sqv(mynn_element)=qv(mynn_element); }
// bl_mynn_subroutines.F90:5121
for(int mynn_element=1;mynn_element<=nl;++mynn_element) { envm_sqc(mynn_element)=qc(mynn_element); }
// bl_mynn_subroutines.F90:5122
for(int mynn_element=1;mynn_element<=nl;++mynn_element) { envm_u(mynn_element)=u(mynn_element); }
// bl_mynn_subroutines.F90:5123
for(int mynn_element=1;mynn_element<=nl;++mynn_element) { envm_v(mynn_element)=v(mynn_element); }
// bl_mynn_subroutines.F90:5124
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5125
rhoz(k) = mpas_div(mpas_add(mpas_mul(rho(k), dz(((k) + (1)))), mpas_mul(rho(((k) + (1))), dz(k))), mpas_add(dz(((k) + (1))), dz(k)));
// bl_mynn_subroutines.F90:5126
}
// bl_mynn_subroutines.F90:5127
rhoz(kte) = rho(kte);
// bl_mynn_subroutines.F90:5130
dxsa = mpas_sub(1.f, mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_sub(12000.0f, dx), mpas_sub(12000.0f, 3000.0f)))), ((float)(0.f))))), ((float)(1.f))));
// bl_mynn_subroutines.F90:5133
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5134
qcn = 0.f;
// bl_mynn_subroutines.F90:5135
overshoot = 0;
// bl_mynn_subroutines.F90:5136
l = mpas_add(minwidth, mpas_mul(dl, ((float)(((i) - (1))))));
// bl_mynn_subroutines.F90:5137
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5140
wmin = mpas_add(0.3f, mpas_mul(l, 0.0005f));
// bl_mynn_subroutines.F90:5141
ent(k, i) = mpas_div(0.33f, mpas_mul(mynn_min(((float)(mynn_max(((float)(upw(((k) - (1)), i))), ((float)(wmin))))), ((float)(0.9f))), l));
// bl_mynn_subroutines.F90:5149
ent(k, i) = mynn_max(((float)(ent(k, i))), ((float)(0.0003f)));
// bl_mynn_subroutines.F90:5153
if (soft32_greater_equal(as_type<uint>(float(zw(k))),as_type<uint>(float(mynn_min(((float)(mpas_add(pblh, 1500.f))), ((float)(4000.f))))))) {
// bl_mynn_subroutines.F90:5154
ent(k, i) = mpas_add(ent(k, i), mpas_mul(mpas_sub(zw(k), mynn_min(((float)(mpas_add(pblh, 1500.f))), ((float)(4000.f)))), 5.0e-6f));
// bl_mynn_subroutines.F90:5155
}
// bl_mynn_subroutines.F90:5158
ent(k, i) = mpas_mul(ent(k, i), mpas_sub(1.0f, rstoch_col(k)));
// bl_mynn_subroutines.F90:5160
ent(k, i) = mynn_min(((float)(ent(k, i))), ((float)(mpas_div(0.9f, mpas_sub(zw(((k) + (1))), zw(k))))));
// bl_mynn_subroutines.F90:5163
uk = mpas_div(mpas_add(mpas_mul(u(k), dz(((k) + (1)))), mpas_mul(u(((k) + (1))), dz(k))), mpas_add(dz(((k) + (1))), dz(k)));
// bl_mynn_subroutines.F90:5164
ukm1 = mpas_div(mpas_add(mpas_mul(u(((k) - (1))), dz(k)), mpas_mul(u(k), dz(((k) - (1))))), mpas_add(dz(((k) - (1))), dz(k)));
// bl_mynn_subroutines.F90:5165
vk = mpas_div(mpas_add(mpas_mul(v(k), dz(((k) + (1)))), mpas_mul(v(((k) + (1))), dz(k))), mpas_add(dz(((k) + (1))), dz(k)));
// bl_mynn_subroutines.F90:5166
vkm1 = mpas_div(mpas_add(mpas_mul(v(((k) - (1))), dz(k)), mpas_mul(v(k), dz(((k) - (1))))), mpas_add(dz(((k) - (1))), dz(k)));
// bl_mynn_subroutines.F90:5169
entexp = mpas_mul(ent(k, i), mpas_sub(zw(((k) + (1))), zw(k)));
// bl_mynn_subroutines.F90:5170
entexm = mpas_mul(entexp, 0.3333f);
// bl_mynn_subroutines.F90:5171
qtn = mpas_add(mpas_mul(upqt(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qt(k), entexp));
// bl_mynn_subroutines.F90:5172
thln = mpas_add(mpas_mul(upthl(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(thl(k), entexp));
// bl_mynn_subroutines.F90:5173
un = mpas_add(mpas_add(mpas_mul(upu(((k) - (1)), i), mpas_sub(1.f, entexm)), mpas_mul(u(k), entexm)), mpas_mul(mpas_mul(dxsa, pgfac), mpas_sub(uk, ukm1)));
// bl_mynn_subroutines.F90:5174
vn = mpas_add(mpas_add(mpas_mul(upv(((k) - (1)), i), mpas_sub(1.f, entexm)), mpas_mul(v(k), entexm)), mpas_mul(mpas_mul(dxsa, pgfac), mpas_sub(vk, vkm1)));
// bl_mynn_subroutines.F90:5175
qken = mpas_add(mpas_mul(upqke(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qke(k), entexp));
// bl_mynn_subroutines.F90:5176
qncn = mpas_add(mpas_mul(upqnc(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qnc(k), entexp));
// bl_mynn_subroutines.F90:5177
qnin = mpas_add(mpas_mul(upqni(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qni(k), entexp));
// bl_mynn_subroutines.F90:5178
qnwfan = mpas_add(mpas_mul(upqnwfa(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qnwfa(k), entexp));
// bl_mynn_subroutines.F90:5179
qnifan = mpas_add(mpas_mul(upqnifa(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qnifa(k), entexp));
// bl_mynn_subroutines.F90:5180
qnbcan = mpas_add(mpas_mul(upqnbca(((k) - (1)), i), mpas_sub(1.f, entexp)), mpas_mul(qnbca(k), entexp));
// bl_mynn_subroutines.F90:5184
qc_ent = qcn;
// bl_mynn_subroutines.F90:5185
qt_ent = qtn;
// bl_mynn_subroutines.F90:5186
thl_ent = thln;
// bl_mynn_subroutines.F90:5196

// bl_mynn_subroutines.F90:5206
pk = mpas_div(mpas_add(mpas_mul(p(k), dz(((k) + (1)))), mpas_mul(p(((k) + (1))), dz(k))), mpas_add(dz(((k) + (1))), dz(k)));
// bl_mynn_subroutines.F90:5208
f_condensation_edmf(qtn,thln,pk,zw(((k)+(1))),thvn,qcn,p1000mb,rcp,xlvcp,t0c,rvovrd);
// bl_mynn_subroutines.F90:5211
thvk = mpas_div(mpas_add(mpas_mul(thv(k), dz(((k) + (1)))), mpas_mul(thv(((k) + (1))), dz(k))), mpas_add(dz(((k) + (1))), dz(k)));
// bl_mynn_subroutines.F90:5212
thvkm1 = mpas_div(mpas_add(mpas_mul(thv(((k) - (1))), dz(k)), mpas_mul(thv(k), dz(((k) - (1))))), mpas_add(dz(((k) - (1))), dz(k)));
// bl_mynn_subroutines.F90:5215
b = mpas_mul(grav, mpas_sub(mpas_div(thvn, thvk), 1.0f));
// bl_mynn_subroutines.F90:5216
if (soft32_greater(as_type<uint>(float(b)),as_type<uint>(float(0.f)))) {
// bl_mynn_subroutines.F90:5217
bcoeff = 0.15f;
// bl_mynn_subroutines.F90:5218
} else {
// bl_mynn_subroutines.F90:5219
bcoeff = 0.2f;
// bl_mynn_subroutines.F90:5220
}
// bl_mynn_subroutines.F90:5230
if (soft32_less(as_type<uint>(float(upw(((k) - (1)), i))),as_type<uint>(float(0.2f)))) {
// bl_mynn_subroutines.F90:5231
wn = mpas_add(upw(((k) - (1)), i), mpas_mul(mpas_add(mpas_mul(mpas_mul(mynn_neg(2.f), ent(k, i)), upw(((k) - (1)), i)), mpas_div(mpas_mul(bcoeff, b), mynn_max(((float)(upw(((k) - (1)), i))), ((float)(0.2f))))), mynn_min(((float)(mpas_sub(zw(k), zw(((k) - (1)))))), ((float)(250.f)))));
// bl_mynn_subroutines.F90:5232
} else {
// bl_mynn_subroutines.F90:5233
wn = mpas_add(upw(((k) - (1)), i), mpas_mul(mpas_add(mpas_mul(mpas_mul(mynn_neg(2.f), ent(k, i)), upw(((k) - (1)), i)), mpas_div(mpas_mul(bcoeff, b), upw(((k) - (1)), i))), mynn_min(((float)(mpas_sub(zw(k), zw(((k) - (1)))))), ((float)(250.f)))));
// bl_mynn_subroutines.F90:5234
}
// bl_mynn_subroutines.F90:5237
if (soft32_greater(as_type<uint>(float(wn)),as_type<uint>(float(mpas_add(upw(((k) - (1)), i), mynn_min(((float)(mpas_div(mpas_mul(1.25f, mpas_sub(zw(k), zw(((k) - (1))))), 200.f))), ((float)(2.0f)))))))) {
// bl_mynn_subroutines.F90:5238
wn = mpas_add(upw(((k) - (1)), i), mynn_min(((float)(mpas_div(mpas_mul(1.25f, mpas_sub(zw(k), zw(((k) - (1))))), 200.f))), ((float)(2.0f))));
// bl_mynn_subroutines.F90:5239
}
// bl_mynn_subroutines.F90:5241
if (soft32_less(as_type<uint>(float(wn)),as_type<uint>(float(mpas_sub(upw(((k) - (1)), i), mynn_min(((float)(mpas_div(mpas_mul(1.25f, mpas_sub(zw(k), zw(((k) - (1))))), 200.f))), ((float)(2.0f)))))))) {
// bl_mynn_subroutines.F90:5242
wn = mpas_sub(upw(((k) - (1)), i), mynn_min(((float)(mpas_div(mpas_mul(1.25f, mpas_sub(zw(k), zw(((k) - (1))))), 200.f))), ((float)(2.0f))));
// bl_mynn_subroutines.F90:5243
}
// bl_mynn_subroutines.F90:5244
wn = mynn_min(((float)(mynn_max(((float)(wn)), ((float)(0.0f))))), ((float)(3.0f)));
// bl_mynn_subroutines.F90:5248
if (((((k) == (((kts) + (1))))) && (soft32_equal(as_type<uint>(float(wn)),as_type<uint>(float(0.f)))))) {
// bl_mynn_subroutines.F90:5249
nup2 = 0;
// bl_mynn_subroutines.F90:5250
break;
// bl_mynn_subroutines.F90:5251
}
// bl_mynn_subroutines.F90:5253

// bl_mynn_subroutines.F90:5264
if (((soft32_less_equal(as_type<uint>(float(wn)),as_type<uint>(float(0.0f)))) && (((overshoot) == (0))))) {
// bl_mynn_subroutines.F90:5265
overshoot = 1;
// bl_mynn_subroutines.F90:5266
if (soft32_greater(as_type<uint>(float(mpas_sub(thvk, thvkm1))),as_type<uint>(float(0.0f)))) {
// bl_mynn_subroutines.F90:5267
bvf = mynn_sqrt(mpas_div(mpas_mul(gtr, mpas_sub(thvk, thvkm1)), dz(k)));
// bl_mynn_subroutines.F90:5269
frz = mpas_div(upw(((k) - (1)), i), mpas_mul(bvf, dz(k)));
// bl_mynn_subroutines.F90:5271
dzp = mpas_mul(dz(k), mynn_max(((float)(mynn_min(((float)(frz)), ((float)(1.0f))))), ((float)(0.0f))));
// bl_mynn_subroutines.F90:5272
}
// bl_mynn_subroutines.F90:5273
} else {
// bl_mynn_subroutines.F90:5274
dzp = dz(k);
// bl_mynn_subroutines.F90:5275
}
// bl_mynn_subroutines.F90:5285
aratio = mynn_min(((float)(mpas_div(upa(((k) - (1)), i), mpas_sub(1.f, upa(((k) - (1)), i))))), ((float)(0.5f)));
// bl_mynn_subroutines.F90:5286
detturb = 0.00008f;
// bl_mynn_subroutines.F90:5287
oow = mpas_div(mynn_neg(0.060f), mynn_max(((float)(1.0f)), ((float)(mpas_mul(0.5f, mpas_add(wn, upw(((k) - (1)), i)))))));
// bl_mynn_subroutines.F90:5288
detrate = mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_mul(oow, mpas_sub(wn, upw(((k) - (1)), i))), dz(k)))), ((float)(detturb))))), ((float)(.0002f)));
// bl_mynn_subroutines.F90:5289
detrateuv = mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_mul(oow, mpas_sub(wn, upw(((k) - (1)), i))), dz(k)))), ((float)(detturb))))), ((float)(.0001f)));
// bl_mynn_subroutines.F90:5290
envm_thl(k) = mpas_add(envm_thl(k), mpas_mul(mpas_mul(mpas_mul(mpas_sub(mpas_mul(0.5f, mpas_add(thl_ent, upthl(((k) - (1)), i))), thl(k)), detrate), aratio), mynn_min(((float)(dzp)), ((float)(dzpmax)))));
// bl_mynn_subroutines.F90:5291
qv_ent = mpas_mul(0.5f, mpas_add(mynn_max(((float)(mpas_sub(qt_ent, qc_ent))), ((float)(0.f))), mynn_max(((float)(mpas_sub(upqt(((k) - (1)), i), upqc(((k) - (1)), i)))), ((float)(0.f)))));
// bl_mynn_subroutines.F90:5292
envm_sqv(k) = mpas_add(envm_sqv(k), mpas_mul(mpas_mul(mpas_mul(mpas_sub(qv_ent, qv(k)), detrate), aratio), mynn_min(((float)(dzp)), ((float)(dzpmax)))));
// bl_mynn_subroutines.F90:5293
if (soft32_greater(as_type<uint>(float(upqc(((k) - (1)), i))),as_type<uint>(float(1e-8f)))) {
// bl_mynn_subroutines.F90:5294
if (soft32_greater(as_type<uint>(float(qc(k))),as_type<uint>(float(1e-6f)))) {
// bl_mynn_subroutines.F90:5295
qc_grid = qc(k);
// bl_mynn_subroutines.F90:5296
} else {
// bl_mynn_subroutines.F90:5297
qc_grid = mpas_mul(cldfra_bl1d(k), qc_bl1d(k));
// bl_mynn_subroutines.F90:5298
}
// bl_mynn_subroutines.F90:5299
envm_sqc(k) = mpas_add(envm_sqc(k), mpas_mul(mpas_mul(mpas_mul(mynn_max(((float)(mpas_sub(mpas_mul(mpas_mul(upa(((k) - (1)), i), 0.5f), mpas_add(qcn, upqc(((k) - (1)), i))), qc_grid))), ((float)(0.0f))), detrate), aratio), mynn_min(((float)(dzp)), ((float)(dzpmax)))));
// bl_mynn_subroutines.F90:5300
}
// bl_mynn_subroutines.F90:5301
envm_u(k) = mpas_add(envm_u(k), mpas_mul(mpas_mul(mpas_mul(mpas_sub(mpas_mul(0.5f, mpas_add(un, upu(((k) - (1)), i))), u(k)), detrateuv), aratio), mynn_min(((float)(dzp)), ((float)(dzpmax)))));
// bl_mynn_subroutines.F90:5302
envm_v(k) = mpas_add(envm_v(k), mpas_mul(mpas_mul(mpas_mul(mpas_sub(mpas_mul(0.5f, mpas_add(vn, upv(((k) - (1)), i))), v(k)), detrateuv), aratio), mynn_min(((float)(dzp)), ((float)(dzpmax)))));
// bl_mynn_subroutines.F90:5304
if (soft32_greater(as_type<uint>(float(wn)),as_type<uint>(float(0.f)))) {
// bl_mynn_subroutines.F90:5306
upw(k, i) = wn;
// bl_mynn_subroutines.F90:5307
upthv(k, i) = thvn;
// bl_mynn_subroutines.F90:5308
upthl(k, i) = thln;
// bl_mynn_subroutines.F90:5309
upqt(k, i) = qtn;
// bl_mynn_subroutines.F90:5310
upqc(k, i) = qcn;
// bl_mynn_subroutines.F90:5311
upu(k, i) = un;
// bl_mynn_subroutines.F90:5312
upv(k, i) = vn;
// bl_mynn_subroutines.F90:5313
upqke(k, i) = qken;
// bl_mynn_subroutines.F90:5314
upqnc(k, i) = qncn;
// bl_mynn_subroutines.F90:5315
upqni(k, i) = qnin;
// bl_mynn_subroutines.F90:5316
upqnwfa(k, i) = qnwfan;
// bl_mynn_subroutines.F90:5317
upqnifa(k, i) = qnifan;
// bl_mynn_subroutines.F90:5318
upqnbca(k, i) = qnbcan;
// bl_mynn_subroutines.F90:5319
upa(k, i) = upa(((k) - (1)), i);
// bl_mynn_subroutines.F90:5320

// bl_mynn_subroutines.F90:5325
ktop = mynn_max(((int)(ktop)), ((int)(k)));
// bl_mynn_subroutines.F90:5326
} else {
// bl_mynn_subroutines.F90:5327
break;
// bl_mynn_subroutines.F90:5328
}
// bl_mynn_subroutines.F90:5329
}
// bl_mynn_subroutines.F90:5331

// bl_mynn_subroutines.F90:5349
}
// bl_mynn_subroutines.F90:5350
} else {
// bl_mynn_subroutines.F90:5352
nup2 = 0.f;
// bl_mynn_subroutines.F90:5353
}
// bl_mynn_subroutines.F90:5355
ktop = mynn_min(((int)(ktop)), ((int)(((kte) - (1)))));
// bl_mynn_subroutines.F90:5356
if (((ktop) == (0))) {
// bl_mynn_subroutines.F90:5357
ztop = 0.0f;
// bl_mynn_subroutines.F90:5358
} else {
// bl_mynn_subroutines.F90:5359
ztop = zw(ktop);
// bl_mynn_subroutines.F90:5360
}
// bl_mynn_subroutines.F90:5362
if (soft32_greater(as_type<uint>(float(nup2)),as_type<uint>(float(0)))) {
// bl_mynn_subroutines.F90:5365
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5366
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5367
s_aw(((k) + (1))) = mpas_add(s_aw(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), psig_w));
// bl_mynn_subroutines.F90:5368
s_awthl(((k) + (1))) = mpas_add(s_awthl(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upthl(k, i)), psig_w));
// bl_mynn_subroutines.F90:5369
s_awqt(((k) + (1))) = mpas_add(s_awqt(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqt(k, i)), psig_w));
// bl_mynn_subroutines.F90:5374
qc_plume = upqc(k, i);
// bl_mynn_subroutines.F90:5378
s_awqc(((k) + (1))) = mpas_add(s_awqc(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), qc_plume), psig_w));
// bl_mynn_subroutines.F90:5379
s_awqv(((k) + (1))) = mpas_sub(s_awqt(((k) + (1))), s_awqc(((k) + (1))));
// bl_mynn_subroutines.F90:5380
}
// bl_mynn_subroutines.F90:5381
}
// bl_mynn_subroutines.F90:5383
if (momentum_opt) {
// bl_mynn_subroutines.F90:5384
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5385
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5386
s_awu(((k) + (1))) = mpas_add(s_awu(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upu(k, i)), psig_w));
// bl_mynn_subroutines.F90:5387
s_awv(((k) + (1))) = mpas_add(s_awv(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upv(k, i)), psig_w));
// bl_mynn_subroutines.F90:5388
}
// bl_mynn_subroutines.F90:5389
}
// bl_mynn_subroutines.F90:5390
}
// bl_mynn_subroutines.F90:5392
if (tke_opt) {
// bl_mynn_subroutines.F90:5393
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5394
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5395
s_awqke(((k) + (1))) = mpas_add(s_awqke(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqke(k, i)), psig_w));
// bl_mynn_subroutines.F90:5396
}
// bl_mynn_subroutines.F90:5397
}
// bl_mynn_subroutines.F90:5398
}
// bl_mynn_subroutines.F90:5400

// bl_mynn_subroutines.F90:5410
if (scalar_opt) {
// bl_mynn_subroutines.F90:5411
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:5412
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5413
s_awqnc(((k) + (1))) = mpas_add(s_awqnc(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqnc(k, i)), psig_w));
// bl_mynn_subroutines.F90:5414
s_awqni(((k) + (1))) = mpas_add(s_awqni(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqni(k, i)), psig_w));
// bl_mynn_subroutines.F90:5415
s_awqnwfa(((k) + (1))) = mpas_add(s_awqnwfa(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqnwfa(k, i)), psig_w));
// bl_mynn_subroutines.F90:5416
s_awqnifa(((k) + (1))) = mpas_add(s_awqnifa(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqnifa(k, i)), psig_w));
// bl_mynn_subroutines.F90:5417
s_awqnbca(((k) + (1))) = mpas_add(s_awqnbca(((k) + (1))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)), upqnbca(k, i)), psig_w));
// bl_mynn_subroutines.F90:5418
}
// bl_mynn_subroutines.F90:5419
}
// bl_mynn_subroutines.F90:5420
}
// bl_mynn_subroutines.F90:5425
if ((!soft32_equal(as_type<uint>(float(s_aw(((kts) + (1))))),as_type<uint>(float(0.f))))) {
// bl_mynn_subroutines.F90:5426
dzi(kts) = mpas_mul(0.5f, mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5427
flx1 = mynn_max(((float)(mpas_div(mpas_mul(s_aw(((kts) + (1))), mpas_sub(th(kts), th(((kts) + (1))))), dzi(kts)))), ((float)(1.0e-5f)));
// bl_mynn_subroutines.F90:5428
} else {
// bl_mynn_subroutines.F90:5429
flx1 = 0.0f;
// bl_mynn_subroutines.F90:5432
}
// bl_mynn_subroutines.F90:5433
adjustment = 1.0f;
// bl_mynn_subroutines.F90:5436
if (((soft32_greater(as_type<uint>(float(flx1)),as_type<uint>(float(mpas_div(mpas_mul(fluxportion, flt), dz(kts)))))) && (soft32_greater(as_type<uint>(float(flx1)),as_type<uint>(float(0.0f)))))) {
// bl_mynn_subroutines.F90:5437
adjustment = mpas_div(mpas_div(mpas_mul(fluxportion, flt), dz(kts)), flx1);
// bl_mynn_subroutines.F90:5438
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_aw(mynn_element+1) = mpas_mul(s_aw(mynn_element+1), adjustment); }
// bl_mynn_subroutines.F90:5439
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awthl(mynn_element+1) = mpas_mul(s_awthl(mynn_element+1), adjustment); }
// bl_mynn_subroutines.F90:5440
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqt(mynn_element+1) = mpas_mul(s_awqt(mynn_element+1), adjustment); }
// bl_mynn_subroutines.F90:5441
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqc(mynn_element+1) = mpas_mul(s_awqc(mynn_element+1), adjustment); }
// bl_mynn_subroutines.F90:5442
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqv(mynn_element+1) = mpas_mul(s_awqv(mynn_element+1), adjustment); }
// bl_mynn_subroutines.F90:5443
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqnc(mynn_element+1) = mpas_mul(s_awqnc(mynn_element+1), adjustment); }
// bl_mynn_subroutines.F90:5444
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqni(mynn_element+1) = mpas_mul(s_awqni(mynn_element+1), adjustment); }
// bl_mynn_subroutines.F90:5445
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqnwfa(mynn_element+1) = mpas_mul(s_awqnwfa(mynn_element+1), adjustment); }
// bl_mynn_subroutines.F90:5446
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqnifa(mynn_element+1) = mpas_mul(s_awqnifa(mynn_element+1), adjustment); }
// bl_mynn_subroutines.F90:5447
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqnbca(mynn_element+1) = mpas_mul(s_awqnbca(mynn_element+1), adjustment); }
// bl_mynn_subroutines.F90:5448
if (momentum_opt) {
// bl_mynn_subroutines.F90:5449
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awu(mynn_element+1) = mpas_mul(s_awu(mynn_element+1), adjustment); }
// bl_mynn_subroutines.F90:5450
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awv(mynn_element+1) = mpas_mul(s_awv(mynn_element+1), adjustment); }
// bl_mynn_subroutines.F90:5451
}
// bl_mynn_subroutines.F90:5452
if (tke_opt) {
// bl_mynn_subroutines.F90:5453
for (int mynn_element=0; mynn_element < (nl+1); ++mynn_element) { s_awqke(mynn_element+1) = mpas_mul(s_awqke(mynn_element+1), adjustment); }
// bl_mynn_subroutines.F90:5454
}
// bl_mynn_subroutines.F90:5455

// bl_mynn_subroutines.F90:5458
for (int mynn_element=0; mynn_element < (8*(nl+1)); ++mynn_element) { upa((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1) = mpas_mul(upa((mynn_element%(nl+1))+1,(mynn_element/(nl+1))+1), adjustment); }
// bl_mynn_subroutines.F90:5459
}
// bl_mynn_subroutines.F90:5464
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5465
for (i=1; (1) > 0 ? i <= (nup) : i >= (nup); i += (1)) {
// bl_mynn_subroutines.F90:5466
edmf_a(k) = mpas_add(edmf_a(k), upa(k, i));
// bl_mynn_subroutines.F90:5467
edmf_w(k) = mpas_add(edmf_w(k), mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upw(k, i)));
// bl_mynn_subroutines.F90:5468
edmf_qt(k) = mpas_add(edmf_qt(k), mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upqt(k, i)));
// bl_mynn_subroutines.F90:5469
edmf_thl(k) = mpas_add(edmf_thl(k), mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upthl(k, i)));
// bl_mynn_subroutines.F90:5470
edmf_ent(k) = mpas_add(edmf_ent(k), mpas_mul(mpas_mul(rhoz(k), upa(k, i)), ent(k, i)));
// bl_mynn_subroutines.F90:5471
edmf_qc(k) = mpas_add(edmf_qc(k), mpas_mul(mpas_mul(rhoz(k), upa(k, i)), upqc(k, i)));
// bl_mynn_subroutines.F90:5472
}
// bl_mynn_subroutines.F90:5473
}
// bl_mynn_subroutines.F90:5474
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5477
if (soft32_greater(as_type<uint>(float(edmf_a(k))),as_type<uint>(float(0.f)))) {
// bl_mynn_subroutines.F90:5478
edmf_w(k) = mpas_div(edmf_w(k), edmf_a(k));
// bl_mynn_subroutines.F90:5479
edmf_qt(k) = mpas_div(edmf_qt(k), edmf_a(k));
// bl_mynn_subroutines.F90:5480
edmf_thl(k) = mpas_div(edmf_thl(k), edmf_a(k));
// bl_mynn_subroutines.F90:5481
edmf_ent(k) = mpas_div(edmf_ent(k), edmf_a(k));
// bl_mynn_subroutines.F90:5482
edmf_qc(k) = mpas_div(edmf_qc(k), edmf_a(k));
// bl_mynn_subroutines.F90:5483
edmf_a(k) = mpas_mul(edmf_a(k), psig_w);
// bl_mynn_subroutines.F90:5485
if (soft32_greater(as_type<uint>(float(mpas_mul(edmf_a(k), edmf_w(k)))),as_type<uint>(float(maxmf)))) { maxmf = mpas_mul(edmf_a(k), edmf_w(k)); }
// bl_mynn_subroutines.F90:5486
}
// bl_mynn_subroutines.F90:5487
}
// bl_mynn_subroutines.F90:5490

// bl_mynn_subroutines.F90:5509
if (env_subs) {
// bl_mynn_subroutines.F90:5510
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5515
envi_w(k) = mpas_mul(onethird, mpas_add(mpas_add(edmf_w(((k) - (1))), edmf_w(k)), edmf_w(((k) + (1)))));
// bl_mynn_subroutines.F90:5516
envi_a(k) = mpas_mul(mpas_mul(onethird, mpas_add(mpas_add(edmf_a(((k) - (1))), edmf_a(k)), edmf_a(((k) + (1))))), adjustment);
// bl_mynn_subroutines.F90:5517
}
// bl_mynn_subroutines.F90:5519
envi_w(kts) = edmf_w(kts);
// bl_mynn_subroutines.F90:5520
envi_a(kts) = edmf_a(kts);
// bl_mynn_subroutines.F90:5522
envi_w(kte) = 0.0f;
// bl_mynn_subroutines.F90:5523
envi_a(kte) = edmf_a(kte);
// bl_mynn_subroutines.F90:5525
envi_w(((kte) + (1))) = 0.0f;
// bl_mynn_subroutines.F90:5526
envi_a(((kte) + (1))) = edmf_a(kte);
// bl_mynn_subroutines.F90:5530
if (soft32_greater(as_type<uint>(float(envi_w(kts))),as_type<uint>(float(mpas_div(mpas_mul(0.9f, dz(kts)), dt))))) {
// bl_mynn_subroutines.F90:5531
sublim = mpas_div(mpas_div(mpas_mul(0.9f, dz(kts)), dt), envi_w(kts));
// bl_mynn_subroutines.F90:5532
} else {
// bl_mynn_subroutines.F90:5533
sublim = 1.0f;
// bl_mynn_subroutines.F90:5534
}
// bl_mynn_subroutines.F90:5536
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:5537
temp = envi_a(k);
// bl_mynn_subroutines.F90:5538
envi_a(k) = mpas_sub(1.0f, temp);
// bl_mynn_subroutines.F90:5539
envi_w(k) = mpas_div(mpas_mul(mpas_mul(mpas_mul(csub, sublim), envi_w(k)), temp), mpas_sub(1.f, temp));
// bl_mynn_subroutines.F90:5540
}
// bl_mynn_subroutines.F90:5543
dzi(kts) = mpas_mul(0.5f, mpas_add(dz(kts), dz(((kts) + (1)))));
// bl_mynn_subroutines.F90:5544
sub_thl(kts) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(0.5f, envi_w(kts)), envi_a(kts)), mpas_sub(mpas_mul(rho(((kts) + (1))), thl(((kts) + (1)))), mpas_mul(rho(kts), thl(kts)))), dzi(kts)), rhoz(k));
// bl_mynn_subroutines.F90:5546
sub_sqv(kts) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(0.5f, envi_w(kts)), envi_a(kts)), mpas_sub(mpas_mul(rho(((kts) + (1))), qv(((kts) + (1)))), mpas_mul(rho(kts), qv(kts)))), dzi(kts)), rhoz(k));
// bl_mynn_subroutines.F90:5548
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5549
dzi(k) = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) + (1)))));
// bl_mynn_subroutines.F90:5550
sub_thl(k) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, mpas_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), mpas_add(envi_a(k), envi_a(((k) - (1))))), mpas_sub(mpas_mul(rho(((k) + (1))), thl(((k) + (1)))), mpas_mul(rho(k), thl(k)))), dzi(k)), rhoz(k));
// bl_mynn_subroutines.F90:5552
sub_sqv(k) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, mpas_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), mpas_add(envi_a(k), envi_a(((k) - (1))))), mpas_sub(mpas_mul(rho(((k) + (1))), qv(((k) + (1)))), mpas_mul(rho(k), qv(k)))), dzi(k)), rhoz(k));
// bl_mynn_subroutines.F90:5554
}
// bl_mynn_subroutines.F90:5556
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5557
det_thl(k) = mpas_mul(mpas_mul(mpas_mul(cdet, mpas_sub(envm_thl(k), thl(k))), envi_a(k)), psig_w);
// bl_mynn_subroutines.F90:5558
det_sqv(k) = mpas_mul(mpas_mul(mpas_mul(cdet, mpas_sub(envm_sqv(k), qv(k))), envi_a(k)), psig_w);
// bl_mynn_subroutines.F90:5559
det_sqc(k) = mpas_mul(mpas_mul(mpas_mul(cdet, mpas_sub(envm_sqc(k), qc(k))), envi_a(k)), psig_w);
// bl_mynn_subroutines.F90:5560
}
// bl_mynn_subroutines.F90:5562
if (momentum_opt) {
// bl_mynn_subroutines.F90:5563
sub_u(kts) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(0.5f, envi_w(kts)), envi_a(kts)), mpas_sub(mpas_mul(rho(((kts) + (1))), u(((kts) + (1)))), mpas_mul(rho(kts), u(kts)))), dzi(kts)), rhoz(k));
// bl_mynn_subroutines.F90:5565
sub_v(kts) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(0.5f, envi_w(kts)), envi_a(kts)), mpas_sub(mpas_mul(rho(((kts) + (1))), v(((kts) + (1)))), mpas_mul(rho(kts), v(kts)))), dzi(kts)), rhoz(k));
// bl_mynn_subroutines.F90:5567
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5568
sub_u(k) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, mpas_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), mpas_add(envi_a(k), envi_a(((k) - (1))))), mpas_sub(mpas_mul(rho(((k) + (1))), u(((k) + (1)))), mpas_mul(rho(k), u(k)))), dzi(k)), rhoz(k));
// bl_mynn_subroutines.F90:5570
sub_v(k) = mpas_div(mpas_div(mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, mpas_add(envi_w(k), envi_w(((k) - (1))))), 0.5f), mpas_add(envi_a(k), envi_a(((k) - (1))))), mpas_sub(mpas_mul(rho(((k) + (1))), v(((k) + (1)))), mpas_mul(rho(k), v(k)))), dzi(k)), rhoz(k));
// bl_mynn_subroutines.F90:5572
}
// bl_mynn_subroutines.F90:5574
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5575
det_u(k) = mpas_mul(mpas_mul(mpas_mul(cdet, mpas_sub(envm_u(k), u(k))), envi_a(k)), psig_w);
// bl_mynn_subroutines.F90:5576
det_v(k) = mpas_mul(mpas_mul(mpas_mul(cdet, mpas_sub(envm_v(k), v(k))), envi_a(k)), psig_w);
// bl_mynn_subroutines.F90:5577
}
// bl_mynn_subroutines.F90:5578
}
// bl_mynn_subroutines.F90:5579
}
// bl_mynn_subroutines.F90:5584
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:5585
exneri(k) = mpas_div(mpas_add(mpas_mul(exner(k), dz(((k) + (1)))), mpas_mul(exner(((k) + (1))), dz(k))), mpas_add(dz(((k) + (1))), dz(k)));
// bl_mynn_subroutines.F90:5586
edmf_th(k) = mpas_add(edmf_thl(k), mpas_mul(mpas_div(xlvcp, exneri(k)), edmf_qc(k)));
// bl_mynn_subroutines.F90:5587
dzi(k) = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) + (1)))));
// bl_mynn_subroutines.F90:5588
}
// bl_mynn_subroutines.F90:5593
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (2))) : k >= (((kte) - (2))); k += (1)) {
// bl_mynn_subroutines.F90:5594
if (((k) > (ktop))) { break; }
// bl_mynn_subroutines.F90:5595
if (((soft32_greater(as_type<uint>(float(mpas_mul(0.5f, mpas_add(edmf_qc(k), edmf_qc(((k) - (1))))))),as_type<uint>(float(0.0f)))) && (soft32_less(as_type<uint>(float(cldfra_bl1d(k))),as_type<uint>(float(cf_thresh)))))) {
// bl_mynn_subroutines.F90:5597
aup = mpas_div(mpas_add(mpas_mul(edmf_a(k), dzi(((k) - (1)))), mpas_mul(edmf_a(((k) - (1))), dzi(k))), mpas_add(dzi(((k) - (1))), dzi(k)));
// bl_mynn_subroutines.F90:5598
thp = mpas_div(mpas_add(mpas_mul(edmf_th(k), dzi(((k) - (1)))), mpas_mul(edmf_th(((k) - (1))), dzi(k))), mpas_add(dzi(((k) - (1))), dzi(k)));
// bl_mynn_subroutines.F90:5599
qtp = mpas_div(mpas_add(mpas_mul(edmf_qt(k), dzi(((k) - (1)))), mpas_mul(edmf_qt(((k) - (1))), dzi(k))), mpas_add(dzi(((k) - (1))), dzi(k)));
// bl_mynn_subroutines.F90:5603
esat = f_esat_blend(tk(k), t0c, tice);
// bl_mynn_subroutines.F90:5605
qsl = mpas_div(mpas_mul(ep_2, esat), mynn_max(((float)(1.e-7f)), ((float)(mpas_sub(p(k), mpas_mul(ep_3, esat))))));
// bl_mynn_subroutines.F90:5608
if (((soft32_greater(as_type<uint>(float(edmf_qc(k))),as_type<uint>(float(0.0f)))) && (soft32_greater(as_type<uint>(float(edmf_qc(((k) - (1))))),as_type<uint>(float(0.0f)))))) {
// bl_mynn_subroutines.F90:5609
qcp = mpas_div(mpas_add(mpas_mul(edmf_qc(k), dzi(((k) - (1)))), mpas_mul(edmf_qc(((k) - (1))), dzi(k))), mpas_add(dzi(((k) - (1))), dzi(k)));
// bl_mynn_subroutines.F90:5610
} else {
// bl_mynn_subroutines.F90:5611
qcp = mynn_max(((float)(edmf_qc(k))), ((float)(edmf_qc(((k) - (1))))));
// bl_mynn_subroutines.F90:5612
}
// bl_mynn_subroutines.F90:5615
xl = f_xl_blend(tk(k), t0c, tice, cice, cliq, cpv, xls, xlv);
// bl_mynn_subroutines.F90:5616
qsat_tk = f_qsat_blend(tk(k), t0c, tice, p(k), 0);
// bl_mynn_subroutines.F90:5618
rsl = mpas_div(mpas_mul(xl, qsat_tk), mpas_mul(r_v, mynn_square(tk(k))));
// bl_mynn_subroutines.F90:5620
cpm = mpas_add(cp, mpas_mul(qt(k), cpv));
// bl_mynn_subroutines.F90:5621
a = mpas_div(1.f, mpas_add(1.f, mpas_div(mpas_mul(xl, rsl), cpm)));
// bl_mynn_subroutines.F90:5622
b9 = mpas_mul(a, rsl);
// bl_mynn_subroutines.F90:5624
q2p = mpas_div(xlvcp, exner(k));
// bl_mynn_subroutines.F90:5625
pt = mpas_add(thl(k), mpas_mul(mpas_mul(q2p, qcp), aup));
// bl_mynn_subroutines.F90:5626
bb = mpas_div(mpas_mul(b9, tk(k)), pt);
// bl_mynn_subroutines.F90:5632
qww = mpas_add(1.f, mpas_mul(0.61f, qt(k)));
// bl_mynn_subroutines.F90:5633
alpha = mpas_mul(0.61f, pt);
// bl_mynn_subroutines.F90:5634
beta = mpas_sub(mpas_div(mpas_mul(pt, xl), mpas_mul(tk(k), cp)), mpas_mul(1.61f, pt));
// bl_mynn_subroutines.F90:5638
if (soft32_greater(as_type<uint>(float(a)),as_type<uint>(float(0.0f)))) {
// bl_mynn_subroutines.F90:5639
f = mynn_min(((float)(mpas_div(1.0f, a))), ((float)(4.0f)));
// bl_mynn_subroutines.F90:5640
} else {
// bl_mynn_subroutines.F90:5641
f = 1.0f;
// bl_mynn_subroutines.F90:5642
}
// bl_mynn_subroutines.F90:5649
sigq = mpas_mul(mpas_mul(10.f, aup), mpas_sub(qtp, qt(k)));
// bl_mynn_subroutines.F90:5651
sigq = mynn_max(((float)(sigq)), ((float)(mpas_mul(qsat_tk, 0.02f))));
// bl_mynn_subroutines.F90:5652
sigq = mynn_min(((float)(sigq)), ((float)(mpas_mul(qsat_tk, 0.25f))));
// bl_mynn_subroutines.F90:5654
qmq = mpas_mul(a, mpas_sub(qt(k), qsat_tk));
// bl_mynn_subroutines.F90:5655
q1 = mpas_div(qmq, sigq);
// bl_mynn_subroutines.F90:5657
if (soft32_greater_equal(as_type<uint>(float(mpas_sub(landsea, 1.5f))),as_type<uint>(float(0)))) {
// bl_mynn_subroutines.F90:5661
mf_cf = mynn_min(((float)(mynn_max(((float)(mpas_add(0.5f, mpas_mul(0.36f, mynn_atan(mpas_mul(1.55f, q1)))))), ((float)(0.01f))))), ((float)(0.6f)));
// bl_mynn_subroutines.F90:5662
mf_cf = mynn_max(((float)(mf_cf)), ((float)(mpas_mul(1.2f, aup))));
// bl_mynn_subroutines.F90:5663
mf_cf = mynn_min(((float)(mf_cf)), ((float)(mpas_mul(5.0f, aup))));
// bl_mynn_subroutines.F90:5664
} else {
// bl_mynn_subroutines.F90:5668
mf_cf = mynn_min(((float)(mynn_max(((float)(mpas_add(0.5f, mpas_mul(0.36f, mynn_atan(mpas_mul(1.55f, q1)))))), ((float)(0.01f))))), ((float)(0.6f)));
// bl_mynn_subroutines.F90:5669
mf_cf = mynn_max(((float)(mf_cf)), ((float)(mpas_mul(1.8f, aup))));
// bl_mynn_subroutines.F90:5670
mf_cf = mynn_min(((float)(mf_cf)), ((float)(mpas_mul(5.0f, aup))));
// bl_mynn_subroutines.F90:5671
}
// bl_mynn_subroutines.F90:5684
if (soft32_greater_equal(as_type<uint>(float(mpas_sub(landsea, 1.5f))),as_type<uint>(float(0)))) {
// bl_mynn_subroutines.F90:5685
if (soft32_greater(as_type<uint>(float(mpas_mul(qcp, aup))),as_type<uint>(float(5e-5f)))) {
// bl_mynn_subroutines.F90:5686
qc_bl1d(k) = mpas_sub(mpas_mul(1.86f, mpas_mul(qcp, aup)), 2.2e-5f);
// bl_mynn_subroutines.F90:5687
} else {
// bl_mynn_subroutines.F90:5688
qc_bl1d(k) = mpas_mul(1.18f, mpas_mul(qcp, aup));
// bl_mynn_subroutines.F90:5689
}
// bl_mynn_subroutines.F90:5690
cldfra_bl1d(k) = mf_cf;
// bl_mynn_subroutines.F90:5691
ac_mf = mf_cf;
// bl_mynn_subroutines.F90:5692
} else {
// bl_mynn_subroutines.F90:5693
if (soft32_greater(as_type<uint>(float(mpas_mul(qcp, aup))),as_type<uint>(float(5e-5f)))) {
// bl_mynn_subroutines.F90:5694
qc_bl1d(k) = mpas_sub(mpas_mul(1.86f, mpas_mul(qcp, aup)), 2.2e-5f);
// bl_mynn_subroutines.F90:5695
} else {
// bl_mynn_subroutines.F90:5696
qc_bl1d(k) = mpas_mul(1.18f, mpas_mul(qcp, aup));
// bl_mynn_subroutines.F90:5697
}
// bl_mynn_subroutines.F90:5698
cldfra_bl1d(k) = mf_cf;
// bl_mynn_subroutines.F90:5699
ac_mf = mf_cf;
// bl_mynn_subroutines.F90:5700
}
// bl_mynn_subroutines.F90:5708
q1 = mynn_max(((float)(q1)), ((float)(mynn_neg(2.25f))));
// bl_mynn_subroutines.F90:5713
if (soft32_greater_equal(as_type<uint>(float(q1)),as_type<uint>(float(1.0f)))) {
// bl_mynn_subroutines.F90:5714
fng = 1.0f;
// bl_mynn_subroutines.F90:5715
} else if (((soft32_greater_equal(as_type<uint>(float(q1)),as_type<uint>(float(mynn_neg(1.7f))))) && (soft32_less(as_type<uint>(float(q1)),as_type<uint>(float(1.0f)))))) {
// bl_mynn_subroutines.F90:5716
fng = mynn_exp(mpas_mul(mynn_neg(0.4f), mpas_sub(q1, 1.0f)));
// bl_mynn_subroutines.F90:5717
} else if (((soft32_greater_equal(as_type<uint>(float(q1)),as_type<uint>(float(mynn_neg(2.5f))))) && (soft32_less(as_type<uint>(float(q1)),as_type<uint>(float(mynn_neg(1.7f))))))) {
// bl_mynn_subroutines.F90:5718
fng = mpas_add(3.0f, mynn_exp(mpas_mul(mynn_neg(3.8f), mpas_add(q1, 1.7f))));
// bl_mynn_subroutines.F90:5719
} else {
// bl_mynn_subroutines.F90:5720
fng = mynn_min(((float)(mpas_add(23.9f, mynn_exp(mpas_mul(mynn_neg(1.6f), mpas_add(q1, 2.5f)))))), ((float)(60.f)));
// bl_mynn_subroutines.F90:5721
}
// bl_mynn_subroutines.F90:5724
vt(k) = mpas_sub(mpas_sub(qww, mpas_mul(mpas_mul(mpas_mul(mpas_mul(1.5f, aup), beta), bb), fng)), 1.f);
// bl_mynn_subroutines.F90:5725
vq(k) = mpas_sub(mpas_add(alpha, mpas_mul(mpas_mul(mpas_mul(mpas_mul(1.5f, aup), beta), a), fng)), tv0);
// bl_mynn_subroutines.F90:5726
}
// bl_mynn_subroutines.F90:5727
}
// bl_mynn_subroutines.F90:5729
}
// bl_mynn_subroutines.F90:5732
if (((ktop) > (0))) {
// bl_mynn_subroutines.F90:5733
maxqc=edmf_qc(1); for(int l=2;l<=ktop;l++) maxqc=mynn_max(maxqc,edmf_qc(l));
// bl_mynn_subroutines.F90:5734
if (soft32_less(as_type<uint>(float(maxqc)),as_type<uint>(float(1.e-8f)))) { maxmf = mpas_mul(mynn_neg(1.0f), maxmf); }
// bl_mynn_subroutines.F90:5735
}
// bl_mynn_subroutines.F90:5740
if (soft32_greater(as_type<uint>(float(edmf_w(1))),as_type<uint>(float(4.0f)))) {
// bl_mynn_subroutines.F90:5742
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5743
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5744
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5772
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5773
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5774
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5775
; // Native diagnostic print; no field arithmetic.
// bl_mynn_subroutines.F90:5777
}

r.maxwidth[col]=maxwidth;r.maxmf[col]=maxmf;r.ztop[col]=ztop;r.ktop[col]=ktop;
#undef zw
#undef dz
#undef p
#undef rho
#undef u
#undef v
#undef w
#undef th
#undef thl
#undef thv
#undef tk
#undef qt
#undef qv
#undef qc
#undef qke
#undef qnc
#undef qni
#undef qnwfa
#undef qnifa
#undef qnbca
#undef exner
#undef vt
#undef vq
#undef sgm
#undef edmf_a
#undef edmf_w
#undef edmf_qt
#undef edmf_thl
#undef edmf_ent
#undef edmf_qc
#undef s_aw
#undef s_awthl
#undef s_awqt
#undef s_awqv
#undef s_awqc
#undef s_awu
#undef s_awv
#undef s_awqke
#undef s_awqnc
#undef s_awqni
#undef s_awqnwfa
#undef s_awqnifa
#undef s_awqnbca
#undef sub_thl
#undef sub_sqv
#undef sub_u
#undef sub_v
#undef det_thl
#undef det_sqv
#undef det_sqc
#undef det_u
#undef det_v
#undef qc_bl1d
#undef cldfra_bl1d
#undef qc_bl1d_old
#undef cldfra_bl1d_old
#undef rstoch_col
#undef edmf_th
#undef s_aw2
#undef exneri
#undef dzi
#undef rhoz
#undef envm_a
#undef envm_w
#undef envm_thl
#undef envm_sqv
#undef envm_sqc
#undef envm_u
#undef envm_v
#undef envi_a
#undef envi_w
#undef upw
#undef upthl
#undef upqt
#undef upqc
#undef upqv
#undef upa
#undef upu
#undef upv
#undef upthv
#undef upqke
#undef upqnc
#undef upqni
#undef upqnwfa
#undef upqnifa
#undef upqnbca
#undef ent
#undef entf
#undef enti
}

kernel void hexv842_mpas_bl_mynn_dmp_pack0_f32(constant Hexv842MpasBlMynnDmpPack0F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDmpPack0F32Params &cfg [[buffer(1)]],uint i [[thread_position_in_grid]]){uint n=uint(cfg.nc*(cfg.nl+1));if(i>=n)return;uint col=i/uint(cfg.nl+1),k=i%uint(cfg.nl+1);
if(k<uint(cfg.nl+1)) r.packed[ulong(0)*n+i]=r.zw[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl)) r.packed[ulong(1)*n+i]=r.dz[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(2)*n+i]=r.p[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(3)*n+i]=r.rho[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(4)*n+i]=r.u[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(5)*n+i]=r.v[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(6)*n+i]=r.w[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(7)*n+i]=r.th[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(8)*n+i]=r.thl[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(9)*n+i]=r.thv[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(10)*n+i]=r.tk[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(11)*n+i]=r.qt[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(12)*n+i]=r.qv[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(13)*n+i]=r.qc[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(14)*n+i]=r.qke[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(15)*n+i]=r.qnc[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(16)*n+i]=r.qni[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(17)*n+i]=r.qnwfa[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(18)*n+i]=r.qnifa[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(19)*n+i]=r.qnbca[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(20)*n+i]=r.exner[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(21)*n+i]=r.vt[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(22)*n+i]=r.vq[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(23)*n+i]=r.sgm[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(24)*n+i]=r.edmf_a[ulong(col)*(cfg.nl)+k];
}

kernel void hexv842_mpas_bl_mynn_dmp_pack1_f32(constant Hexv842MpasBlMynnDmpPack1F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDmpPack1F32Params &cfg [[buffer(1)]],uint i [[thread_position_in_grid]]){uint n=uint(cfg.nc*(cfg.nl+1));if(i>=n)return;uint col=i/uint(cfg.nl+1),k=i%uint(cfg.nl+1);
if(k<uint(cfg.nl)) r.packed[ulong(25)*n+i]=r.edmf_w[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(26)*n+i]=r.edmf_qt[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(27)*n+i]=r.edmf_thl[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(28)*n+i]=r.edmf_ent[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(29)*n+i]=r.edmf_qc[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl+1)) r.packed[ulong(30)*n+i]=r.s_aw[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl+1)) r.packed[ulong(31)*n+i]=r.s_awthl[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl+1)) r.packed[ulong(32)*n+i]=r.s_awqt[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl+1)) r.packed[ulong(33)*n+i]=r.s_awqv[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl+1)) r.packed[ulong(34)*n+i]=r.s_awqc[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl+1)) r.packed[ulong(35)*n+i]=r.s_awu[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl+1)) r.packed[ulong(36)*n+i]=r.s_awv[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl+1)) r.packed[ulong(37)*n+i]=r.s_awqke[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl+1)) r.packed[ulong(38)*n+i]=r.s_awqnc[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl+1)) r.packed[ulong(39)*n+i]=r.s_awqni[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl+1)) r.packed[ulong(40)*n+i]=r.s_awqnwfa[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl+1)) r.packed[ulong(41)*n+i]=r.s_awqnifa[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl+1)) r.packed[ulong(42)*n+i]=r.s_awqnbca[ulong(col)*(cfg.nl+1)+k];
if(k<uint(cfg.nl)) r.packed[ulong(43)*n+i]=r.sub_thl[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(44)*n+i]=r.sub_sqv[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(45)*n+i]=r.sub_u[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(46)*n+i]=r.sub_v[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(47)*n+i]=r.det_thl[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(48)*n+i]=r.det_sqv[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(49)*n+i]=r.det_sqc[ulong(col)*(cfg.nl)+k];
}

kernel void hexv842_mpas_bl_mynn_dmp_pack2_f32(constant Hexv842MpasBlMynnDmpPack2F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDmpPack2F32Params &cfg [[buffer(1)]],uint i [[thread_position_in_grid]]){uint n=uint(cfg.nc*(cfg.nl+1));if(i>=n)return;uint col=i/uint(cfg.nl+1),k=i%uint(cfg.nl+1);
if(k<uint(cfg.nl)) r.packed[ulong(50)*n+i]=r.det_u[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(51)*n+i]=r.det_v[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(52)*n+i]=r.qc_bl1d[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(53)*n+i]=r.cldfra_bl1d[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(54)*n+i]=r.qc_bl1d_old[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(55)*n+i]=r.cldfra_bl1d_old[ulong(col)*(cfg.nl)+k];
if(k<uint(cfg.nl)) r.packed[ulong(56)*n+i]=r.rstoch_col[ulong(col)*(cfg.nl)+k];
}

kernel void hexv842_mpas_bl_mynn_dmp_scatter_f32(constant Hexv842MpasBlMynnDmpScatterF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDmpScatterF32Params &cfg [[buffer(1)]],uint i [[thread_position_in_grid]]){uint n=uint(cfg.nc*(cfg.nl+1));if(i>=n)return;uint col=i/uint(cfg.nl+1),k=i%uint(cfg.nl+1);
if(k<uint(cfg.nl)) r.vt[ulong(col)*(cfg.nl)+k]=r.packed[ulong(21)*n+i];
if(k<uint(cfg.nl)) r.vq[ulong(col)*(cfg.nl)+k]=r.packed[ulong(22)*n+i];
if(k<uint(cfg.nl)) r.edmf_a[ulong(col)*(cfg.nl)+k]=r.packed[ulong(24)*n+i];
if(k<uint(cfg.nl)) r.edmf_w[ulong(col)*(cfg.nl)+k]=r.packed[ulong(25)*n+i];
if(k<uint(cfg.nl)) r.edmf_qt[ulong(col)*(cfg.nl)+k]=r.packed[ulong(26)*n+i];
if(k<uint(cfg.nl)) r.edmf_thl[ulong(col)*(cfg.nl)+k]=r.packed[ulong(27)*n+i];
if(k<uint(cfg.nl)) r.edmf_ent[ulong(col)*(cfg.nl)+k]=r.packed[ulong(28)*n+i];
if(k<uint(cfg.nl)) r.edmf_qc[ulong(col)*(cfg.nl)+k]=r.packed[ulong(29)*n+i];
if(k<uint(cfg.nl+1)) r.s_aw[ulong(col)*(cfg.nl+1)+k]=r.packed[ulong(30)*n+i];
if(k<uint(cfg.nl+1)) r.s_awthl[ulong(col)*(cfg.nl+1)+k]=r.packed[ulong(31)*n+i];
if(k<uint(cfg.nl+1)) r.s_awqt[ulong(col)*(cfg.nl+1)+k]=r.packed[ulong(32)*n+i];
if(k<uint(cfg.nl+1)) r.s_awqv[ulong(col)*(cfg.nl+1)+k]=r.packed[ulong(33)*n+i];
if(k<uint(cfg.nl+1)) r.s_awqc[ulong(col)*(cfg.nl+1)+k]=r.packed[ulong(34)*n+i];
if(k<uint(cfg.nl+1)) r.s_awu[ulong(col)*(cfg.nl+1)+k]=r.packed[ulong(35)*n+i];
if(k<uint(cfg.nl+1)) r.s_awv[ulong(col)*(cfg.nl+1)+k]=r.packed[ulong(36)*n+i];
if(k<uint(cfg.nl+1)) r.s_awqke[ulong(col)*(cfg.nl+1)+k]=r.packed[ulong(37)*n+i];
if(k<uint(cfg.nl+1)) r.s_awqnc[ulong(col)*(cfg.nl+1)+k]=r.packed[ulong(38)*n+i];
if(k<uint(cfg.nl+1)) r.s_awqni[ulong(col)*(cfg.nl+1)+k]=r.packed[ulong(39)*n+i];
if(k<uint(cfg.nl+1)) r.s_awqnwfa[ulong(col)*(cfg.nl+1)+k]=r.packed[ulong(40)*n+i];
if(k<uint(cfg.nl+1)) r.s_awqnifa[ulong(col)*(cfg.nl+1)+k]=r.packed[ulong(41)*n+i];
if(k<uint(cfg.nl+1)) r.s_awqnbca[ulong(col)*(cfg.nl+1)+k]=r.packed[ulong(42)*n+i];
if(k<uint(cfg.nl)) r.sub_thl[ulong(col)*(cfg.nl)+k]=r.packed[ulong(43)*n+i];
if(k<uint(cfg.nl)) r.sub_sqv[ulong(col)*(cfg.nl)+k]=r.packed[ulong(44)*n+i];
if(k<uint(cfg.nl)) r.sub_u[ulong(col)*(cfg.nl)+k]=r.packed[ulong(45)*n+i];
if(k<uint(cfg.nl)) r.sub_v[ulong(col)*(cfg.nl)+k]=r.packed[ulong(46)*n+i];
if(k<uint(cfg.nl)) r.det_thl[ulong(col)*(cfg.nl)+k]=r.packed[ulong(47)*n+i];
if(k<uint(cfg.nl)) r.det_sqv[ulong(col)*(cfg.nl)+k]=r.packed[ulong(48)*n+i];
if(k<uint(cfg.nl)) r.det_sqc[ulong(col)*(cfg.nl)+k]=r.packed[ulong(49)*n+i];
if(k<uint(cfg.nl)) r.det_u[ulong(col)*(cfg.nl)+k]=r.packed[ulong(50)*n+i];
if(k<uint(cfg.nl)) r.det_v[ulong(col)*(cfg.nl)+k]=r.packed[ulong(51)*n+i];
if(k<uint(cfg.nl)) r.qc_bl1d[ulong(col)*(cfg.nl)+k]=r.packed[ulong(52)*n+i];
if(k<uint(cfg.nl)) r.cldfra_bl1d[ulong(col)*(cfg.nl)+k]=r.packed[ulong(53)*n+i];
}
