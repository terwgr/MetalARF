#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_turbulence25_f32_bindings.metalh"
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

static inline ulong mynn_result64(ulong a,ulong b,ulong r){if(!soft64_is_nan(r))return r;if(soft64_is_nan(a))return soft64_quiet_nan(a);if(soft64_is_nan(b))return soft64_quiet_nan(b);return 0xfff8000000000000ul;}
static inline ulong mynn_real64(float a){return soft64_from_f32_bits(as_type<uint>(a));}
static inline float mynn_real32(ulong a){return as_type<float>(soft64_to_f32_bits(a));}
static inline ulong mynn_dadd(ulong a,ulong b){return mynn_result64(a,b,soft64_add(a,b));}
static inline ulong mynn_dsub(ulong a,ulong b){return mynn_result64(a,b,soft64_sub(a,b));}
static inline ulong mynn_dmul(ulong a,ulong b){return mynn_result64(a,b,soft64_mul(a,b));}
static inline ulong mynn_ddiv(ulong a,ulong b){return mynn_result64(a,b,soft64_div(a,b));}
static inline ulong mynn_sqrt(ulong a){return mynn_result64(a,a,soft64_sqrt(a));}
static inline ulong mynn_neg(ulong a){return soft64_negate_bits(a);}
static inline ulong mynn_square(ulong a){return mynn_dmul(a,a);}
static inline ulong mynn_max64(ulong a,ulong b){return soft64_greater(a,b)?a:b;}

kernel void hexv842_mpas_bl_mynn_turbulence25_f32(constant Hexv842MpasBlMynnTurbulence25F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnTurbulence25F32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;float grav=cfg.gravity,karman=cfg.karman,gtr=mynn_div(grav,300.f),tv0=mynn_mul(cfg.ep1,300.f);float zi=r.zi[col],rmo=r.rmo[col],fltv=r.fltv[col],psig_bl=r.psig_bl[col];constexpr float qmin=0.f,zmax=1.f,onethird=1.f/3.f;
constexpr float b1 = 24.0f;
constexpr float b2 = 15.0f;
constexpr float twothirds = ((2.f) / (3.f));
constexpr float qkemin = 1.e-12f;
constexpr float pr = 0.74f;
constexpr float g1 = 0.235f;
constexpr float c2 = 0.729f;
constexpr float c3 = 0.340f;
constexpr float c5 = 0.2f;
constexpr float a1 = ((((b1) * (((1.0f) - (((3.0f) * (g1))))))) / (6.0f));
constexpr float c1 = ((g1) - (((1.0f) / (((((3.0f) * (a1))) * (2.88449914061481660f))))));
constexpr float a2 = ((((a1) * (((g1) - (c1))))) / (((g1) * (pr))));
constexpr float g2 = ((((((b2) / (b1))) * (((1.0f) - (c3))))) + (((((((2.0f) * (a1))) / (b1))) * (((3.0f) - (((2.0f) * (c2))))))));
constexpr float ckmod = 1.f;
constexpr float cc2 = ((1.0f) - (c2));
constexpr float cc3 = ((1.0f) - (c3));
constexpr float e1c = ((((((3.0f) * (a2))) * (b2))) * (cc3));
constexpr float e2c = ((((((9.0f) * (a1))) * (a2))) * (cc2));
constexpr float e3c = ((((((((9.0f) * (a2))) * (a2))) * (cc2))) * (((1.0f) - (c5))));
constexpr float e4c = ((((((12.0f) * (a1))) * (a2))) * (cc2));
constexpr float e5c = ((((6.0f) * (a1))) * (a1));
#define dz(i) r.dz[ulong(col)*nl+((i)-1)]
#define u(i) r.u[ulong(col)*nl+((i)-1)]
#define v(i) r.v[ulong(col)*nl+((i)-1)]
#define thl(i) r.thl[ulong(col)*nl+((i)-1)]
#define thetav(i) r.thetav[ulong(col)*nl+((i)-1)]
#define qw(i) r.qw[ulong(col)*nl+((i)-1)]
#define ql(i) r.ql[ulong(col)*nl+((i)-1)]
#define vt(i) r.vt[ulong(col)*nl+((i)-1)]
#define vq(i) r.vq[ulong(col)*nl+((i)-1)]
#define qke(i) r.qke[ulong(col)*nl+((i)-1)]
#define tsq(i) r.tsq[ulong(col)*nl+((i)-1)]
#define qsq(i) r.qsq[ulong(col)*nl+((i)-1)]
#define cov(i) r.cov[ulong(col)*nl+((i)-1)]
#define theta(i) r.theta[ulong(col)*nl+((i)-1)]
#define cldfra_bl1d(i) r.cldfra_bl1d[ulong(col)*nl+((i)-1)]
#define edmf_w1(i) r.edmf_w1[ulong(col)*nl+((i)-1)]
#define edmf_a1(i) r.edmf_a1[ulong(col)*nl+((i)-1)]
#define tkeprodtd(i) r.tkeprodtd[ulong(col)*nl+((i)-1)]
#define sh(i) r.sh[ulong(col)*nl+((i)-1)]
#define sm(i) r.sm[ulong(col)*nl+((i)-1)]
#define el(i) r.el[ulong(col)*nl+((i)-1)]
#define dfm(i) r.dfm[ulong(col)*nl+((i)-1)]
#define dfh(i) r.dfh[ulong(col)*nl+((i)-1)]
#define dfq(i) r.dfq[ulong(col)*nl+((i)-1)]
#define tcd(i) r.tcd[ulong(col)*nl+((i)-1)]
#define qcd(i) r.qcd[ulong(col)*nl+((i)-1)]
#define pdk(i) r.pdk[ulong(col)*nl+((i)-1)]
#define pdt(i) r.pdt[ulong(col)*nl+((i)-1)]
#define pdq(i) r.pdq[ulong(col)*nl+((i)-1)]
#define pdc(i) r.pdc[ulong(col)*nl+((i)-1)]
#define qshear1d(i) r.qshear1d[ulong(col)*nl+((i)-1)]
#define qbuoy1d(i) r.qbuoy1d[ulong(col)*nl+((i)-1)]
#define qkw(i) r.scratch[(ulong(col)*7+0)*nl+((i)-1)]
#define dtl(i) r.scratch[(ulong(col)*7+1)*nl+((i)-1)]
#define dqw(i) r.scratch[(ulong(col)*7+2)*nl+((i)-1)]
#define dtv(i) r.scratch[(ulong(col)*7+3)*nl+((i)-1)]
#define gm(i) r.scratch[(ulong(col)*7+4)*nl+((i)-1)]
#define gh(i) r.scratch[(ulong(col)*7+5)*nl+((i)-1)]
#define qtke(i) r.scratch[(ulong(col)*7+6)*nl+((i)-1)]
#define u1(i) u(i)
#define v1(i) v(i)
#define zw(i) r.zw[ulong(col)*(nl+1)+((i)-1)]

float q3sq_old;
float dlsq1;
float qwtp_old;
float qwtp_new;
float dudz;
float dvdz;
float dtdz;
float upwp;
float vpwp;
float tpwp;
int k;
float e6c;
float dzk;
float afk;
float abk;
float vtt;
float vqq;
float cw25;
float clow;
float cupp;
float gamt;
float gamq;
float smd;
float gamv;
float elq;
float elh;
float cldavg;
float a2fac;
float duz;
float ri;
float auh;
float aum;
float adh;
float adm;
float aeh;
float aem;
float req;
float rsl;
float rsl2;
float gmelq;
float sm20;
float sh20;
float sm25max;
float sh25max;
float sm25min;
float sh25min;
float sm_pbl;
float sh_pbl;
float zi2;
float wt;
float slht;
float wtpr;
ulong q2sq;
ulong t2sq;
ulong r2sq;
ulong c2sq;
ulong elsq;
ulong gmel;
ulong ghel;
ulong q3sq;
ulong t3sq;
ulong r3sq;
ulong c3sq;
ulong dlsq;
ulong qdiv;
ulong e1;
ulong e2;
ulong e3;
ulong e4;
ulong mynn_enum;
ulong eden;
ulong wden;
float prnum;
float shb;
constexpr float prlimit = 5.0f;
// bl_mynn_subroutines.F90:1664
{

int k;
float rfc;
float f1;
float f2;
float rf1;
float rf2;
float smc;
float shc;
float ri1;
float ri2;
float ri3;
float ri4;
float duz;
float dtz;
float dqz;
float vtt;
float vqq;
float dtq;
float dzk;
float afk;
float abk;
float ri;
float rf;
float a2fac;
// bl_mynn_subroutines.F90:711
rfc = mpas_div(g1, mpas_add(g1, g2));
// bl_mynn_subroutines.F90:712
f1 = mpas_add(mpas_add(mpas_mul(b1, mpas_sub(g1, c1)), mpas_mul(mpas_mul(mpas_mul(3.0f, a2), mpas_sub(1.0f, c2)), mpas_sub(1.0f, c5))), mpas_mul(mpas_mul(2.0f, a1), mpas_sub(3.0f, mpas_mul(2.0f, c2))));
// bl_mynn_subroutines.F90:714
f2 = mpas_sub(mpas_mul(b1, mpas_add(g1, g2)), mpas_mul(mpas_mul(3.0f, a1), mpas_sub(1.0f, c2)));
// bl_mynn_subroutines.F90:715
rf1 = mpas_div(mpas_mul(b1, mpas_sub(g1, c1)), f1);
// bl_mynn_subroutines.F90:716
rf2 = mpas_div(mpas_mul(b1, g1), f2);
// bl_mynn_subroutines.F90:717
smc = mpas_div(mpas_mul(mpas_div(a1, a2), f1), f2);
// bl_mynn_subroutines.F90:718
shc = mpas_mul(mpas_mul(3.0f, a2), mpas_add(g1, g2));
// bl_mynn_subroutines.F90:720
ri1 = mpas_div(0.5f, smc);
// bl_mynn_subroutines.F90:721
ri2 = mpas_mul(rf1, smc);
// bl_mynn_subroutines.F90:722
ri3 = mpas_sub(mpas_mul(mpas_mul(4.0f, rf2), smc), mpas_mul(2.0f, ri2));
// bl_mynn_subroutines.F90:723
ri4 = mynn_square(ri2);
// bl_mynn_subroutines.F90:725
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:726
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:727
afk = mpas_div(dz(k), mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:728
abk = mpas_sub(1.0f, afk);
// bl_mynn_subroutines.F90:729
duz = mpas_add(mynn_square(mpas_sub(u(k), u(((k) - (1))))), mynn_square(mpas_sub(v(k), v(((k) - (1))))));
// bl_mynn_subroutines.F90:730
duz = mpas_div(duz, mynn_square(dzk));
// bl_mynn_subroutines.F90:731
dtz = mpas_div(mpas_sub(thl(k), thl(((k) - (1)))), dzk);
// bl_mynn_subroutines.F90:732
dqz = mpas_div(mpas_sub(qw(k), qw(((k) - (1)))), dzk);
// bl_mynn_subroutines.F90:734
vtt = mpas_add(mpas_add(1.0f, mpas_mul(vt(k), abk)), mpas_mul(vt(((k) - (1))), afk));
// bl_mynn_subroutines.F90:735
vqq = mpas_add(mpas_add(tv0, mpas_mul(vq(k), abk)), mpas_mul(vq(((k) - (1))), afk));
// bl_mynn_subroutines.F90:736
dtq = mpas_add(mpas_mul(vtt, dtz), mpas_mul(vqq, dqz));
// bl_mynn_subroutines.F90:740
dtl(k) = dtz;
// bl_mynn_subroutines.F90:741
dqw(k) = dqz;
// bl_mynn_subroutines.F90:742
dtv(k) = dtq;
// bl_mynn_subroutines.F90:747
gm(k) = duz;
// bl_mynn_subroutines.F90:748
gh(k) = mpas_mul(mynn_neg(dtq), gtr);
// bl_mynn_subroutines.F90:751
if (k == kte && (kte-kts)%2 == 0) { ri = mpas_div(mynn_neg(gh(k)), mynn_max(((float)(1.0e-10f)), ((float)(duz)))); } else { ri = mpas_div(mynn_neg(gh(k)), mynn_max(((float)(duz)), ((float)(1.0e-10f)))); }
// bl_mynn_subroutines.F90:754
if (((ckmod) == (1))) {
// bl_mynn_subroutines.F90:755
if (k == kts + 2*((kte-kts-1)/2) + 1) { a2fac = mpas_div(1.f, mpas_add(1.f, mynn_max(((float)(ri)), ((float)(0.0f))))); } else { a2fac = mpas_div(1.f, mpas_add(1.f, mynn_max(((float)(0.0f)), ((float)(ri))))); }
// bl_mynn_subroutines.F90:756
} else {
// bl_mynn_subroutines.F90:757
a2fac = 1.f;
// bl_mynn_subroutines.F90:758
}
// bl_mynn_subroutines.F90:760
rfc = mpas_div(g1, mpas_add(g1, g2));
// bl_mynn_subroutines.F90:761
f1 = mpas_add(mpas_add(mpas_mul(b1, mpas_sub(g1, c1)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(3.0f, a2), a2fac), mpas_sub(1.0f, c2)), mpas_sub(1.0f, c5))), mpas_mul(mpas_mul(2.0f, a1), mpas_sub(3.0f, mpas_mul(2.0f, c2))));
// bl_mynn_subroutines.F90:763
f2 = mpas_sub(mpas_mul(b1, mpas_add(g1, g2)), mpas_mul(mpas_mul(3.0f, a1), mpas_sub(1.0f, c2)));
// bl_mynn_subroutines.F90:764
rf1 = mpas_div(mpas_mul(b1, mpas_sub(g1, c1)), f1);
// bl_mynn_subroutines.F90:765
rf2 = mpas_div(mpas_mul(b1, g1), f2);
// bl_mynn_subroutines.F90:766
smc = mpas_div(mpas_mul(mpas_div(a1, mpas_mul(a2, a2fac)), f1), f2);
// bl_mynn_subroutines.F90:767
shc = mpas_mul(mpas_mul(3.0f, mpas_mul(a2, a2fac)), mpas_add(g1, g2));
// bl_mynn_subroutines.F90:769
ri1 = mpas_div(0.5f, smc);
// bl_mynn_subroutines.F90:770
ri2 = mpas_mul(rf1, smc);
// bl_mynn_subroutines.F90:771
ri3 = mpas_sub(mpas_mul(mpas_mul(4.0f, rf2), smc), mpas_mul(2.0f, ri2));
// bl_mynn_subroutines.F90:772
ri4 = mynn_square(ri2);
// bl_mynn_subroutines.F90:775
rf = mynn_min(((float)(mpas_mul(ri1, mpas_sub(mpas_add(ri, ri2), mynn_sqrt(mpas_add(mpas_sub(mynn_square(ri), mpas_mul(ri3, ri)), ri4)))))), ((float)(rfc)));
// bl_mynn_subroutines.F90:777
sh(k) = mpas_div(mpas_mul(shc, mpas_sub(rfc, rf)), mpas_sub(1.0f, rf));
// bl_mynn_subroutines.F90:778
sm(k) = mpas_mul(mpas_div(mpas_mul(smc, mpas_sub(rf1, rf)), mpas_sub(rf2, rf)), sh(k));
// bl_mynn_subroutines.F90:779
}

}
// bl_mynn_subroutines.F90:1670
{

float elt;
float vsc;
float wt;
float wt2;
float zi2;
float h1;
float h2;
float hs;
float elblmin0;
float elblavg0;
float cldavg;
float cns;
float alp1;
float alp2;
float alp3;
float alp4;
float alp5;
float alp6;
constexpr float minzi = 300.f;
constexpr float maxdz = 750.f;
constexpr float mindz = 300.f;
constexpr float zslh = 100.f;
constexpr float csl = 2.f;
int i;
int j;
int k;
float afk;
float abk;
float zwk;
float zwk1;
float dzk;
float qdz;
float vflx;
float bv;
float tau_cloud;
float wstar;
float elb;
float els;
float elf;
float el_stab;
float el_mf;
float el_stab_mf;
float elb_mf;
float pblh_plus_ent;
float uonset;
float ugrid;
float wt_u;
float el_les;
constexpr float ctau = 1000.f;
// bl_mynn_subroutines.F90:878

// bl_mynn_subroutines.F90:1061
uonset = mpas_add(3.5f, mpas_mul(dz(kts), 0.1f));
// bl_mynn_subroutines.F90:1062
ugrid = mynn_sqrt(mpas_add(mynn_square(u1(kts)), mynn_square(v1(kts))));
// bl_mynn_subroutines.F90:1063
cns = 3.5f;
// bl_mynn_subroutines.F90:1064
alp1 = 0.22f;
// bl_mynn_subroutines.F90:1065
alp2 = 0.30f;
// bl_mynn_subroutines.F90:1066
alp3 = 2.0f;
// bl_mynn_subroutines.F90:1067
alp4 = 5.0f;
// bl_mynn_subroutines.F90:1068
alp5 = alp2;
// bl_mynn_subroutines.F90:1069
alp6 = 50.0f;
// bl_mynn_subroutines.F90:1073
zi2 = mynn_max(((float)(zi)), ((float)(300.f)));
// bl_mynn_subroutines.F90:1076
h1 = mynn_max(((float)(mpas_mul(0.3f, zi2))), ((float)(300.f)));
// bl_mynn_subroutines.F90:1077
h1 = mynn_min(((float)(h1)), ((float)(600.f)));
// bl_mynn_subroutines.F90:1078
h2 = mpas_mul(h1, 0.5f);
// bl_mynn_subroutines.F90:1080
qtke(kts) = mynn_max(((float)(mpas_mul(0.5f, qke(kts)))), ((float)(0.01f)));
// bl_mynn_subroutines.F90:1081
qkw(kts) = mynn_sqrt(mynn_max(((float)(qke(kts))), ((float)(1.0e-4f))));
// bl_mynn_subroutines.F90:1083
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:1084
afk = mpas_div(dz(k), mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:1085
abk = mpas_sub(1.0f, afk);
// bl_mynn_subroutines.F90:1086
qkw(k) = mynn_sqrt(mynn_max(((float)(mpas_add(mpas_mul(qke(k), abk), mpas_mul(qke(((k) - (1))), afk)))), ((float)(1.0e-3f))));
// bl_mynn_subroutines.F90:1087
qtke(k) = mpas_mul(0.5f, mynn_square(qkw(k)));
// bl_mynn_subroutines.F90:1088
}
// bl_mynn_subroutines.F90:1090
elt = 1.0e-5f;
// bl_mynn_subroutines.F90:1091
vsc = 1.0e-5f;
// bl_mynn_subroutines.F90:1094
pblh_plus_ent = mynn_max(((float)(mpas_add(zi, h1))), ((float)(100.f)));
// bl_mynn_subroutines.F90:1095
k = ((kts) + (1));
// bl_mynn_subroutines.F90:1096
zwk = zw(k);
// bl_mynn_subroutines.F90:1097
while (soft32_less_equal(as_type<uint>(zwk),as_type<uint>(pblh_plus_ent))) {
// bl_mynn_subroutines.F90:1098
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:1099
qdz = mpas_mul(mynn_min(((float)(mynn_max(((float)(mpas_sub(qkw(k), qmin))), ((float)(0.03f))))), ((float)(30.0f))), dzk);
// bl_mynn_subroutines.F90:1100
elt = mpas_add(elt, mpas_mul(qdz, zwk));
// bl_mynn_subroutines.F90:1101
vsc = mpas_add(vsc, qdz);
// bl_mynn_subroutines.F90:1102
k = ((k) + (1));
// bl_mynn_subroutines.F90:1103
zwk = zw(k);
// bl_mynn_subroutines.F90:1104
}
// bl_mynn_subroutines.F90:1106
elt = mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_mul(alp1, elt), vsc))), ((float)(10.f))))), ((float)(400.f)));
// bl_mynn_subroutines.F90:1109
vflx = fltv;
// bl_mynn_subroutines.F90:1110
vsc = mynn_pow(((float)(mpas_mul(mpas_mul(gtr, elt), mynn_max(((float)(vflx)), ((float)(0.0f)))))), ((float)(onethird)));
// bl_mynn_subroutines.F90:1113
el(kts) = 0.0f;
// bl_mynn_subroutines.F90:1114
zwk1 = zw(((kts) + (1)));
// bl_mynn_subroutines.F90:1116
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:1117
zwk = zw(k);
// bl_mynn_subroutines.F90:1118
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:1119
cldavg = mpas_mul(0.5f, mpas_add(cldfra_bl1d(((k) - (1))), cldfra_bl1d(k)));
// bl_mynn_subroutines.F90:1122
if (soft32_greater(as_type<uint>(dtv(k)),0u)) {
// bl_mynn_subroutines.F90:1124
bv = mynn_max(((float)(mynn_sqrt(mpas_mul(gtr, dtv(k))))), ((float)(0.001f)));
// bl_mynn_subroutines.F90:1126
elb_mf = mpas_mul(mpas_div(mynn_max(((float)(mpas_mul(alp2, qkw(k)))), ((float)(mpas_mul(mpas_mul(alp6, edmf_a1(((k) - (1)))), edmf_w1(((k) - (1))))))), bv), mpas_add(1.0f, mpas_mul(alp3, mynn_sqrt(mpas_div(vsc, mpas_mul(bv, elt))))));
// bl_mynn_subroutines.F90:1129
elb = mynn_min(((float)(mpas_div(mynn_max(((float)(mpas_mul(alp5, qkw(k)))), ((float)(mpas_mul(mpas_mul(alp6, edmf_a1(k)), edmf_w1(k))))), bv))), ((float)(zwk)));
// bl_mynn_subroutines.F90:1132
wstar = mpas_mul(1.25f, mynn_pow(((float)(mpas_mul(mpas_mul(gtr, zi), mynn_max(((float)(vflx)), ((float)(1.0e-4f)))))), ((float)(onethird))));
// bl_mynn_subroutines.F90:1133
tau_cloud = mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_mul(ctau, wstar), grav))), ((float)(30.f))))), ((float)(150.f)));
// bl_mynn_subroutines.F90:1135
wt = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(zwk, mpas_add(zi2, h1)), h2))), .5f);
// bl_mynn_subroutines.F90:1136
tau_cloud = mpas_add(mpas_mul(tau_cloud, mpas_sub(1.f, wt)), mpas_mul(50.f, wt));
// bl_mynn_subroutines.F90:1137
elf = mynn_min(((float)(mynn_max(((float)(mpas_mul(tau_cloud, mynn_sqrt(mynn_min(((float)(qtke(k))), ((float)(40.f))))))), ((float)(mpas_div(mpas_mul(mpas_mul(alp6, edmf_a1(k)), edmf_w1(k)), bv)))))), ((float)(zwk)));
// bl_mynn_subroutines.F90:1147
} else {
// bl_mynn_subroutines.F90:1159
wstar = mpas_mul(1.25f, mynn_pow(((float)(mpas_mul(mpas_mul(gtr, zi), mynn_max(((float)(vflx)), ((float)(1.0e-4f)))))), ((float)(onethird))));
// bl_mynn_subroutines.F90:1160
tau_cloud = mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_mul(ctau, wstar), grav))), ((float)(50.f))))), ((float)(200.f)));
// bl_mynn_subroutines.F90:1162
wt = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(zwk, mpas_add(zi2, h1)), h2))), .5f);
// bl_mynn_subroutines.F90:1164
tau_cloud = mpas_add(mpas_mul(tau_cloud, mpas_sub(1.f, wt)), mpas_mul(mynn_max(((float)(100.f)), ((float)(mpas_mul(dzk, 0.25f)))), wt));
// bl_mynn_subroutines.F90:1166
elb = mynn_min(((float)(mpas_mul(tau_cloud, mynn_sqrt(mynn_min(((float)(qtke(k))), ((float)(40.f))))))), ((float)(zwk)));
// bl_mynn_subroutines.F90:1168
elf = elb;
// bl_mynn_subroutines.F90:1169
elb_mf = elb;
// bl_mynn_subroutines.F90:1170
}
// bl_mynn_subroutines.F90:1171
elf = mpas_div(elf, mpas_add(1.f, mpas_div(elf, 800.f)));
// bl_mynn_subroutines.F90:1172
elb_mf = mynn_max(((float)(elb_mf)), ((float)(0.01f)));
// bl_mynn_subroutines.F90:1175
if (soft32_greater(as_type<uint>(rmo),0u)) {
// bl_mynn_subroutines.F90:1176
els = mpas_div(mpas_mul(karman, zwk), mpas_add(1.0f, mpas_mul(cns, mynn_min(((float)(mpas_mul(zwk, rmo))), ((float)(zmax))))));
// bl_mynn_subroutines.F90:1177
} else {
// bl_mynn_subroutines.F90:1178
els = mpas_mul(mpas_mul(karman, zwk), mynn_pow(((float)(mpas_sub(1.0f, mpas_mul(mpas_mul(alp4, zwk), rmo)))), ((float)(0.2f))));
// bl_mynn_subroutines.F90:1179
}
// bl_mynn_subroutines.F90:1182
wt = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(zwk, mpas_add(zi2, h1)), h2))), .5f);
// bl_mynn_subroutines.F90:1185
el(k) = mynn_sqrt(mpas_div(mynn_square(els), mpas_add(mpas_add(1.f, mpas_div(mynn_square(els), mynn_square(elt))), mpas_div(mynn_square(els), mynn_square(elb_mf)))));
// bl_mynn_subroutines.F90:1186
el(k) = mpas_add(mpas_mul(el(k), mpas_sub(1.f, wt)), mpas_mul(elf, wt));
// bl_mynn_subroutines.F90:1189
el_les = mynn_min(((float)(mpas_div(els, mpas_add(1.f, mpas_div(els, 12.f))))), ((float)(elb_mf)));
// bl_mynn_subroutines.F90:1190
el(k) = mpas_add(mpas_mul(el(k), psig_bl), mpas_mul(mpas_sub(1.f, psig_bl), el_les));
// bl_mynn_subroutines.F90:1192
}
// bl_mynn_subroutines.F90:1194

}
// bl_mynn_subroutines.F90:1684
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:1685
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:1686
afk = mpas_div(dz(k), mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:1687
abk = mpas_sub(1.0f, afk);
// bl_mynn_subroutines.F90:1688
elsq = mynn_real64(mynn_square(el(k)));
// bl_mynn_subroutines.F90:1689
q3sq = mynn_real64(mynn_square(qkw(k)));
// bl_mynn_subroutines.F90:1690
q2sq = mynn_dmul(mynn_dmul(mynn_real64(b1), elsq), mynn_real64(mpas_add(mpas_mul(sm(k), gm(k)), mpas_mul(sh(k), gh(k)))));
// bl_mynn_subroutines.F90:1692
sh20 = mynn_max(((float)(sh(k))), ((float)(1e-5f)));
// bl_mynn_subroutines.F90:1693
sm20 = mynn_max(((float)(sm(k))), ((float)(1e-5f)));
// bl_mynn_subroutines.F90:1694
sh(k) = mynn_max(((float)(sh(k))), ((float)(1e-5f)));
// bl_mynn_subroutines.F90:1697
duz = mpas_add(mynn_square(mpas_sub(u(k), u(((k) - (1))))), mynn_square(mpas_sub(v(k), v(((k) - (1))))));
// bl_mynn_subroutines.F90:1698
duz = mpas_div(duz, mynn_square(dzk));
// bl_mynn_subroutines.F90:1700
ri = mpas_div(mynn_neg(gh(k)), mynn_max(((float)(duz)), ((float)(1.0e-10f))));
// bl_mynn_subroutines.F90:1701
if (((ckmod) == (1))) {
// bl_mynn_subroutines.F90:1702
a2fac = mpas_div(1.f, mpas_add(1.f, mynn_max(((float)(ri)), ((float)(0.0f)))));
// bl_mynn_subroutines.F90:1703
} else {
// bl_mynn_subroutines.F90:1704
a2fac = 1.f;
// bl_mynn_subroutines.F90:1705
}
// bl_mynn_subroutines.F90:1713
prnum = mynn_min(((float)(mpas_add(0.76f, mpas_mul(4.0f, mynn_max(((float)(ri)), ((float)(0.0f))))))), ((float)(prlimit)));
// bl_mynn_subroutines.F90:1717
gmel = mynn_dmul(mynn_real64(gm(k)), elsq);
// bl_mynn_subroutines.F90:1718
ghel = mynn_dmul(mynn_real64(gh(k)), elsq);
// bl_mynn_subroutines.F90:1722

// bl_mynn_subroutines.F90:1736
dlsq = elsq;
// bl_mynn_subroutines.F90:1737
if (soft64_less(mynn_ddiv(q3sq,dlsq),mynn_real64(mynn_neg(gh(k))))) { q3sq = mynn_dmul(mynn_neg(dlsq), mynn_real64(gh(k))); }
// bl_mynn_subroutines.F90:1739
if (soft64_less(q3sq,q2sq)) {
// bl_mynn_subroutines.F90:1741
qdiv = mynn_sqrt(mynn_ddiv(q3sq, q2sq));
// bl_mynn_subroutines.F90:1758
sh(k) = mynn_real32(mynn_dmul(mynn_real64(sh(k)), qdiv));
// bl_mynn_subroutines.F90:1759
sm(k) = mynn_real32(mynn_dmul(mynn_real64(sm(k)), qdiv));
// bl_mynn_subroutines.F90:1776
e1 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(mynn_real64(e1c), ghel), mynn_real64(a2fac)), mynn_square(qdiv)));
// bl_mynn_subroutines.F90:1777
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_dmul(mynn_real64(e2c), ghel), mynn_real64(a2fac)), mynn_square(qdiv)));
// bl_mynn_subroutines.F90:1778
e3 = mynn_dadd(e1, mynn_dmul(mynn_dmul(mynn_dmul(mynn_real64(e3c), ghel), mynn_real64(mynn_square(a2fac))), mynn_square(qdiv)));
// bl_mynn_subroutines.F90:1779
e4 = mynn_dsub(e1, mynn_dmul(mynn_dmul(mynn_dmul(mynn_real64(e4c), ghel), mynn_real64(a2fac)), mynn_square(qdiv)));
// bl_mynn_subroutines.F90:1780
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(mynn_dmul(e3, mynn_real64(e5c)), gmel), mynn_square(qdiv)));
// bl_mynn_subroutines.F90:1781
eden = mynn_max64(eden,0x3bc79ca10c924223ul);
// bl_mynn_subroutines.F90:1786
} else {
// bl_mynn_subroutines.F90:1792
e1 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_real64(e1c), ghel), mynn_real64(a2fac)));
// bl_mynn_subroutines.F90:1793
e2 = mynn_dsub(q3sq, mynn_dmul(mynn_dmul(mynn_real64(e2c), ghel), mynn_real64(a2fac)));
// bl_mynn_subroutines.F90:1794
e3 = mynn_dadd(e1, mynn_dmul(mynn_dmul(mynn_real64(e3c), ghel), mynn_real64(mynn_square(a2fac))));
// bl_mynn_subroutines.F90:1795
e4 = mynn_dsub(e1, mynn_dmul(mynn_dmul(mynn_real64(e4c), ghel), mynn_real64(a2fac)));
// bl_mynn_subroutines.F90:1796
eden = mynn_dadd(mynn_dmul(e2, e4), mynn_dmul(mynn_dmul(e3, mynn_real64(e5c)), gmel));
// bl_mynn_subroutines.F90:1797
eden = mynn_max64(eden,0x3bc79ca10c924223ul);
// bl_mynn_subroutines.F90:1799
qdiv = mynn_real64(1.0f);
// bl_mynn_subroutines.F90:1801
sm(k) = mynn_real32(mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, mynn_real64(a1)), mynn_dsub(e3, mynn_dmul(mynn_real64(mpas_mul(3.0f, c1)), e4))), eden));
// bl_mynn_subroutines.F90:1805
sh(k) = mynn_real32(mynn_ddiv(mynn_dmul(mynn_dmul(q3sq, mynn_real64(mpas_mul(a2, a2fac))), mynn_dadd(e2, mynn_dmul(mynn_real64(mpas_mul(mpas_mul(3.0f, c1), e5c)), gmel))), eden));
// bl_mynn_subroutines.F90:1812
}
// bl_mynn_subroutines.F90:1815
gmelq = mynn_real32(mynn_max64(mynn_ddiv(gmel, q3sq),0x3e45798ee2308c3aul));
// bl_mynn_subroutines.F90:1816
sm25max = 4.f;
// bl_mynn_subroutines.F90:1817
sh25max = 4.f;
// bl_mynn_subroutines.F90:1818
sm25min = 0.0f;
// bl_mynn_subroutines.F90:1819
sh25min = 0.0f;
// bl_mynn_subroutines.F90:1823

// bl_mynn_subroutines.F90:1840
if (soft32_greater(as_type<uint>(sh(k)),as_type<uint>(sh25max))) { sh(k) = sh25max; }
// bl_mynn_subroutines.F90:1841
if (soft32_less(as_type<uint>(sh(k)),as_type<uint>(sh25min))) { sh(k) = sh25min; }
// bl_mynn_subroutines.F90:1853
shb = mynn_max(((float)(0.002f)), ((float)(sh(k))));
// bl_mynn_subroutines.F90:1854
if (soft64_less(q3sq,q2sq)) { sm(k) = mynn_min(((float)(sm(k))), ((float)(mpas_mul(prlimit, shb)))); } else { sm(k) = mynn_min(((float)(mpas_mul(prlimit, shb))), ((float)(sm(k)))); }
// bl_mynn_subroutines.F90:1857

// bl_mynn_subroutines.F90:2000
gamt = 0.0f;
// bl_mynn_subroutines.F90:2001
gamq = 0.0f;
// bl_mynn_subroutines.F90:2002
gamv = 0.0f;
// bl_mynn_subroutines.F90:2003

// bl_mynn_subroutines.F90:2007
cldavg = mpas_mul(0.5f, mpas_add(cldfra_bl1d(((k) - (1))), cldfra_bl1d(k)));
// bl_mynn_subroutines.F90:2008
if (((soft32_greater(as_type<uint>(edmf_a1(k)),as_type<uint>(0.001f))) || (soft32_greater(as_type<uint>(cldavg),as_type<uint>(0.02f))))) {
// bl_mynn_subroutines.F90:2010
sm(k) = mynn_max(((float)(sm(k))), ((float)(mpas_mul(0.03f, mynn_min(((float)(mpas_mul(mpas_mul(10.f, edmf_a1(k)), edmf_w1(k)))), ((float)(1.0f)))))));
// bl_mynn_subroutines.F90:2011
sh(k) = mynn_max(((float)(sh(k))), ((float)(mpas_mul(0.03f, mynn_min(((float)(mpas_mul(mpas_mul(10.f, edmf_a1(k)), edmf_w1(k)))), ((float)(1.0f)))))));
// bl_mynn_subroutines.F90:2013
sm(k) = mynn_max(((float)(sm(k))), ((float)(mpas_mul(0.05f, mynn_min(((float)(cldavg)), ((float)(1.0f)))))));
// bl_mynn_subroutines.F90:2014
sh(k) = mynn_max(((float)(sh(k))), ((float)(mpas_mul(0.05f, mynn_min(((float)(cldavg)), ((float)(1.0f)))))));
// bl_mynn_subroutines.F90:2015
}
// bl_mynn_subroutines.F90:2017
elq = mpas_mul(el(k), qkw(k));
// bl_mynn_subroutines.F90:2018
elh = mynn_real32(mynn_dmul(mynn_real64(elq), qdiv));
// bl_mynn_subroutines.F90:2022
pdk(k) = mpas_add(mpas_mul(elq, mpas_add(mpas_add(mpas_mul(sm(k), gm(k)), mpas_mul(sh(k), gh(k))), gamv)), mpas_mul(0.5f, tkeprodtd(k)));
// bl_mynn_subroutines.F90:2025
pdt(k) = mpas_mul(mpas_mul(elh, mpas_add(mpas_mul(sh(k), dtl(k)), gamt)), dtl(k));
// bl_mynn_subroutines.F90:2026
pdq(k) = mpas_mul(mpas_mul(elh, mpas_add(mpas_mul(sh(k), dqw(k)), gamq)), dqw(k));
// bl_mynn_subroutines.F90:2027
pdc(k) = mpas_add(mpas_mul(mpas_mul(mpas_mul(elh, mpas_add(mpas_mul(sh(k), dtl(k)), gamt)), dqw(k)), 0.5f), mpas_mul(mpas_mul(mpas_mul(elh, mpas_add(mpas_mul(sh(k), dqw(k)), gamq)), dtl(k)), 0.5f));
// bl_mynn_subroutines.F90:2032
tcd(k) = mpas_mul(elq, gamt);
// bl_mynn_subroutines.F90:2033
qcd(k) = mpas_mul(elq, gamq);
// bl_mynn_subroutines.F90:2036
dfm(k) = mpas_div(mpas_mul(elq, sm(k)), dzk);
// bl_mynn_subroutines.F90:2037
dfh(k) = mpas_div(mpas_mul(elq, sh(k)), dzk);
// bl_mynn_subroutines.F90:2041
dfq(k) = dfm(k);
// bl_mynn_subroutines.F90:2044

// bl_mynn_subroutines.F90:2076
}
// bl_mynn_subroutines.F90:2079
dfm(kts) = 0.0f;
// bl_mynn_subroutines.F90:2080
dfh(kts) = 0.0f;
// bl_mynn_subroutines.F90:2081
dfq(kts) = 0.0f;
// bl_mynn_subroutines.F90:2082
tcd(kts) = 0.0f;
// bl_mynn_subroutines.F90:2083
qcd(kts) = 0.0f;
// bl_mynn_subroutines.F90:2085
tcd(kte) = 0.0f;
// bl_mynn_subroutines.F90:2086
qcd(kte) = 0.0f;
// bl_mynn_subroutines.F90:2089
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2090
dzk = dz(k);
// bl_mynn_subroutines.F90:2091
tcd(k) = mpas_div(mpas_sub(tcd(((k) + (1))), tcd(k)), dzk);
// bl_mynn_subroutines.F90:2092
qcd(k) = mpas_div(mpas_sub(qcd(((k) + (1))), qcd(k)), dzk);
// bl_mynn_subroutines.F90:2093
}
// bl_mynn_subroutines.F90:2095


#undef dz
#undef u
#undef v
#undef thl
#undef thetav
#undef qw
#undef ql
#undef vt
#undef vq
#undef qke
#undef tsq
#undef qsq
#undef cov
#undef theta
#undef cldfra_bl1d
#undef edmf_w1
#undef edmf_a1
#undef tkeprodtd
#undef sh
#undef sm
#undef el
#undef dfm
#undef dfh
#undef dfq
#undef tcd
#undef qcd
#undef pdk
#undef pdt
#undef pdq
#undef pdc
#undef qshear1d
#undef qbuoy1d
#undef qkw
#undef dtl
#undef dqw
#undef dtv
#undef gm
#undef gh
#undef qtke
#undef u1
#undef v1
#undef zw
}
