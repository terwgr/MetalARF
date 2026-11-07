#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_length2_f32_bindings.metalh"
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

#include "hexv842_mpas_bl_mynn_length2_validate_f32_bindings.metalh"
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

kernel void hexv842_mpas_bl_mynn_length2_f32(constant Hexv842MpasBlMynnLength2F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnLength2F32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;float grav=cfg.gravity,karman=cfg.karman,gtr=mynn_div(grav,300.f);float zi=r.zi[col],rmo=r.rmo[col],fltv=r.fltv[col],psig_bl=r.psig_bl[col];constexpr float qmin=0.f,zmax=1.f,onethird=1.f/3.f;
#define dz(i) r.dz[ulong(col)*nl+((i)-1)]
#define u1(i) r.u1[ulong(col)*nl+((i)-1)]
#define v1(i) r.v1[ulong(col)*nl+((i)-1)]
#define qke(i) r.qke[ulong(col)*nl+((i)-1)]
#define dtv(i) r.dtv[ulong(col)*nl+((i)-1)]
#define cldfra_bl1d(i) r.cldfra_bl1d[ulong(col)*nl+((i)-1)]
#define edmf_w1(i) r.edmf_w1[ulong(col)*nl+((i)-1)]
#define edmf_a1(i) r.edmf_a1[ulong(col)*nl+((i)-1)]
#define qkw(i) r.qkw[ulong(col)*nl+((i)-1)]
#define el(i) r.el[ulong(col)*nl+((i)-1)]
#define zw(i) r.zw[ulong(col)*(nl+1)+((i)-1)]
#define qtke(i) r.qtke[ulong(col)*nl+((i)-1)]

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

#undef dz
#undef u1
#undef v1
#undef qke
#undef dtv
#undef cldfra_bl1d
#undef edmf_w1
#undef edmf_a1
#undef qkw
#undef el
#undef zw
#undef qtke
}

kernel void hexv842_mpas_bl_mynn_length2_validate_f32(constant Hexv842MpasBlMynnLength2ValidateF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnLength2ValidateF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl;bool bad=false,stop=false;float zi=r.zi[col];
uint zbits=as_type<uint>(zi),rmobits=as_type<uint>(r.rmo[col]),fbits=as_type<uint>(r.fltv[col]),pbits=as_type<uint>(r.psig_bl[col]);
bad|=(zbits&0x7fffffffu)>=0x7f800000u || soft32_less(zbits,0u) || (rmobits&0x7fffffffu)>=0x7f800000u || (fbits&0x7fffffffu)>=0x7f800000u || (pbits&0x7fffffffu)>=0x7f800000u;
float zi2=mynn_max(zi,300.f),h1=mynn_min(mynn_max(mynn_mul(.3f,zi2),300.f),600.f),threshold=mynn_max(mynn_add(zi,h1),100.f);
for(int k=0;k<=nl;k++){uint z=as_type<uint>(r.zw[ulong(col)*(nl+1)+k]);bad|=(z&0x7fffffffu)>=0x7f800000u;if(k>=1&&soft32_greater(z,as_type<uint>(threshold)))stop=true;
if(k<nl){ulong j=ulong(col)*nl+k;uint d=as_type<uint>(r.dz[j]);bad|=d<0x00800000u || d>=0x7f800000u;
bad|=(as_type<uint>(r.u1[j])&0x7fffffffu)>=0x7f800000u;
bad|=(as_type<uint>(r.v1[j])&0x7fffffffu)>=0x7f800000u;
bad|=(as_type<uint>(r.qke[j])&0x7fffffffu)>=0x7f800000u;
bad|=(as_type<uint>(r.dtv[j])&0x7fffffffu)>=0x7f800000u;
bad|=(as_type<uint>(r.cldfra_bl1d[j])&0x7fffffffu)>=0x7f800000u;
bad|=(as_type<uint>(r.edmf_w1[j])&0x7fffffffu)>=0x7f800000u;
bad|=(as_type<uint>(r.edmf_a1[j])&0x7fffffffu)>=0x7f800000u;
}}
r.invalid[col]=int(bad||!stop);
}
