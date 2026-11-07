#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_moisture_f32_bindings.metalh"
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

kernel void hexv842_mpas_bl_mynn_moisture_f32(constant Hexv842MpasBlMynnMoistureF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnMoistureF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kte=nl;float delt=cfg.delt,xlvcp=cfg.xlvcp,xlscp=cfg.xlscp;
#define dp(i) r.dp[ulong(col)*nl+((i)-1)]
#define exner(i) r.exner[ulong(col)*nl+((i)-1)]
#define qv(i) r.qv[ulong(col)*nl+((i)-1)]
#define qc(i) r.qc[ulong(col)*nl+((i)-1)]
#define qi(i) r.qi[ulong(col)*nl+((i)-1)]
#define qs(i) r.qs[ulong(col)*nl+((i)-1)]
#define th(i) r.th[ulong(col)*nl+((i)-1)]
#define dqv(i) r.dqv[ulong(col)*nl+((i)-1)]
#define dqc(i) r.dqc[ulong(col)*nl+((i)-1)]
#define dqi(i) r.dqi[ulong(col)*nl+((i)-1)]
#define dqs(i) r.dqs[ulong(col)*nl+((i)-1)]
#define dth(i) r.dth[ulong(col)*nl+((i)-1)]

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
dqc2 = mynn_max(((float)(0.0f)), ((float)(mpas_sub(qcmin, qc(k)))));
// bl_mynn_subroutines.F90:4126
dqi2 = mynn_max(((float)(0.0f)), ((float)(mpas_sub(qimin, qi(k)))));
// bl_mynn_subroutines.F90:4127
dqs2 = mynn_max(((float)(0.0f)), ((float)(mpas_sub(qimin, qs(k)))));
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
qc(k) = mpas_add(qc(k), dqc2);
// bl_mynn_subroutines.F90:4138
qi(k) = mpas_add(qi(k), dqi2);
// bl_mynn_subroutines.F90:4139
qs(k) = mpas_add(qs(k), dqs2);
// bl_mynn_subroutines.F90:4140
qv(k) = mpas_sub(mpas_sub(mpas_sub(qv(k), dqc2), dqi2), dqs2);
// bl_mynn_subroutines.F90:4141
th(k) = mpas_add(mpas_add(th(k), mpas_mul(mpas_div(xlvcp, exner(k)), dqc2)), mpas_mul(mpas_div(xlscp, exner(k)), mpas_add(dqi2, dqs2)));
// bl_mynn_subroutines.F90:4145
dqv2 = mynn_max(((float)(0.0f)), ((float)(mpas_sub(qvmin, qv(k)))));
// bl_mynn_subroutines.F90:4146
dqv(k) = mpas_add(dqv(k), mpas_div(dqv2, delt));
// bl_mynn_subroutines.F90:4147
qv(k) = mpas_add(qv(k), dqv2);
// bl_mynn_subroutines.F90:4148
if (((k) != (1))) {
// bl_mynn_subroutines.F90:4149
qv(((k) - (1))) = mpas_sub(qv(((k) - (1))), mpas_div(mpas_mul(dqv2, dp(k)), dp(((k) - (1)))));
// bl_mynn_subroutines.F90:4150
dqv(((k) - (1))) = mpas_sub(dqv(((k) - (1))), mpas_div(mpas_div(mpas_mul(dqv2, dp(k)), dp(((k) - (1)))), delt));
// bl_mynn_subroutines.F90:4151
}
// bl_mynn_subroutines.F90:4152
qv(k) = mynn_max(((float)(qv(k))), ((float)(qvmin)));
// bl_mynn_subroutines.F90:4153
qc(k) = mynn_max(((float)(qc(k))), ((float)(qcmin)));
// bl_mynn_subroutines.F90:4154
qi(k) = mynn_max(((float)(qi(k))), ((float)(qimin)));
// bl_mynn_subroutines.F90:4155
qs(k) = mynn_max(((float)(qs(k))), ((float)(qimin)));
// bl_mynn_subroutines.F90:4156
}
// bl_mynn_subroutines.F90:4160
if (soft32_greater(as_type<uint>(dqv2),as_type<uint>(1.e-20f))) {
// bl_mynn_subroutines.F90:4161
sum = 0.0f;
// bl_mynn_subroutines.F90:4162
for (k=1; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4163
if (soft32_greater(as_type<uint>(qv(k)),as_type<uint>(mpas_mul(2.f,qvmin)))) { sum = mpas_add(sum, mpas_mul(qv(k), dp(k))); }
// bl_mynn_subroutines.F90:4164
}
// bl_mynn_subroutines.F90:4165
aa = mpas_div(mpas_mul(dqv2, dp(1)), mynn_max(((float)(1.e-20f)), ((float)(sum))));
// bl_mynn_subroutines.F90:4166
if (soft32_less(as_type<uint>(aa),as_type<uint>(0.5f))) {
// bl_mynn_subroutines.F90:4167
for (k=1; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4168
if (soft32_greater(as_type<uint>(qv(k)),as_type<uint>(mpas_mul(2.f,qvmin)))) {
// bl_mynn_subroutines.F90:4169
dum = mpas_mul(aa, qv(k));
// bl_mynn_subroutines.F90:4170
qv(k) = mpas_sub(qv(k), dum);
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
return;

#undef dp
#undef exner
#undef qv
#undef qc
#undef qi
#undef qs
#undef th
#undef dqv
#undef dqc
#undef dqi
#undef dqs
#undef dth
}
