#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_driver_pre_f32_bindings.metalh"
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

kernel void hexv842_mpas_bl_mynn_driver_pre_f32(constant Hexv842MpasBlMynnDriverPreF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDriverPreF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl,k;float ust1=r.ust1[col],xlvcp=cfg.xlvcp,xlscp=cfg.xlscp;
#define dz1(i) r.dz1[ulong(col)*nl+((i)-1)]
#define th1(i) r.th1[ulong(col)*nl+((i)-1)]
#define sqv1(i) r.sqv1[ulong(col)*nl+((i)-1)]
#define sqc1(i) r.sqc1[ulong(col)*nl+((i)-1)]
#define sqi1(i) r.sqi1[ulong(col)*nl+((i)-1)]
#define sqs1(i) r.sqs1[ulong(col)*nl+((i)-1)]
#define ex1(i) r.ex1[ulong(col)*nl+((i)-1)]
#define qv1(i) r.qv1[ulong(col)*nl+((i)-1)]
#define qc1(i) r.qc1[ulong(col)*nl+((i)-1)]
#define qi1(i) r.qi1[ulong(col)*nl+((i)-1)]
#define qs1(i) r.qs1[ulong(col)*nl+((i)-1)]
#define sqw(i) r.sqw[ulong(col)*nl+((i)-1)]
#define thl(i) r.thl[ulong(col)*nl+((i)-1)]
#define thetav(i) r.thetav[ulong(col)*nl+((i)-1)]
#define qke1(i) r.qke1[ulong(col)*nl+((i)-1)]
#define sh1(i) r.sh1[ulong(col)*nl+((i)-1)]
#define sm1(i) r.sm1[ulong(col)*nl+((i)-1)]
#define el_pbl1(i) r.el_pbl1[ulong(col)*nl+((i)-1)]
#define tsq1(i) r.tsq1[ulong(col)*nl+((i)-1)]
#define qsq1(i) r.qsq1[ulong(col)*nl+((i)-1)]
#define cov1(i) r.cov1[ulong(col)*nl+((i)-1)]
#define cldfra_bl1(i) r.cldfra_bl1[ulong(col)*nl+((i)-1)]
#define qc_bl1(i) r.qc_bl1[ulong(col)*nl+((i)-1)]
#define qi_bl1(i) r.qi_bl1[ulong(col)*nl+((i)-1)]
#define zw(i) r.zw[ulong(col)*(nl+1)+((i)-1)]
// bl_mynn.F90:634
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:635
qv1(k) = mpas_div(sqv1(k), mpas_sub(1.f, sqv1(k)));
// bl_mynn.F90:636
qc1(k) = mpas_div(sqc1(k), mpas_sub(1.f, sqv1(k)));
// bl_mynn.F90:637
qi1(k) = mpas_div(sqi1(k), mpas_sub(1.f, sqv1(k)));
// bl_mynn.F90:638
qs1(k) = mpas_div(sqs1(k), mpas_sub(1.f, sqv1(k)));
// bl_mynn.F90:639
}
// bl_mynn.F90:641
k = kts;
// bl_mynn.F90:642
zw(k) = 0.f;
// bl_mynn.F90:643
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) + (1))) : k >= (((kte) + (1))); k += (1)) {
// bl_mynn.F90:644
zw(k) = mpas_add(zw(((k) - (1))), dz1(((k) - (1))));
// bl_mynn.F90:645
}
// bl_mynn.F90:681
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:683
sqw(k) = mpas_add(mpas_add(sqv1(k), sqc1(k)), sqi1(k));
// bl_mynn.F90:684
thl(k) = mpas_sub(mpas_sub(th1(k), mpas_mul(mpas_div(xlvcp, ex1(k)), sqc1(k))), mpas_mul(mpas_div(xlscp, ex1(k)), sqi1(k)));
// bl_mynn.F90:685
thetav(k) = mpas_mul(th1(k), mpas_add(1.f, mpas_mul(0.608f, sqv1(k))));
// bl_mynn.F90:691
}

if(cfg.cold){
// bl_mynn.F90:710
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:711
sh1(k) = 0.f;
// bl_mynn.F90:712
sm1(k) = 0.f;
// bl_mynn.F90:713
el_pbl1(k) = 0.f;
// bl_mynn.F90:714
tsq1(k) = 0.f;
// bl_mynn.F90:715
qsq1(k) = 0.f;
// bl_mynn.F90:716
cov1(k) = 0.f;
// bl_mynn.F90:717
cldfra_bl1(k) = 0.f;
// bl_mynn.F90:718
qc_bl1(k) = 0.f;
// bl_mynn.F90:719
qi_bl1(k) = 0.f;
// bl_mynn.F90:720
qke1(k) = 0.f;
// bl_mynn.F90:721
}
// bl_mynn.F90:730
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn.F90:731
qke1(k) = mpas_mul(mpas_mul(5.f, ust1), mynn_max(((float)(mpas_div(mpas_sub(mpas_mul(ust1, 700.f), zw(k)), mpas_mul(mynn_max(((float)(ust1)), ((float)(0.01f))), 700.f)))), ((float)(0.01f))));
// bl_mynn.F90:732
}

}
#undef dz1
#undef th1
#undef sqv1
#undef sqc1
#undef sqi1
#undef sqs1
#undef ex1
#undef qv1
#undef qc1
#undef qi1
#undef qs1
#undef sqw
#undef thl
#undef thetav
#undef qke1
#undef sh1
#undef sm1
#undef el_pbl1
#undef tsq1
#undef qsq1
#undef cov1
#undef cldfra_bl1
#undef qc_bl1
#undef qi_bl1
#undef zw
}
