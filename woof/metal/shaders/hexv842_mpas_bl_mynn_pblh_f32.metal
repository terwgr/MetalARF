#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_pblh_f32_bindings.metalh"
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

#include "hexv842_mpas_bl_mynn_pblh_validate_f32_bindings.metalh"
#include "hexv842_mpas_bl_mynn_tanh.metalh"
static inline float mynn_tanh(float x){return mynn_result(x,x,as_type<float>(mynn_tanh_bits(as_type<uint>(x))));}

kernel void hexv842_mpas_bl_mynn_pblh_f32(constant Hexv842MpasBlMynnPblhF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnPblhF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;float zi,landsea=r.landsea[col];int kzi;
#define thetav1d(i) r.thetav1d[ulong(col)*nl+((i)-1)]
#define qke1d(i) r.qke1d[ulong(col)*nl+((i)-1)]
#define zw1d(i) r.zw1d[ulong(col)*nl+((i)-1)]
#define dz1d(i) r.dz1d[ulong(col)*nl+((i)-1)]

float pblh_tke;
float qtke;
float qtkem1;
float wt;
float maxqke;
float tkeeps;
float minthv;
float delt_thv;
constexpr float sbl_lim = 200.f;
constexpr float sbl_damp = 400.f;
int i;
int j;
int k;
int kthv;
int ktke;
// bl_mynn_subroutines.F90:4513
kzi = 2;
// bl_mynn_subroutines.F90:4516
k = ((kts) + (1));
// bl_mynn_subroutines.F90:4517
kthv = 1;
// bl_mynn_subroutines.F90:4518
minthv = 9.e9f;
// bl_mynn_subroutines.F90:4519
while (soft32_less_equal(as_type<uint>(zw1d(k)),as_type<uint>(200.f))) {
// bl_mynn_subroutines.F90:4521
if (soft32_greater(as_type<uint>(minthv),as_type<uint>(thetav1d(k)))) {
// bl_mynn_subroutines.F90:4522
minthv = thetav1d(k);
// bl_mynn_subroutines.F90:4523
kthv = k;
// bl_mynn_subroutines.F90:4524
}
// bl_mynn_subroutines.F90:4525
k = ((k) + (1));
// bl_mynn_subroutines.F90:4527
}
// bl_mynn_subroutines.F90:4530
zi = 0.f;
// bl_mynn_subroutines.F90:4531
k = ((kthv) + (1));
// bl_mynn_subroutines.F90:4532
if (soft32_greater_equal(as_type<uint>(mpas_sub(landsea,1.5f)),0u)) {
// bl_mynn_subroutines.F90:4534
delt_thv = 1.0f;
// bl_mynn_subroutines.F90:4535
} else {
// bl_mynn_subroutines.F90:4537
delt_thv = 1.25f;
// bl_mynn_subroutines.F90:4538
}
// bl_mynn_subroutines.F90:4540
zi = 0.f;
// bl_mynn_subroutines.F90:4541
k = ((kthv) + (1));
// bl_mynn_subroutines.F90:4543
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:4544
if (soft32_greater_equal(as_type<uint>(thetav1d(k)),as_type<uint>(mpas_add(minthv,delt_thv)))) {
// bl_mynn_subroutines.F90:4545
zi = mpas_sub(zw1d(k), mpas_mul(dz1d(((k) - (1))), mynn_min(((float)(mpas_div(mpas_sub(thetav1d(k), mpas_add(minthv, delt_thv)), mynn_max(((float)(mpas_sub(thetav1d(k), thetav1d(((k) - (1)))))), ((float)(1e-6f)))))), ((float)(1.0f)))));
// bl_mynn_subroutines.F90:4548
}
// bl_mynn_subroutines.F90:4550
if (((k) == (((kte) - (1))))) { zi = zw1d(((kts) + (1))); }
// bl_mynn_subroutines.F90:4551
if ((!soft32_equal(as_type<uint>(zi),0u))) { break; }
// bl_mynn_subroutines.F90:4552
}
// bl_mynn_subroutines.F90:4559
ktke = 1;
// bl_mynn_subroutines.F90:4560
maxqke = mynn_max(((float)(qke1d(kts))), ((float)(0.f)));
// bl_mynn_subroutines.F90:4563
tkeeps = mpas_div(maxqke, 40.f);
// bl_mynn_subroutines.F90:4564
tkeeps = mynn_max(((float)(tkeeps)), ((float)(0.02f)));
// bl_mynn_subroutines.F90:4565
pblh_tke = 0.f;
// bl_mynn_subroutines.F90:4567
k = ((ktke) + (1));
// bl_mynn_subroutines.F90:4569
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:4571
qtke = mynn_max(((float)(mpas_div(qke1d(k), 2.f))), ((float)(0.f)));
// bl_mynn_subroutines.F90:4572
qtkem1 = mynn_max(((float)(mpas_div(qke1d(((k) - (1))), 2.f))), ((float)(0.f)));
// bl_mynn_subroutines.F90:4573
if (soft32_less_equal(as_type<uint>(qtke),as_type<uint>(tkeeps))) {
// bl_mynn_subroutines.F90:4574
pblh_tke = mpas_sub(zw1d(k), mpas_mul(dz1d(((k) - (1))), mynn_min(((float)(mpas_div(mpas_sub(tkeeps, qtke), mynn_max(((float)(mpas_sub(qtkem1, qtke))), ((float)(1e-6f)))))), ((float)(1.0f)))));
// bl_mynn_subroutines.F90:4577
pblh_tke = mynn_max(((float)(pblh_tke)), ((float)(zw1d(((kts) + (1))))));
// bl_mynn_subroutines.F90:4579
}
// bl_mynn_subroutines.F90:4581
if (((k) == (((kte) - (1))))) { pblh_tke = zw1d(((kts) + (1))); }
// bl_mynn_subroutines.F90:4582
if ((!soft32_equal(as_type<uint>(pblh_tke),0u))) { break; }
// bl_mynn_subroutines.F90:4583
}
// bl_mynn_subroutines.F90:4591
pblh_tke = mynn_min(((float)(pblh_tke)), ((float)(mpas_add(zi, 350.f))));
// bl_mynn_subroutines.F90:4592
pblh_tke = mynn_max(((float)(pblh_tke)), ((float)(mynn_max(((float)(mpas_sub(zi, 350.f))), ((float)(10.f))))));
// bl_mynn_subroutines.F90:4594
wt = mpas_add(mpas_mul(.5f, mynn_tanh(mpas_div(mpas_sub(zi, sbl_lim), sbl_damp))), .5f);
// bl_mynn_subroutines.F90:4595
if (soft32_less_equal(as_type<uint>(maxqke),as_type<uint>(0.05f))) {
// bl_mynn_subroutines.F90:4597
} else {
// bl_mynn_subroutines.F90:4599
zi = mpas_add(mpas_mul(pblh_tke, mpas_sub(1.f, wt)), mpas_mul(zi, wt));
// bl_mynn_subroutines.F90:4600
}
// bl_mynn_subroutines.F90:4603
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:4604
if (soft32_greater_equal(as_type<uint>(zw1d(k)),as_type<uint>(zi))) {
// bl_mynn_subroutines.F90:4605
kzi = ((k) - (1));
// bl_mynn_subroutines.F90:4606
break;
// bl_mynn_subroutines.F90:4607
}
// bl_mynn_subroutines.F90:4608
}

#undef thetav1d
#undef qke1d
#undef zw1d
#undef dz1d
r.zi[col]=zi;r.kzi[col]=kzi;
}

kernel void hexv842_mpas_bl_mynn_pblh_validate_f32(constant Hexv842MpasBlMynnPblhValidateF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnPblhValidateF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl;bool stop=false,bad=(as_type<uint>(r.landsea[col])&0x7fffffffu)>=0x7f800000u;
for(int k=0;k<nl;k++){ulong j=ulong(col)*nl+k;uint z=as_type<uint>(r.zw1d[j]);
 bad|=((z&0x7fffffffu)>=0x7f800000u)||((as_type<uint>(r.thetav1d[j])&0x7fffffffu)>=0x7f800000u)||((as_type<uint>(r.qke1d[j])&0x7fffffffu)>=0x7f800000u)||((as_type<uint>(r.dz1d[j])&0x7fffffffu)>=0x7f800000u);
 if(k>=1 && soft32_greater(z,as_type<uint>(200.f)))stop=true;}
r.invalid[col]=int(bad||!stop);
}
