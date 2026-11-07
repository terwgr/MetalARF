#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_stability_f32_bindings.metalh"
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

#include "hexv842_mpas_bl_mynn_stability_validate_f32_bindings.metalh"
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
static inline float f_phim(float zet) {
float dummy_0;
float dummy_1;
float dummy_11;
float dummy_2;
float dummy_22;
float dummy_3;
float dummy_33;
float dummy_4;
float dummy_44;
float dummy_psi;
constexpr float am_st = 6.1f;
constexpr float bm_st = 2.5f;
constexpr float rbm_st = ((1.f) / (bm_st));
constexpr float ah_st = 5.3f;
constexpr float bh_st = 1.1f;
constexpr float rbh_st = ((1.f) / (bh_st));
constexpr float am_unst = 10.f;
constexpr float ah_unst = 34.f;
float phi_m;
float phim;
// bl_mynn_subroutines.F90:6362
if (soft32_greater_equal(as_type<uint>(zet),0u)) {
// bl_mynn_subroutines.F90:6363
dummy_0 = mpas_add(1, mynn_pow(((float)(zet)), ((float)(bm_st))));
// bl_mynn_subroutines.F90:6364
dummy_1 = mpas_add(zet, mynn_pow(((float)(dummy_0)), ((float)(rbm_st))));
// bl_mynn_subroutines.F90:6365
dummy_11 = mpas_add(1, mpas_mul(mynn_pow(((float)(dummy_0)), ((float)(mpas_sub(rbm_st, 1)))), mynn_pow(((float)(zet)), ((float)(mpas_sub(bm_st, 1))))));
// bl_mynn_subroutines.F90:6366
dummy_2 = mpas_mul(mpas_div(mynn_neg(am_st), dummy_1), dummy_11);
// bl_mynn_subroutines.F90:6367
phi_m = mpas_sub(1, mpas_mul(zet, dummy_2));
// bl_mynn_subroutines.F90:6368
} else {
// bl_mynn_subroutines.F90:6369
dummy_0 = mynn_pow(((float)(mpas_sub(1.0f, mpas_mul(cphm_unst, zet)))), ((float)(0.25f)));
// bl_mynn_subroutines.F90:6370
phi_m = mpas_div(1.f, dummy_0);
// bl_mynn_subroutines.F90:6371
dummy_psi = mpas_add(mpas_sub(mpas_add(mpas_mul(2.f, mynn_log(mpas_mul(0.5f, mpas_add(1.f, dummy_0)))), mynn_log(mpas_mul(0.5f, mpas_add(1.f, mynn_square(dummy_0))))), mpas_mul(2.f, mynn_atan(dummy_0))), 1.570796f);
// bl_mynn_subroutines.F90:6373
dummy_0 = mpas_sub(1.f, mpas_mul(am_unst, zet));
// bl_mynn_subroutines.F90:6374
dummy_1 = mynn_pow(((float)(dummy_0)), ((float)(0.333333f)));
// bl_mynn_subroutines.F90:6375
dummy_11 = mpas_mul(mpas_mul(mynn_neg(0.33333f), am_unst), mynn_pow(((float)(dummy_0)), ((float)(mynn_neg(0.6666667f)))));
// bl_mynn_subroutines.F90:6376
dummy_2 = mpas_mul(0.33333f, mpas_add(mpas_add(mynn_square(dummy_1), dummy_1), 1.f));
// bl_mynn_subroutines.F90:6377
dummy_22 = mpas_mul(mpas_mul(0.3333f, dummy_11), mpas_add(mpas_mul(2.f, dummy_1), 1.f));
// bl_mynn_subroutines.F90:6378
dummy_3 = mpas_mul(0.57735f, mpas_add(mpas_mul(2.f, dummy_1), 1.f));
// bl_mynn_subroutines.F90:6379
dummy_33 = mpas_mul(1.1547f, dummy_11);
// bl_mynn_subroutines.F90:6380
dummy_4 = mpas_add(mpas_sub(mpas_mul(1.5f, mynn_log(dummy_2)), mpas_mul(1.73205f, mynn_atan(dummy_3))), 1.813799364f);
// bl_mynn_subroutines.F90:6381
dummy_44 = mpas_sub(mpas_mul(mpas_div(1.5f, dummy_2), dummy_22), mpas_div(mpas_mul(1.73205f, dummy_33), mpas_add(1.f, mynn_square(dummy_3))));
// bl_mynn_subroutines.F90:6383
dummy_0 = mynn_square(zet);
// bl_mynn_subroutines.F90:6384
dummy_1 = mpas_div(1.f, mpas_add(1.f, dummy_0));
// bl_mynn_subroutines.F90:6385
dummy_11 = mpas_mul(2.f, zet);
// bl_mynn_subroutines.F90:6386
dummy_2 = mpas_mul(mpas_add(mpas_add(mpas_div(mpas_sub(1, phi_m), zet), mpas_mul(dummy_11, dummy_4)), mpas_mul(dummy_0, dummy_44)), dummy_1);
// bl_mynn_subroutines.F90:6387
dummy_22 = mpas_mul(mpas_mul(mynn_neg(dummy_11), mpas_add(dummy_psi, mpas_mul(dummy_0, dummy_4))), mynn_square(dummy_1));
// bl_mynn_subroutines.F90:6389
phi_m = mpas_sub(1.f, mpas_mul(zet, mpas_add(dummy_2, dummy_22)));
// bl_mynn_subroutines.F90:6390
}
// bl_mynn_subroutines.F90:6393
phim = phi_m;
return phim;
}
static inline float f_phih(float zet) {
float dummy_0;
float dummy_1;
float dummy_11;
float dummy_2;
float dummy_22;
float dummy_3;
float dummy_33;
float dummy_4;
float dummy_44;
float dummy_psi;
constexpr float am_st = 6.1f;
constexpr float bm_st = 2.5f;
constexpr float rbm_st = ((1.f) / (bm_st));
constexpr float ah_st = 5.3f;
constexpr float bh_st = 1.1f;
constexpr float rbh_st = ((1.f) / (bh_st));
constexpr float am_unst = 10.f;
constexpr float ah_unst = 34.f;
float phh;
float phih;
// bl_mynn_subroutines.F90:6414
if (soft32_greater_equal(as_type<uint>(zet),0u)) {
// bl_mynn_subroutines.F90:6415
dummy_0 = mpas_add(1, mynn_pow(((float)(zet)), ((float)(bh_st))));
// bl_mynn_subroutines.F90:6416
dummy_1 = mpas_add(zet, mynn_pow(((float)(dummy_0)), ((float)(rbh_st))));
// bl_mynn_subroutines.F90:6417
dummy_11 = mpas_add(1, mpas_mul(mynn_pow(((float)(dummy_0)), ((float)(mpas_sub(rbh_st, 1)))), mynn_pow(((float)(zet)), ((float)(mpas_sub(bh_st, 1))))));
// bl_mynn_subroutines.F90:6418
dummy_2 = mpas_mul(mpas_div(mynn_neg(ah_st), dummy_1), dummy_11);
// bl_mynn_subroutines.F90:6419
phih = mpas_sub(1, mpas_mul(zet, dummy_2));
// bl_mynn_subroutines.F90:6420
} else {
// bl_mynn_subroutines.F90:6421
dummy_0 = mynn_pow(((float)(mpas_sub(1.0f, mpas_mul(cphh_unst, zet)))), ((float)(0.5f)));
// bl_mynn_subroutines.F90:6422
phh = mpas_div(1.f, dummy_0);
// bl_mynn_subroutines.F90:6423
dummy_psi = mpas_mul(2.f, mynn_log(mpas_mul(0.5f, mpas_add(1.f, dummy_0))));
// bl_mynn_subroutines.F90:6425
dummy_0 = mpas_sub(1.f, mpas_mul(ah_unst, zet));
// bl_mynn_subroutines.F90:6426
dummy_1 = mynn_pow(((float)(dummy_0)), ((float)(0.333333f)));
// bl_mynn_subroutines.F90:6427
dummy_11 = mpas_mul(mpas_mul(mynn_neg(0.33333f), ah_unst), mynn_pow(((float)(dummy_0)), ((float)(mynn_neg(0.6666667f)))));
// bl_mynn_subroutines.F90:6428
dummy_2 = mpas_mul(0.33333f, mpas_add(mpas_add(mynn_square(dummy_1), dummy_1), 1.f));
// bl_mynn_subroutines.F90:6429
dummy_22 = mpas_mul(mpas_mul(0.3333f, dummy_11), mpas_add(mpas_mul(2.f, dummy_1), 1.f));
// bl_mynn_subroutines.F90:6430
dummy_3 = mpas_mul(0.57735f, mpas_add(mpas_mul(2.f, dummy_1), 1.f));
// bl_mynn_subroutines.F90:6431
dummy_33 = mpas_mul(1.1547f, dummy_11);
// bl_mynn_subroutines.F90:6432
dummy_4 = mpas_add(mpas_sub(mpas_mul(1.5f, mynn_log(dummy_2)), mpas_mul(1.73205f, mynn_atan(dummy_3))), 1.813799364f);
// bl_mynn_subroutines.F90:6433
dummy_44 = mpas_sub(mpas_mul(mpas_div(1.5f, dummy_2), dummy_22), mpas_div(mpas_mul(1.73205f, dummy_33), mpas_add(1.f, mynn_square(dummy_3))));
// bl_mynn_subroutines.F90:6435
dummy_0 = mynn_square(zet);
// bl_mynn_subroutines.F90:6436
dummy_1 = mpas_div(1.f, mpas_add(1.f, dummy_0));
// bl_mynn_subroutines.F90:6437
dummy_11 = mpas_mul(2.f, zet);
// bl_mynn_subroutines.F90:6438
dummy_2 = mpas_mul(mpas_add(mpas_add(mpas_div(mpas_sub(1, phh), zet), mpas_mul(dummy_11, dummy_4)), mpas_mul(dummy_0, dummy_44)), dummy_1);
// bl_mynn_subroutines.F90:6439
dummy_22 = mpas_mul(mpas_mul(mynn_neg(dummy_11), mpas_add(dummy_psi, mpas_mul(dummy_0, dummy_4))), mynn_square(dummy_1));
// bl_mynn_subroutines.F90:6441
phih = mpas_sub(1.f, mpas_mul(zet, mpas_add(dummy_2, dummy_22)));
// bl_mynn_subroutines.F90:6442
}
return phih;
}
static inline void f_scale_aware(float dx, float pbl1, thread float& psig_bl, thread float& psig_shcu) {
float dxdh;
// bl_mynn_subroutines.F90:6288
psig_bl = 1.0f;
// bl_mynn_subroutines.F90:6289
psig_shcu = 1.0f;
// bl_mynn_subroutines.F90:6290
dxdh = mpas_div(mynn_max(((float)(mpas_mul(2.5f, dx))), ((float)(10.f))), mynn_min(((float)(pbl1)), ((float)(3000.f))));
// bl_mynn_subroutines.F90:6298
psig_bl = mpas_div(mpas_add(mynn_square(dxdh), mpas_mul(0.106f, mynn_pow(((float)(dxdh)), ((float)(0.667f))))), mpas_add(mpas_add(mynn_square(dxdh), mpas_mul(0.066f, mynn_pow(((float)(dxdh)), ((float)(0.667f))))), 0.071f));
// bl_mynn_subroutines.F90:6301
dxdh = mpas_div(mynn_max(((float)(mpas_mul(2.5f, dx))), ((float)(10.f))), mynn_min(((float)(mpas_add(pbl1, 500.f))), ((float)(3500.f))));
// bl_mynn_subroutines.F90:6328
psig_shcu = mpas_div(mpas_add(mynn_square(dxdh), mpas_mul(0.145f, mynn_pow(((float)(dxdh)), ((float)(0.667f))))), mpas_add(mpas_add(mynn_square(dxdh), mpas_mul(0.172f, mynn_pow(((float)(dxdh)), ((float)(0.667f))))), 0.170f));
// bl_mynn_subroutines.F90:6338
if ((soft32_greater(as_type<uint>(psig_bl),0x3f800000u))) { psig_bl = 1.0f; }
// bl_mynn_subroutines.F90:6339
if ((soft32_less(as_type<uint>(psig_bl),0u))) { psig_bl = 0.0f; }
// bl_mynn_subroutines.F90:6341
if ((soft32_greater(as_type<uint>(psig_shcu),0x3f800000u))) { psig_shcu = 1.0f; }
// bl_mynn_subroutines.F90:6342
if ((soft32_less(as_type<uint>(psig_shcu),0u))) { psig_shcu = 0.0f; }
}
kernel void hexv842_mpas_bl_mynn_stability_f32(constant Hexv842MpasBlMynnStabilityF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnStabilityF32Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
if(i>=uint(p.count))return;float bl,sh;f_scale_aware(r.dx[i],r.pblh[i],bl,sh);r.phim[i]=f_phim(r.zet[i]);r.phih[i]=f_phih(r.zet[i]);r.psig_bl[i]=bl;r.psig_shcu[i]=sh;
}

kernel void hexv842_mpas_bl_mynn_stability_validate_f32(constant Hexv842MpasBlMynnStabilityValidateF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnStabilityValidateF32Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]) {
if(i>=uint(p.count))return;uint z=as_type<uint>(r.zet[i])&0x7fffffffu,d=as_type<uint>(r.dx[i]),h=as_type<uint>(r.pblh[i]);
r.invalid[i]=(z>=0x7f800000u || d<0x00800000u || d>=0x7f800000u || h<0x00800000u || h>=0x7f800000u);
}
