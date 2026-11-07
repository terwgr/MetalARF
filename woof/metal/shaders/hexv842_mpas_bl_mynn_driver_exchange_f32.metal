#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_driver_exchange_f32_bindings.metalh"
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

kernel void hexv842_mpas_bl_mynn_driver_exchange_f32(constant Hexv842MpasBlMynnDriverExchangeF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDriverExchangeF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]){
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl,k;float dzk;
#define dfm(i) r.dfm[ulong(col)*nl+((i)-1)]
#define dfh(i) r.dfh[ulong(col)*nl+((i)-1)]
#define dz(i) r.dz[ulong(col)*nl+((i)-1)]
#define k_m(i) r.k_m[ulong(col)*nl+((i)-1)]
#define k_h(i) r.k_h[ulong(col)*nl+((i)-1)]
// bl_mynn_subroutines.F90:4331
k_m(kts) = 0.f;
// bl_mynn_subroutines.F90:4332
k_h(kts) = 0.f;
// bl_mynn_subroutines.F90:4334
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:4335
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:4336
k_m(k) = mpas_mul(dfm(k), dzk);
// bl_mynn_subroutines.F90:4337
k_h(k) = mpas_mul(dfh(k), dzk);
// bl_mynn_subroutines.F90:4338
}

#undef dfm
#undef dfh
#undef dz
#undef k_m
#undef k_h
}
