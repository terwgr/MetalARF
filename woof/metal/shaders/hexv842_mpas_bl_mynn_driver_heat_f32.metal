#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_driver_heat_f32_bindings.metalh"
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

kernel void hexv842_mpas_bl_mynn_driver_heat_f32(constant Hexv842MpasBlMynnDriverHeatF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDriverHeatF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]){
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl,k;float cp=cfg.cp,b1=24.f;
#define qke1(i) r.qke1[ulong(col)*nl+((i)-1)]
#define el_pbl1(i) r.el_pbl1[ulong(col)*nl+((i)-1)]
#define p1(i) r.p1[ulong(col)*nl+((i)-1)]
#define diss_heat(i) r.diss_heat[ulong(col)*nl+((i)-1)]
// bl_mynn.F90:973
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn.F90:975
diss_heat(k) = mynn_min(((float)(mynn_max(((float)(mpas_div(mpas_div(mpas_mul(1.0f, mynn_pow(((float)(qke1(k))), ((float)(1.5f)))), mpas_mul(b1, mynn_max(((float)(mpas_mul(0.5f, mpas_add(el_pbl1(k), el_pbl1(((k) + (1))))))), ((float)(1.f))))), cp))), ((float)(0.0f))))), ((float)(0.002f)));
// bl_mynn.F90:978
diss_heat(k) = mpas_mul(diss_heat(k), mynn_exp(mpas_div(mynn_neg(10000.f), mynn_max(((float)(p1(k))), ((float)(1.f))))));
// bl_mynn.F90:979
}
// bl_mynn.F90:980
diss_heat(kte) = 0.f;

#undef qke1
#undef el_pbl1
#undef p1
#undef diss_heat
}
