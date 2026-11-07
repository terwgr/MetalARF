#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_length1_validate_f32_bindings.metalh"
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

kernel void hexv842_mpas_bl_mynn_length1_validate_f32(constant Hexv842MpasBlMynnLength1ValidateF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnLength1ValidateF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl;bool bad=nl<2,stop=false;float zi=r.zi[col];
uint zbits=as_type<uint>(zi),rmobits=as_type<uint>(r.rmo[col]),fbits=as_type<uint>(r.fltv[col]),pbits=as_type<uint>(r.psig_bl[col]);
bad|=(zbits&0x7fffffffu)>=0x7f800000u || soft32_less(zbits,0u) || (rmobits&0x7fffffffu)>=0x7f800000u || (fbits&0x7fffffffu)>=0x7f800000u || (pbits&0x7fffffffu)>=0x7f800000u;
float zi2=mynn_max(zi,300.f),h1=mynn_min(mynn_max(mynn_mul(.3f,zi2),300.f),600.f),threshold=mynn_add(zi2,h1);
for(int k=0;k<=nl;k++){uint z=as_type<uint>(r.zw[ulong(col)*(nl+1)+k]);bad|=(z&0x7fffffffu)>=0x7f800000u;if(k>=1&&soft32_greater(z,as_type<uint>(threshold)))stop=true;
if(k<nl){ulong j=ulong(col)*nl+k;uint d=as_type<uint>(r.dz[j]);bad|=d<0x00800000u || d>=0x7f800000u;
bad|=(as_type<uint>(r.u1[j])&0x7fffffffu)>=0x7f800000u;
bad|=(as_type<uint>(r.v1[j])&0x7fffffffu)>=0x7f800000u;
bad|=(as_type<uint>(r.qke[j])&0x7fffffffu)>=0x7f800000u;
bad|=(as_type<uint>(r.dtv[j])&0x7fffffffu)>=0x7f800000u;
bad|=(as_type<uint>(r.theta[j])&0x7fffffffu)>=0x7f800000u;
bad|=(as_type<uint>(r.edmf_w1[j])&0x7fffffffu)>=0x7f800000u;
bad|=(as_type<uint>(r.edmf_a1[j])&0x7fffffffu)>=0x7f800000u;
}}
r.invalid[col]=int(bad||!stop);
}
