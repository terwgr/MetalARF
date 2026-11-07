#include <metal_stdlib>
using namespace metal;
#include "soft_binary32.metalh"
#pragma clang fp contract(off)
static inline float tc_binary(float a,float b,int op) {
 uint x=as_type<uint>(a),y=as_type<uint>(b),z;
 if(soft32_is_nan(x))return as_type<float>(soft32_quiet_nan(x));
 if(soft32_is_nan(y))return as_type<float>(soft32_quiet_nan(y));
 z=op==0?soft32_add(x,y):op==1?soft32_sub(x,y):op==2?soft32_mul(x,y):soft32_div(x,y);
 if(soft32_is_nan(z))z=0xffc00000u;
 return as_type<float>(z);
}
static inline float tc_add(float a,float b){return tc_binary(a,b,0);}
static inline float tc_sub(float a,float b){return tc_binary(a,b,1);}
static inline float tc_mul(float a,float b){return tc_binary(a,b,2);}
static inline float tc_div(float a,float b){return tc_binary(a,b,3);}
static inline float tc_max(float a,float b){return soft32_greater(as_type<uint>(a),as_type<uint>(b))?a:b;}
static inline float tc_min(float a,float b){return soft32_less(as_type<uint>(a),as_type<uint>(b))?a:b;}
static inline float rslf(float p,float t) {
float result;
float esl;
float x;
float c0 = .611583699e03f;
float c1 = .444606896e02f;
float c2 = .143177157e01f;
float c3 = .264224321e-1f;
float c4 = .299291081e-3f;
float c5 = .203154182e-5f;
float c6 = .702620698e-8f;
float c7 = .379534310e-11f;
float c8 = (-.321582393e-13f);
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:171
x = tc_max((-80.f),tc_sub(t,273.16f));
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:174
esl = tc_add(c0,tc_mul(x,(tc_add(c1,tc_mul(x,(tc_add(c2,tc_mul(x,(tc_add(c3,tc_mul(x,(tc_add(c4,tc_mul(x,(tc_add(c5,tc_mul(x,(tc_add(c6,tc_mul(x,(tc_add(c7,tc_mul(x,c8)))))))))))))))))))))));
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:175
esl = tc_min(esl,tc_mul(p,0.15f));
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:176
result = tc_div(tc_mul(.622f,esl),(tc_sub(p,esl)));
return result;
}
static inline float rsif(float p,float t) {
float result;
float esi;
float x;
float c0 = .609868993e03f;
float c1 = .499320233e02f;
float c2 = .184672631e01f;
float c3 = .402737184e-1f;
float c4 = .565392987e-3f;
float c5 = .521693933e-5f;
float c6 = .307839583e-7f;
float c7 = .105785160e-9f;
float c8 = .161444444e-12f;
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:206
x = tc_max((-80.f),tc_sub(t,273.16f));
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:207
esi = tc_add(c0,tc_mul(x,(tc_add(c1,tc_mul(x,(tc_add(c2,tc_mul(x,(tc_add(c3,tc_mul(x,(tc_add(c4,tc_mul(x,(tc_add(c5,tc_mul(x,(tc_add(c6,tc_mul(x,(tc_add(c7,tc_mul(x,c8)))))))))))))))))))))));
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:208
esi = tc_min(esi,tc_mul(p,0.15f));
// MPAS src/core_atmosphere/physics/mpas_atmphys_functions.F:209
result = tc_div(tc_mul(.622f,esi),(tc_sub(p,esi)));
return result;
}
#include "hexv842_mpas_thompson_column_relhum_f32_bindings.metalh"
kernel void hexv842_mpas_thompson_column_relhum_f32(constant Hexv842MpasThompsonColumnRelhumF32Resources &r [[buffer(0)]], constant Hexv842MpasThompsonColumnRelhumF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.n))return;
 float t=tc_mul(r.th[i],r.pii[i]);
 float tempc=tc_sub(t,273.16f);
 float qvs=rslf(r.pressure[i],t);
 if(soft32_less_equal(as_type<uint>(tempc),0u))qvs=rsif(r.pressure[i],t);
 float relhum=tc_div(r.qv[i],qvs);
 r.relhum[i]=tc_mul(relhum,100.0f);
}
