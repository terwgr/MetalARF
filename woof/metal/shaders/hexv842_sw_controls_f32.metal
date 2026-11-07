#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "soft_glibc_solar_trig.metalh"
#include "hexv842_sw_controls_f32_bindings.metalh"
#pragma clang fp contract(off)
static inline float sw_sin(float v) { return as_type<float>(soft_glibc_solar_sinf(as_type<uint>(v))); }
static inline float sw_cos(float v) { return as_type<float>(soft_glibc_solar_cosf(as_type<uint>(v))); }
static inline float sw_asin(float v) { return as_type<float>(soft_glibc_solar_asinf(as_type<uint>(v))); }
kernel void hexv842_sw_controls_f32(constant Hexv842SwControlsF32Resources &r [[buffer(0)]],
    constant Hexv842SwControlsF32Params &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
 if(tid)return;
 r.out[0]=mpas_div(p.seconds,60.f);r.out[1]=mpas_div(p.interval,60.f);
 float sinob=sw_sin(mpas_mul(23.5f,p.degrad));
 float sx=mpas_mul(p.dpd,p.julian>=80.f?mpas_sub(p.julian,80.f):mpas_add(p.julian,285.f));sx=mpas_mul(sx,p.degrad);
 r.out[2]=sw_asin(mpas_mul(sinob,sw_sin(sx)));
 float rjul=mpas_mul(mpas_div(mpas_mul(p.julian,360.f),365.f),p.degrad);
 float eccentricity=mpas_add(1.000110f,mpas_mul(.034221f,sw_cos(rjul)));
 eccentricity=mpas_add(eccentricity,mpas_mul(.001280f,sw_sin(rjul)));
 eccentricity=mpas_add(eccentricity,mpas_mul(.000719f,sw_cos(mpas_mul(2.f,rjul))));
 eccentricity=mpas_add(eccentricity,mpas_mul(.000077f,sw_sin(mpas_mul(2.f,rjul))));
 r.out[3]=mpas_mul(1370.f,eccentricity);
}
