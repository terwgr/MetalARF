#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "soft_glibc_solar_trig.metalh"
#include "hexv842_aerosol_start_f32_bindings.metalh"
using namespace ntc;
#pragma clang fp contract(off)
static inline F mid(device const float *z,int k){return tc_mul(F(.5f),tc_add(F(z[k]),F(z[k+1])));}
static inline void profile(device const float *z,device float *out,int nz,F lg,F n0,F n1){
 F h=mid(z,0), h01= h<=F(1000.f)?F(.8f):h>=F(2500.f)?F(.01f):tc_mul(F(.8f),f32_word(soft_glibc_solar_cosf(tc_sub(tc_mul(h,F(.001f)),F(1.f)).bits)));
 F scale=tc_div(lg,h01);
 for(int k=0;k<nz;++k){F dh=tc_sub(mid(z,k==0?1:k),h);F arg=-tc_mul(tc_div(dh,F(1000.f)),scale);out[k]=as_type<float>(tc_add(n1,tc_mul(n0,tc_exp(arg))).bits);}
}
kernel void hexv842_aerosol_start_f32(constant Hexv842AerosolStartF32Resources &r [[buffer(0)]],constant Hexv842AerosolStartF32Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]){
 if(i>=uint(p.ns))return;
 device const float *z=r.zgrid+i*(p.nz+1);device float *w=r.nwfa+i*p.nz;device float *a=r.nifa+i*p.nz;
 if(p.w_on)profile(z,w,p.nz,F(p.w_log),F(p.w0),F(p.w1));
 if(p.i_on)profile(z,a,p.nz,F(p.i_log),F(p.i0),F(p.i1));
 F mass=tc_mul(tc_mul(tc_mul(F(r.rho[i]),F(r.zz[i])),tc_sub(F(z[1]),F(z[0]))),F(r.area[i]));
 r.nwfa2d[i]=as_type<float>(tc_mul(tc_mul(tc_mul(F(w[0]),F(.000196f)),mass),F(.5e-10f)).bits);
}
