#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "soft_gfk_exp.metalh"
#pragma clang fp contract(off)
#include "hexv842_mpas_rrtmg_lw_secdiff_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_secdiff_f32(constant Hexv842MpasRrtmgLwSecdiffF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwSecdiffF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
if(tid>=uint(p.ncol))return;
float pwvcm=r.pwvcm[tid];
for(int ibnd=1;ibnd<=16;++ibnd){
 float sd;
 if(ibnd==1||ibnd==4||ibnd>=10)sd=1.66f;
 else {
  float exponent=mpas_mul(r.a2[ibnd-1],pwvcm);
  sd=mpas_add(r.a0[ibnd-1],mpas_mul(r.a1[ibnd-1],as_type<float>(soft_gfk_exp(as_type<uint>(exponent)))));
  if(sd>1.80f)sd=1.80f;
  if(sd<1.50f)sd=1.50f;
 }
 r.secdiff[ulong(tid)*16+(ibnd-1)]=sd;
}
}
