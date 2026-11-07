#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "soft_gfk_exp.metalh"
#include "hexv842_mpas_rrtmg_lw_prol_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_rrtmg_lw_prol_f32(constant Hexv842MpasRrtmgLwProlF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwProlF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 if(ulong(tid)>=ulong(p.ncol)*p.nl)return;
 uint col=tid/p.nl; ulong base=ulong(tid)*140; int flag=0;
 for(int ig=0;ig<140;++ig){
  ulong index=base+ig;
  if(r.cloud[index]==1.0f){
   int ib=r.ngb[ig]-1;
   float od=mpas_mul(r.secdiff[ulong(col)*16+ib],r.tau[index]);
   r.odcld[index]=od;
   float trans=as_type<float>(soft_gfk_exp(as_type<uint>(od)^0x80000000u));
   float absorption=mpas_sub(1.0f,trans);
   r.efclfrac[index]=mpas_mul(absorption,r.cloud[index]); flag=1;
  }else{r.odcld[index]=0.0f;r.efclfrac[index]=0.0f;}
 }
 r.icldlyr[tid]|=flag;
}
