"""Generate the native LW per-layer cloud transfer preparation."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
name='hexv842_mpas_rrtmg_lw_prol_f32'
shader='''#include <metal_stdlib>
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
'''
(root/f'woof/metal/shaders/{name}.metal').write_text(shader)
resources=[[n,'float','read'] for n in ('cloud','tau','secdiff')]+[['ngb','int32','read'],['odcld','float','read_write'],['efclfrac','float','read_write'],['icldlyr','int32','read_write']]
params=[[n,'int32'] for n in ('ncol','nl')]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/core/kernels/rrtmg_lw_zbatched.cu','cuda_symbol':'rlw_rtrn_prol_gpoints','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
