"""Generate the live native Thompson aerosol shortwave path."""
from pathlib import Path
import json,re
from woof.hex.mpasphys.rrtmg_sw_v842 import CUDA_SOURCE
root=Path(__file__).resolve().parents[2]
def declaration(name):
    a=CUDA_SOURCE.index('__device__ const float '+name);b=CUDA_SOURCE.index(';',a)+1
    return CUDA_SOURCE[a:b].replace('__device__ const float','constant float')
arrays='\n'.join(declaration(n) for n in ('MPAS_RAOD','MPAS_SSA','MPAS_ASY','MPAS_RHS','AER3_GT_RH_ARR','AER3_GT_LOOKUP'))
a=CUDA_SOURCE.index('static __device__ __forceinline__ float aer3_rslf(')
b=CUDA_SOURCE.index('// MPAS module_ra_rrtmg_sw.F:10026-10664',a)
helpers=CUDA_SOURCE[a:b].replace('static __device__ __forceinline__','static inline').replace('static __device__','static inline')
a=CUDA_SOURCE.index('__device__ float mpas_sw_rh(');b=CUDA_SOURCE.index('// MPAS module_ra_rrtmg_sw_aerosols.F:70-451',a)
helpers+='\n'+CUDA_SOURCE[a:b].replace('__device__ float','static inline float').replace('const float table[3][8][14]','constant float table[3][8][14]')
for old,new in {'FADD':'mpas_add','FSUB':'mpas_sub','FMUL':'mpas_mul','FDIV':'mpas_div','fmaxf':'mpas_max','fminf':'mpas_min','roundf':'round','gfk_exp':'sw_exp'}.items():helpers=re.sub(r'\b'+old+r'\b',new,helpers)
s='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "soft_gfk_exp.metalh"
#include "hexv842_sw_gt_aod_f32_bindings.metalh"
#include "hexv842_sw_aerosol_f32_bindings.metalh"
#pragma clang fp contract(off)
#define AER3_NRH 8
static inline float sw_exp(float x) {return as_type<float>(soft_gfk_exp(as_type<uint>(x)));}
'''+arrays+'\n'+helpers+'''
kernel void hexv842_sw_gt_aod_f32(constant Hexv842SwGtAodF32Resources &r [[buffer(0)]],
 constant Hexv842SwGtAodF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int c=int(tid),nz=p.nz;if(c>=p.nc)return;
 if(r.xland[c]==1.f)r.type[c]=1;else if(r.xland[c]==2.f)r.type[c]=3;else {r.invalid[c]=1;return;}
 float sum=0.f;for(int k=0;k<nz;k++){int i=c*nz+k;r.aod3[i]=aer3_gt_aod(r.pressure[i],r.dz[i],r.temperature[i],r.qv[i],r.nwfa[i],r.nifa[i]);sum=mpas_add(sum,r.aod3[i]);}r.aod2[c]=sum;
}
kernel void hexv842_sw_aerosol_f32(constant Hexv842SwAerosolF32Resources &r [[buffer(0)]],
 constant Hexv842SwAerosolF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int c=int(tid),nz=p.nz,nl=p.nl;if(c>=p.nc)return;
 for(int k=0;k<nl;k++){
  if(k>=nz){for(int b=0;b<14;b++){int i=(c*14+b)*nl+k;r.tau[i]=0.f;r.ssa[i]=1.f;r.asy[i]=0.f;}continue;}
  float rh=mpas_sw_rh(r.pressure[c*nz+k],r.temperature[c*nz+k],r.qv[c*nz+k]);
  for(int b=0;b<14;b++){
   int i=(c*14+b)*nl+k;
   r.tau[i]=mpas_sw_spectral(MPAS_RAOD,r.type[c],b,rh,r.aod3[c*nz+k],true);
   r.ssa[i]=mpas_sw_spectral(MPAS_SSA,r.type[c],b,rh,0.f,false);
   r.asy[i]=mpas_sw_spectral(MPAS_ASY,r.type[c],b,rh,0.f,false);
  }
 }
}
'''
(root/'woof/metal/shaders/hexv842_sw_aerosol_f32.metal').write_text(s)
kernels=[]
for name,res,params,symbol in [
 ('gt_aod',[(n,'float','read') for n in ('pressure','temperature','qv','dz','nwfa','nifa','xland')]+[(n,'float','read_write') for n in ('aod3','aod2')]+[(n,'int32','read_write') for n in ('type','invalid')],['nc','nz'],'mpas_sw_gt_aod'),
 ('aerosol',[(n,'float','read') for n in ('pressure','temperature','qv','aod3')]+[('type','int32','read')]+[(n,'float','read_write') for n in ('tau','ssa','asy')],['nc','nz','nl'],'mpas_sw_aerosol')]:
 kernel='hexv842_sw_'+name+'_f32';kernels.append(dict(name=kernel,module='hexv842_sw_aerosol_f32',source='woof/metal/shaders/hexv842_sw_aerosol_f32.metal',cuda_source='woof/hex/mpasphys/rrtmg_sw_v842.py',cuda_symbol=symbol,resources=res,shapes={r[0]:[['any_1d']] for r in res},parameters=[(x,'int32') for x in params],threads_per_group=128,layout_version=1,arithmetic_policies=['default']))
(root/'woof/metal/signature_fragments/hexv842_sw_aerosol_f32.json').write_text(json.dumps({'kernels':kernels},indent=2)+'\n')
