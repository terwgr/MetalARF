"""Translate the sequential native MPAS SW McICA wrapper to Metal."""
from pathlib import Path
import json,re
from woof.hex.mpasphys.rrtmg_sw_v842 import CUDA_SOURCE
root=Path(__file__).resolve().parents[2]
start=CUDA_SOURCE.index('__device__ float mpas_kiss(')
end=CUDA_SOURCE.index('// MPAS module_ra_rrtmg_lw.F:1751-1818',start)
helper=CUDA_SOURCE[start:end].replace('__device__ float','static inline float').replace('unsigned int&','thread uint&').replace('int kiss=(int)(s1+s2+(s3<<16)+s4);','int kiss=as_type<int>(s1+s2+(s3<<16)+s4);')
start=CUDA_SOURCE.index('extern "C" __global__ void mpas_sw_mcica(')
begin=CUDA_SOURCE.index('{',start);end=begin+1;depth=1
while depth:
    if CUDA_SOURCE[end]=='{':depth+=1
    elif CUDA_SOURCE[end]=='}':depth-=1
    end+=1
body=CUDA_SOURCE[begin+1:end-1].replace('int c=blockDim.x*blockIdx.x+threadIdx.x;','int c=int(tid);')
for old,new in {'FADD':'mpas_add','FSUB':'mpas_sub','FMUL':'mpas_mul','FDIV':'mpas_div','gfk_exp':'sw_exp'}.items():
    helper=re.sub(r'\b'+old+r'\b',new,helper);body=re.sub(r'\b'+old+r'\b',new,body)
shader='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "soft_gfk_exp.metalh"
#include "hexv842_sw_mcica_f32_bindings.metalh"
#pragma clang fp contract(off)
static inline float sw_exp(float x) {return as_type<float>(soft_gfk_exp(as_type<uint>(x)));}
'''+helper+'''
kernel void hexv842_sw_mcica_f32(constant Hexv842SwMcicaF32Resources &r [[buffer(0)]],
 constant Hexv842SwMcicaF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int nc=p.nc,nl=p.nl,icld=p.icld,idcor=p.idcor,julday=p.julday,irng=0,permuteseed=1;
 device const float *play=r.play,*cf=r.cf,*iw=r.iw,*lw=r.lw,*sw=r.sw,*hgt=r.hgt,*latitude=r.latitude,*uniform=r.uniform;
 device float *out=r.out;
'''+body+'\n}\n'
(root/'woof/metal/shaders/hexv842_sw_mcica_f32.metal').write_text(shader)
res=[(n,'float','read') for n in ('play','cf','iw','lw','sw','hgt','latitude','uniform')]+[('out','float','read_write')]
params=[(x,'int32') for x in ('nc','nl','icld','idcor','julday')]
k=dict(name='hexv842_sw_mcica_f32',module='hexv842_sw_mcica_f32',source='woof/metal/shaders/hexv842_sw_mcica_f32.metal',cuda_source='woof/hex/mpasphys/rrtmg_sw_v842.py',cuda_symbol='mpas_sw_mcica',resources=res,shapes={x[0]:[['any_1d']] for x in res},parameters=params,threads_per_group=128,layout_version=1,arithmetic_policies=['default'])
(root/'woof/metal/signature_fragments/hexv842_sw_mcica_f32.json').write_text(json.dumps({'kernels':[k]},indent=2)+'\n')
