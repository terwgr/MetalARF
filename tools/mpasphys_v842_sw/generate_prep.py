"""Translate native MPAS SW sounding preparation to its Metal kernel."""
from pathlib import Path
import json,re
from woof.hex.mpasphys.rrtmg_sw_v842 import CUDA_SOURCE
root=Path(__file__).resolve().parents[2]
arrays=[]
for name in ('MPAS_RETAB','MPAS_O3SUM','MPAS_O3WIN','MPAS_PPSUM','MPAS_PPWIN'):
    arrays.append(re.search(r'__device__ const float '+name+r'\[\d+\]\s*=\s*\{[^}]*\};',CUDA_SOURCE).group().replace('__device__ const','constant'))
start=CUDA_SOURCE.index('__device__ float mpas_ice_radius(')
end=CUDA_SOURCE.index('// MPAS module_ra_rrtmg_sw.F:10197-10550',start)
helpers=CUDA_SOURCE[start:end].replace('half[','half_press[').replace('__device__ float','static inline float').replace('const float*','device const float*').replace('int& upper','thread int& upper')
start=CUDA_SOURCE.index('extern "C" __global__ void mpas_sw_prep(')
begin=CUDA_SOURCE.index('{',start);end=begin+1;depth=1
while depth:
    if CUDA_SOURCE[end]=='{':depth+=1
    elif CUDA_SOURCE[end]=='}':depth-=1
    end+=1
body=CUDA_SOURCE[begin+1:end-1].replace('int c=blockDim.x*blockIdx.x+threadIdx.x;','int c=int(tid);').replace('cf>0.f','soft32_greater(as_type<uint>(cf),0u)')
for old,new in {'FADD':'mpas_add','FSUB':'mpas_sub','FMUL':'mpas_mul','FDIV':'mpas_div','fmaxf':'mpas_max','fminf':'mpas_min','glibc_sinf':'sw_sin','glibc_cosf':'sw_cos','fmodf':'sw_fmod1440'}.items():
    helpers=re.sub(r'\b'+old+r'\b',new,helpers)
    body=re.sub(r'\b'+old+r'\b',new,body)
shader='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "soft_glibc_solar_trig.metalh"
#include "hexv842_sw_prep_f32_bindings.metalh"
#pragma clang fp contract(off)
static inline float sw_sin(float x) {return as_type<float>(soft_glibc_solar_sinf(as_type<uint>(x)));}
static inline float sw_cos(float x) {return as_type<float>(soft_glibc_solar_cosf(as_type<uint>(x)));}
// Binary32 remainder by positive 1440, retaining signed zero and subnormals.
static inline float sw_fmod1440(float value,float divisor) {
 uint bits=as_type<uint>(value),mag=bits&0x7fffffffu,sgn=bits&0x80000000u;
 if(mag<0x44b40000u)return value;
 if(mag>=0x7f800000u)return as_type<float>(0xffc00000u);
 int ex=int(mag>>23),ey=137;uint x=(mag&0x7fffffu)|0x800000u,y=0xb40000u;
 while(ex>ey){if(x>=y)x-=y;x<<=1;--ex;}
 if(x>=y)x-=y;
 if(!x)return as_type<float>(sgn);
 while(x<0x800000u){x<<=1;--ey;}
 return as_type<float>(sgn|(uint(ey)<<23)|(x&0x7fffffu));
}
'''+ '\n'.join(arrays)+'\n'+helpers+'''
kernel void hexv842_sw_prep_f32(constant Hexv842SwPrepF32Resources &r [[buffer(0)]],
 constant Hexv842SwPrepF32Params &params [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int nc=params.nc,nz=params.nz,icloud=params.icloud,o3input=params.o3input,reqc=params.reqc,reqi=params.reqi,reqs=params.reqs,nozn=params.nozn;
 float xtime=r.controls[0],gmt=params.gmt,radt=r.controls[1],degrad=params.degrad,declin=r.controls[2];
 device const float *fields=r.fields,*surf=r.surf,*pw=r.pw,*tw=r.tw,*pin=r.pin,*oz=r.oz;
 device float *layers=r.layers,*interfaces=r.interfaces,*hgt=r.hgt,*coszr=r.coszr;
'''+body+'\n}\n'
(root/'woof/metal/shaders/hexv842_sw_prep_f32.metal').write_text(shader)
res=[(n,'float','read') for n in ('fields','surf','pw','tw','pin','oz','controls')]+[(n,'float','read_write') for n in ('layers','interfaces','hgt','coszr')]
params=[(x,'int32') for x in ('nc','nz','icloud','o3input','reqc','reqi','reqs','nozn')]+[(x,'float32') for x in ('gmt','degrad')]
k=dict(name='hexv842_sw_prep_f32',module='hexv842_sw_prep_f32',source='woof/metal/shaders/hexv842_sw_prep_f32.metal',cuda_source='woof/hex/mpasphys/rrtmg_sw_v842.py',cuda_symbol='mpas_sw_prep',resources=res,shapes={x[0]:[['any_1d']] for x in res},parameters=params,threads_per_group=128,layout_version=1,arithmetic_policies=['default'])
(root/'woof/metal/signature_fragments/hexv842_sw_prep_f32.json').write_text(json.dumps({'kernels':[k]},indent=2)+'\n')
