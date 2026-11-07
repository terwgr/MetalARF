"""Translate the selected v842 sea-ice source into Metal packed kernels."""
from pathlib import Path
import re
from woof.hex.mpasphys.seaice_v842 import _CORE_SOURCE,FIELDS
ROOT=Path(__file__).resolve().parents[2]
source=_CORE_SOURCE
core=source[:source.index('#define P_t3d')]
core=re.sub(r'#if defined\(__CUDA_ARCH__\).*?#else\s*(.*?)#endif',r'\1',core,flags=re.S)
core=core.replace('__device__ inline ','static inline ').replace('__device__ ','static ')
core=core.replace('constexpr float','constant float')
for typ in ('float','int','bool'):
    core=re.sub(r'\b'+typ+r'\s*\*',f'thread {typ} *',core)
    core=re.sub(r'\b'+typ+r'\s*&',f'thread {typ} &',core)

def translate(s):
    for old,new in {'__fadd_rn':'mpas_add','__fsub_rn':'mpas_sub','__fmul_rn':'mpas_mul',
                    '__fdiv_rn':'mpas_div','__float_as_uint':'as_type<uint>',
                    '__uint_as_float':'as_type<float>','mynn_glibc_powf':'seaice_pow',
                    'mynn_glibc_expf':'seaice_exp','fmaxf':'fmax','fminf':'fmin',
                    'fabsf':'fabs'}.items():
        s=re.sub(r'\b'+old+r'\b',new,s)
    return s
core=translate(core)

def body(name):
    start=source.index('extern "C" __global__ void '+name+'(')
    i=source.index('{',start)+1; begin=i; depth=1
    while depth:
        if source[i]=='{': depth+=1
        elif source[i]=='}': depth-=1
        i+=1
    text=source[begin:i-1]
    text=re.sub(r'^\s*int i\s*=\s*blockDim\.x\s*\*\s*blockIdx\.x\s*\+\s*threadIdx\.x\s*;','',text,count=1)
    text=text.replace('a[P_p8w3d][n+i]','r.p8w2[i]')
    text=text.replace('a[P_tslb][(k-1)*n+i]','r.soil[(k-1)*n+i]')
    text=re.sub(r'F\((\w+)\)',lambda m:f'r.fields[{FIELDS.index(m.group(1))}*n+i]',text)
    text=text.replace('atomicMax(last_sflx_column, i);','r.ran[i]=1;')
    text=text.replace('errors[i]','r.errors[i]')
    return translate(text)

prefix='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_seaice_math.metalh"
#include "hexv842_seaice_from_f32_bindings.metalh"
#include "hexv842_seaice_column_f32_bindings.metalh"
#include "hexv842_seaice_to_f32_bindings.metalh"
#include "hexv842_seaice_state_i32_bindings.metalh"
#pragma clang fp contract(off)
#define NSOLD 9
'''
params='''    int i=int(tid),n=p.n,nsoil=p.nsoil;
    float dt=p.dt,threshold=p.threshold,albedo_default=p.albedo_default;
    float thickness_default=p.thickness_default,snowdepth_max=p.snowdepth_max,snowdepth_min=p.snowdepth_min;
    int albedo_opt=p.albedo_opt,thickness_opt=p.thickness_opt,snowdepth_opt=p.snowdepth_opt;
    int frpcpn=p.frpcpn,optional_noah=p.optional_noah,microp=p.microp,convection=p.convection;
'''
shader=prefix+core
for opname,entry,title in [('from','seaice_from_MPAS','From'),('column','seaice_noah','Column'),('to','seaice_to_MPAS','To')]:
    shader+=f'''\nkernel void hexv842_seaice_{opname}_f32(
    constant Hexv842Seaice{title}F32Resources &r [[buffer(0)]],
    constant Hexv842Seaice{title}F32Params &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{
'''+params+body(entry)+'\n}\n'
shader+='''
kernel void hexv842_seaice_state_i32(constant Hexv842SeaiceStateI32Resources &r [[buffer(0)]],
    constant Hexv842SeaiceStateI32Params &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if(tid) return;
    int last=-1; for(int i=0;i<p.n;i++) if(r.ran[i]) last=i;
    if(last>=0) { r.state[0]=p.i_start+last; r.state[1]=p.j_index; }
}
'''
(ROOT/'woof/metal/shaders/hexv842_seaice_f32.metal').write_text(shader)
