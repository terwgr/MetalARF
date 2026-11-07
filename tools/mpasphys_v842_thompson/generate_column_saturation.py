"""Generate native Thompson saturation helpers from the pinned column source."""
from pathlib import Path
import json
from woof.hex.mpasphys import thompson_v842 as native
root=Path(__file__).resolve().parents[2]

def method(name):
    source=native._COLUMN
    start=source.index('__device__ __forceinline__ float '+name+'(')
    brace=source.index('{',start);end=native._body_end(source,brace)
    return source[start:end].replace('__device__ __forceinline__','static inline')

name='hexv842_mpas_thompson_column_saturation_f32'
header='''#include <metal_stdlib>
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
'''
code=header+method('rslf')+'\n'+method('rsif')+f'''\n#include "{name}_bindings.metalh"
kernel void {name}(constant Hexv842MpasThompsonColumnSaturationF32Resources &r [[buffer(0)]],constant Hexv842MpasThompsonColumnSaturationF32Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]) {{
 if(i>=uint(p.n))return;
 r.liquid[i]=rslf(r.pressure[i],r.temperature[i]);
 r.ice[i]=rsif(r.pressure[i],r.temperature[i]);
}}
'''
(root/f'woof/metal/shaders/{name}.metal').write_text(code)
resources=[['pressure','float','read'],['temperature','float','read'],['liquid','float','read_write'],['ice','float','read_write']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/thompson_v842.py','cuda_symbol':'rslf,rsif','resources':resources,'shapes':{r[0]:[['any_1d']] for r in resources},'parameters':[['n','int32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
