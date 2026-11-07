"""Translate the native longwave wrapper's KISS McICA path to Metal."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.rrtmg_lw_v842 import _DRIVER_SOURCE as source
macros=source[source.index('#define M(f,k)'):source.index('// MPAS: module_ra_rrtmg_lw.F:11954-11983.')]
def function(name):
    m=re.search(r'__device__ (?:float|void) '+name+r'\s*\(',source)
    i=source.index('{',m.end());j=i+1;depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    return source[m.start():j]
k= function('mpas_kiss').replace('__device__','static inline').replace('unsigned int* seed','thread uint* seed').replace('int kiss=(int)(seed[0]+seed[1]+(seed[2]<<16)+seed[3]);','int kiss=as_type<int>(seed[0]+seed[1]+(seed[2]<<16)+seed[3]);')
f=function('mpas_mcica_make')
f=f.replace('int irng,int permuteseed,const float* cdf_input,const float* cdf2_input,int cdf_stride,\n    const float* tauc,const int* ngb,float* tau,\n    ','')
f=f.replace('unsigned int seed[4];mpas_mt_sequence mt;','unsigned int seed[4];')
f=f.replace('if(irng==0){','{').replace('for(int j=0;j<permuteseed;++j)mpas_kiss(seed);','for(int j=0;j<150;++j)mpas_kiss(seed);').replace('}else if(!cdf_input)mpas_mt_scalar(mt,(unsigned int)permuteseed);','}')
f=f.replace('cdf_input?cdf_input[ig*cdf_stride]:(irng==0?mpas_kiss(seed):mpas_mt_real(mt))','mpas_kiss(seed)')
f=f.replace('cdf_input?cdf_input[ig*cdf_stride+k]:(irng==0?mpas_kiss(seed):mpas_mt_real(mt))','mpas_kiss(seed)')
f=f.replace('cdf2_input?cdf2_input[ig*cdf_stride+k]:(irng==0?mpas_kiss(seed):mpas_mt_real(mt))','mpas_kiss(seed)')
f=f.replace('            if(tau)tau[idx]=cloudy?tauc[(ngb[ig]-1)*nl+k]:0.0f;\n','')
f=f.replace('__device__','static inline')
f=re.sub(r'const float\s*\*',r'device const float*',f)
f=re.sub(r'(?<!const )\bfloat\s*\*',r'device float*',f)
for token in ('irng','cdf_input','cdf2_input','mpas_mt','tauc','ngb','tau['):
    if token in f:raise ValueError(token)
name='hexv842_mpas_rrtmg_lw_mcica_f32'
header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "soft_gfk_exp.metalh"
#include "hexv842_mpas_rrtmg_lw_mcica_f32_bindings.metalh"
#pragma clang fp contract(off)
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#define rlw_exp(x) as_type<float>(soft_gfk_exp(as_type<uint>(x)))
'''
entry='''kernel void hexv842_mpas_rrtmg_lw_mcica_f32(constant Hexv842MpasRrtmgLwMcicaF32Resources &r [[buffer(0)]], constant Hexv842MpasRrtmgLwMcicaF32Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
if(c>=uint(p.nc))return;
mpas_mcica_make(p.nl,p.overlap,p.idcor,p.julday,r.surface+ulong(c)*6,r.a+ulong(c)*12*p.nl,r.cloud+ulong(c)*p.nl*140,r.liquid+ulong(c)*p.nl*140,r.ice+ulong(c)*p.nl*140,r.snow+ulong(c)*p.nl*140);
}
'''
(root/f'woof/metal/shaders/{name}.metal').write_text(header+macros+'\n'+k+'\n'+f+'\n'+entry)
resources=[[n,'float','read'] for n in ('surface','a')]+[[n,'float','read_write'] for n in ('cloud','liquid','ice','snow')]
params=[[n,'int32'] for n in ('nc','nl','overlap','idcor','julday')]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/rrtmg_lw_v842.py','cuda_symbol':'mpas_lw_mcica','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
