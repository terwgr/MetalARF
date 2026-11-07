from pathlib import Path
import re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.rrtmg_lw_v842 import _DRIVER_SOURCE
s=_DRIVER_SOURCE
constant=[]
for name in ('retab','pprof','tprof','o3sum','ppsum','o3win','ppwin'):
 m=re.search(r'__device__ const float mpas_'+name+r'\[\d+\] = \{.*?\};',s,re.S)
 if not m:raise ValueError(name)
 constant.append(m.group().replace('__device__ const','constant'))
macros=s[s.index('#define M(f,k)'):s.index('// MPAS: module_ra_rrtmg_lw.F:11954-11983.')]
def function(name):
 m=re.search(r'__device__ (?:float|void) '+name+r'\s*\(',s)
 i=s.index('{',m.end());j=i+1;depth=1
 while depth:
  if s[j]=='{':depth+=1
  elif s[j]=='}':depth-=1
  j+=1
 return s[m.start():j]
funcs='\n'.join(function(name) for name in ('mpas_profile','mpas_ice_radius','mpas_liquid_radius','mpas_vinterp_one','mpas_lw_prepare_one'))
# The native interpolation's one-level lower read is undefined. The caller
# requires at least two levels, so this helper uses the defined lower word.
funcs=re.sub(r'    // MPAS: module_ra_rrtmg_vinterp.F:155-161\..*?    float lower_pin=', '    float lower_pin=', funcs, flags=re.S)
funcs=funcs.replace('(*upper+1<nz)?pin[*upper+1]:0.0f','pin[*upper+1]').replace('(*upper+1<nz)?ozone[*upper+1]:0.0f','ozone[*upper+1]')
funcs=funcs.replace('__device__','static inline').replace('fminf(', 'mpas_min(').replace('fmaxf(', 'mpas_max(').replace('cf>0.0f', 'soft32_greater(as_type<uint>(cf),0u)')
funcs=re.sub(r'const float\s*\*',r'device const float*',funcs)
funcs=re.sub(r'(?<!const )\bfloat\s*\*',r'device float*',funcs)
funcs=funcs.replace('int* upper','thread int* upper')
header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#include "hexv842_mpas_rrtmg_lw_prepare_f32_bindings.metalh"
'''
entry='''kernel void hexv842_mpas_rrtmg_lw_prepare_f32(constant Hexv842MpasRrtmgLwPrepareF32Resources &r [[buffer(0)]], constant Hexv842MpasRrtmgLwPrepareF32Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
if(c>=uint(p.nc))return;
mpas_lw_prepare_one(p.nm,p.nl,p.icloud,p.reqc,p.reqi,p.reqs,p.o3input,p.nz,p.presence,r.m+ulong(c)*12*p.nm,r.interfaces+ulong(c)*2*(p.nm+1),r.surface+ulong(c)*6,r.pin,r.ozone+(p.o3input==2?ulong(c)*p.nz:0),r.a+ulong(c)*12*p.nl,r.lev+ulong(c)*2*(p.nl+1));
}
'''
name='hexv842_mpas_rrtmg_lw_prepare_f32'
(root/f'woof/metal/shaders/{name}.metal').write_text(header+'\n'.join(constant)+'\n'+macros+'\n'+funcs+'\n'+entry)
import json
resources=[['m','float','read'],['interfaces','float','read'],['surface','float','read'],['pin','float','read'],['ozone','float','read'],['a','float','read_write'],['lev','float','read_write']]
params=[[name,'int32'] for name in ('nc','nm','nl','icloud','reqc','reqi','reqs','o3input','nz','presence')]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/rrtmg_lw_v842.py','cuda_symbol':'mpas_lw_prepare','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
