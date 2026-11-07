from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.rrtmg_lw_v842 import _DRIVER_SOURCE as source
macros=source[source.index('#define M(f,k)'):source.index('// MPAS: module_ra_rrtmg_lw.F:11954-11983.')]
m=re.search(r'__device__ void mpas_lw_outputs_one\s*\(',source)
i=source.index('{',m.end());j=i+1;depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
function=source[m.start():j].replace('__device__','static inline')
function=re.sub(r'const float\s*\*',r'device const float*',function)
function=re.sub(r'(?<!const )\bfloat\s*\*',r'device float*',function)
function=function.replace('M(7,k)>0.0f','soft32_greater(as_type<uint>(M(7,k)),0u)')
header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_DV(a,b) cu_div(a,b)
'''
specs=[('counts',[['interfaces','float','read'],['counts','int32','read_write']],[['nc','int32'],['nm','int32']],'''float top=RLW_DV(r.interfaces[ulong(c)*2*(p.nm+1)+p.nm],100.0f);
r.counts[c]=p.nm+max(int(RLW_AD(RLW_DV(top,4.0f),0.5f)),1);''','mpas_layer_counts'),
('outputs',[[n,'float','read'] for n in ('m','a','u','d','uc','dc','hr')]+[[n,'float','read_write'] for n in ('tendency','radii','flux')],[[n,'int32'] for n in ('nc','nm','nl','icloud')],'''mpas_lw_outputs_one(p.nm,p.nl,p.icloud,r.m+ulong(c)*12*p.nm,r.a+ulong(c)*12*p.nl,r.u+ulong(c)*(p.nl+1),r.d+ulong(c)*(p.nl+1),r.uc+ulong(c)*(p.nl+1),r.dc+ulong(c)*(p.nl+1),r.hr+ulong(c)*p.nl,r.tendency+ulong(c)*p.nm,r.radii+ulong(c)*3*p.nm,r.flux+ulong(c)*11);''','mpas_lw_outputs')]
for suffix,resources,params,body,symbol in specs:
    name='hexv842_mpas_rrtmg_lw_'+suffix+'_f32'
    typename='Hexv842MpasRrtmgLw'+suffix.title()+'F32'
    code=header+f'#include "{name}_bindings.metalh"\n'
    if suffix=='outputs':code+=macros+'\n'+function+'\n'
    code+=f'kernel void {name}(constant {typename}Resources &r [[buffer(0)]], constant {typename}Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {{\nif(c>=uint(p.nc))return;\n{body}\n}}\n'
    (root/f'woof/metal/shaders/{name}.metal').write_text(code)
    spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/rrtmg_lw_v842.py','cuda_symbol':symbol,'resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
    (root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
