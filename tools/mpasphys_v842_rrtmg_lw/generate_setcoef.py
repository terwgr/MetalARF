"""Translate native LW coefficient interpolation in source order."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.rrtmg_lw_v842 import _solver_source
source=_solver_source();m=re.search(r'extern "C" __global__ void rlw_setcoef\s*\(',source)
i=source.index('{',m.end());j=i+1;depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
args=source[m.end():i].rsplit(')',1)[0]
args=re.sub(r'//[^\n]*','',args)
resources=[]
for arg in args.split(','):
    item=arg.strip()
    if '*' not in item:continue
    match=re.fullmatch(r'(const\s+)?(float|int)\s*\*\s*__restrict__\s*(\w+)',item)
    if not match:raise ValueError(item)
    const,t,n=match.groups();resources.append([n,'float' if t=='float' else 'int32','read' if const else 'read_write'])
body=source[i+1:j-1].replace('int col = blockIdx.x * blockDim.x + threadIdx.x;','int col = int(gid);').replace('long long','long')
body=re.sub(r'(?<!device )const float\s*\*',r'device const float*',body)
body=body.replace('!(pavl > 0.0f)','!soft32_greater(as_type<uint>(pavl),0u)')
body=re.sub(r'OUT1\((colco2|colo3|coln2o|colco|colch4)\) == 0\.0f',r'soft32_equal(as_type<uint>(OUT1(\1)),0u)',body)
name='hexv842_mpas_rrtmg_lw_setcoef_f32'
header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "soft_gfk_log.metalh"
#include "hexv842_mpas_rrtmg_lw_setcoef_f32_bindings.metalh"
#pragma clang fp contract(off)
#define NBNDLW 16
#define MXMOL 38
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#define rlw_log(x) as_type<float>(soft_gfk_log(as_type<uint>(x)))
kernel void hexv842_mpas_rrtmg_lw_setcoef_f32(constant Hexv842MpasRrtmgLwSetcoefF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwSetcoefF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nlayers=p.nlayers,istart=p.istart;
'''
for n,t,access in resources:
    header+=f' device {"const " if access=="read" else ""}{"int" if t=="int32" else "float"}* {n}=r.{n};\n'
(root/f'woof/metal/shaders/{name}.metal').write_text(header+body+'\n}\n')
params=[[n,'int32'] for n in ('ncol','nlayers','istart')]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/core/kernels/rrtmg_lw.cu','cuda_symbol':'rlw_setcoef','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
