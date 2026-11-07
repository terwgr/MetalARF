"""Translate the native coalesced longwave cloud optics stage to Metal."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.core.rrtmg_lw import _lw_coalesced_source
source=_lw_coalesced_source((root/'woof/core/kernels/rrtmg_lw_chain.cu').read_text())
m=re.search(r'extern "C" __global__ void rlw_cldprmc_coalesced\s*\(',source)
i=source.index('{',m.end());j=i+1;depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
body=source[i+1:j-1].replace('long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;','long long tid = (long long)gid;')
body=body.replace('long long','long')
body=body.replace('atomicExch(err_flag, ', 'atomic_exchange_explicit((device atomic_int*)err_flag, ').replace('atomicExch(&ncb_flag[col], 1)', 'atomic_exchange_explicit((device atomic_int*)&ncb_flag[col], 1, memory_order_relaxed)')
body=re.sub(r'atomic_exchange_explicit\(\(device atomic_int\*\)err_flag, (\d)\)',r'atomic_exchange_explicit((device atomic_int*)err_flag, \1, memory_order_relaxed)',body)
body=body.replace('ice_plus_snow == 0.0f','soft32_equal(as_type<uint>(ice_plus_snow),0u)').replace('MC(cswpmc, lay) > 0.0f','soft32_greater(as_type<uint>(MC(cswpmc, lay)),0u)').replace('MC(clwpmc, lay) == 0.0f','soft32_equal(as_type<uint>(MC(clwpmc, lay)),0u)')
if 'atomicExch' in body:raise ValueError('unconverted atomic')
name='hexv842_mpas_rrtmg_lw_cldpr_f32'
resources=[[n,'float','read'] for n in ('cldfmc','ciwpmc','clwpmc','cswpmc','reicmc','relqmc','resnmc','absice1','absice2','absice3','absice0','absliq1')]+[['ngb','int32','read'],['taucmc','float','read_write'],['ncb_flag','int32','read_write'],['err_flag','int32','read_write']]
params=[[n,'int32'] for n in ('ncol','nl','inflag','iceflag','liqflag')]+[['absliq0','float32']]
header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_cldpr_f32_bindings.metalh"
#pragma clang fp contract(off)
#define NGPTLW 140
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
kernel void hexv842_mpas_rrtmg_lw_cldpr_f32(constant Hexv842MpasRrtmgLwCldprF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwCldprF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl,inflag=p.inflag,iceflag=p.iceflag,liqflag=p.liqflag;float absliq0=p.absliq0;
'''
for n,t,access in resources:
    header+=f' device {"const " if access=="read" else ""}{"int" if t=="int32" else "float"}* {n}=r.{n};\n'
(root/f'woof/metal/shaders/{name}.metal').write_text(header+body+'\n}\n')
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/core/kernels/rrtmg_lw_chain.cu','cuda_symbol':'rlw_cldprmc_coalesced','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
