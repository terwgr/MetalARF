from pathlib import Path
import re,json
root=Path(__file__).resolve().parents[2]
s=(root/'woof/core/kernels/rrtmg_lw_chain.cu').read_text()
part=s.split('extern "C" __global__ void rlw_inatm(',1)[1]
body=part[part.index('{')+1:part.index('\n}')]
body=body.replace('int col = blockIdx.x * blockDim.x + threadIdx.x;', 'int col=int(c);').replace('(long long)col', '(ulong)col')
body=body.replace('const float* pz = plev +', 'device const float* pz = r.plev +')
for name in ('wkl','wx','coldry','wbrodl','pwvcm_v'):
 body=re.sub(r'\b'+name+r'\[','r.'+name+'[',body)
for name in ('ncol','nl','grav','avogad'):
 body=re.sub(r'\b'+name+r'\b','p.'+name,body)
header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#define MXMOL 38
#define MAXXSEC 4
#include "hexv842_mpas_rrtmg_lw_inatm_f32_bindings.metalh"
'''
shader=header+'''kernel void hexv842_mpas_rrtmg_lw_inatm_f32(constant Hexv842MpasRrtmgLwInatmF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwInatmF32Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {\n'''+body+'\n}\n'
(root/'woof/metal/shaders/hexv842_mpas_rrtmg_lw_inatm_f32.metal').write_text(shader)
name='hexv842_mpas_rrtmg_lw_inatm_f32'
resources=[['plev','float','read'],['wkl','float','read_write'],['wx','float','read_write'],['coldry','float','read_write'],['wbrodl','float','read_write'],['pwvcm_v','float','read_write']]
params=[['ncol','int32'],['nl','int32'],['grav','float32'],['avogad','float32']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/core/kernels/rrtmg_lw_chain.cu','cuda_symbol':'rlw_inatm','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
