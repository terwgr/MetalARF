"""Generate native MPAS RRTMG-LW initialization and band reductions."""
from pathlib import Path
import json,re,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from woof.hex.mpasphys.rrtmg_lw_v842 import _INIT_SOURCE
constants=[]
for name in ('ngc','ngn','ngm','wt'):
    m=re.search(r'extern "C" __device__ const (?:float|int) mpas_static_'+name+r'\[\d+\] = \{.*?\};',_INIT_SOURCE,re.S)
    if m is None:raise ValueError(name)
    constants.append(m.group().replace('extern "C" __device__ const','constant'))
header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#define rlw_exp(x) gfk_exp(x)
'''
items=[('rwgt','mpas_init_rwgt',[['rwgt','float','read_write']],[['n','int32']],1),
       ('tables','mpas_init_tables',[['tau','float','read_write'],['exp_table','float','read_write'],['tfn','float','read_write']],[['n','int32']],128),
       ('cmbgb','mpas_cmbgb',[['raw','float','read'],['rwgt','float','read'],['out','float','read_write']],[[x,'int32'] for x in ('band','points','first','weighted')],128)]
for short,symbol,resources,params,threads in items:
    name='hexv842_mpas_rrtmg_lw_'+short+'_f32';title=''.join(x.title() for x in name.split('_'))
    part=_INIT_SOURCE.split('extern "C" __global__ void '+symbol+'(',1)[1]
    body=part[part.index('{')+1:part.index('\n}')]
    body=body.replace('if(threadIdx.x||blockIdx.x)return;', 'if(c)return;')
    body=body.replace('int i=blockIdx.x*blockDim.x+threadIdx.x;', 'int i=int(c);')
    body=body.replace('int idx=blockIdx.x*blockDim.x+threadIdx.x;', 'int idx=int(c);').replace('fmaxf(', 'fmax(')
    for key,_,_ in resources:body=re.sub(r'\b'+key+r'\[', 'r.'+key+'[',body)
    for key,_ in params:body=re.sub(r'\b'+key+r'\b','p.'+key,body)
    out=header+f'#include "{name}_bindings.metalh"\n'+ '\n'.join(constants)+'\n'
    out+=f'kernel void {name}(constant {title}Resources &r [[buffer(0)]], constant {title}Params &p [[buffer(1)]], uint c [[thread_position_in_grid]]) {{\n'+body+'\n}\n'
    (ROOT/f'woof/metal/shaders/{name}.metal').write_text(out)
    spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/rrtmg_lw_v842.py','cuda_symbol':symbol,'resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':threads,'layout_version':1,'arithmetic_policies':['default']}]}
    (ROOT/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
