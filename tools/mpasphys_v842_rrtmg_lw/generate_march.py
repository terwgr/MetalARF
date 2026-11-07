from pathlib import Path
import json,re,sys
root=Path.cwd();sys.path.insert(0,str(root))
from woof.core.rrtmg_lw import _lw_coalesced_source
source=_lw_coalesced_source((root/'woof/core/kernels/rrtmg_lw_chain.cu').read_text())
m=re.search(r'extern "C" __global__ void rlw_rtrn_march_coalesced\s*\(',source);i=source.index('{',m.end());j=i+1;depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
body=source[i+1:j-1].replace('long long','long')
body=body.replace('long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;','long tid = long(gid);')
body=body.replace('float atrans[RLW_MAXLAY], atot[RLW_MAXLAY], bbugas[RLW_MAXLAY],\n          bbutot[RLW_MAXLAY];','''device float* atrans = scratch + tid*4*nl;
    device float* atot = atrans + nl;
    device float* bbugas = atot + nl;
    device float* bbutot = bbugas + nl;''')
body=body.replace('(unsigned char)iclddn','(int)iclddn')
body=body.replace('odepth < 0.0f','soft32_less(as_type<uint>(odepth),0u)')
if 'blockIdx' in body or 'RLW_MAXLAY]' in body:raise ValueError('untranslated source')
name='hexv842_mpas_rrtmg_lw_march_f32'
resources=[[n,'float','read'] for n in ('cldfmc','odcld','efclfrac')]+[['icldlyr','int32','read']]+[[n,'float','read'] for n in ('secdiff','semiss','planklay','planklev','plankbnd','fracs','taut','tau_tbl','exp_tbl','tfn_tbl')]+[['ngb','int32','read']]+[[n,'float','read_write'] for n in ('radld_p','radclrd_p','radlu_p','radclru_p')]+[['iclddn_p','int32','read_write']]+[[n,'float','read_write'] for n in ('radlu_sfc','radclru_sfc','scratch')]
params=[['ncol','int32'],['nl','int32'],['bpade','float32']]
header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_march_f32_bindings.metalh"
#pragma clang fp contract(off)
#define NGPTLW 140
#define NBNDLW 16
#define RLW_MAXLAY 128
#define RLW_REC6 0.166667f
#define RLW_TBLINT 10000.0f
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
kernel void hexv842_mpas_rrtmg_lw_march_f32(constant Hexv842MpasRrtmgLwMarchF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwMarchF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float bpade=p.bpade;
'''
for n,t,access in resources:
    header+=f' device {"const " if access=="read" else ""}{"int" if t=="int32" else "float"}* {n}=r.{n};\n'
(root/f'woof/metal/shaders/{name}.metal').write_text(header+body+'\n}\n')
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/core/kernels/rrtmg_lw_chain.cu','cuda_symbol':'rlw_rtrn_march_coalesced','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
