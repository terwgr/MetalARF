"""Translate native LW second gas band with explicit Metal table bindings."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.rrtmg_lw_v842 import _solver_source
from woof.core.rrtmg_lw import GPU_BAND_TABS
source=_solver_source();m=re.search(r'extern "C" __global__ void rlw_taugb2\s*\(',source)
i=source.index('{',m.end());j=i+1;depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
macros=source[source.index('#define SL_PAVEL 0'):source.index('// UNIVERSAL band-kernel signature')].replace('long long','long')
body=source[i+1:j-1].replace('TAUGB_PROLOGUE','''long tid=long(gid);if(tid>=long(ncol)*nl)return;
    int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];''')
tables=GPU_BAND_TABS[2]
for index,table in enumerate(tables):
    body=body.replace(f'const float* {table} = tabs[{index}];',f'device const float* {table} = r.{table};')
if 'tabs[' in body:raise ValueError('unmapped native band table')
name='hexv842_mpas_rrtmg_lw_band2_f32'
resources=[['laytrop_v','int32','read'],['fs','float','read'],['isv','int32','read'],['wx','float','read'],['chi_mls','float','read']]+[[n,'float','read'] for n in tables]+[['gas_tau','float','read_write'],['fracs','float','read_write']]
params=[['ncol','int32'],['nl','int32'],['oneminus','float32']]
header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_band2_f32_bindings.metalh"
#pragma clang fp contract(off)
#define NGPTLW 140
#define MAXXSEC 4
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
'''+macros+'''
kernel void hexv842_mpas_rrtmg_lw_band2_f32(constant Hexv842MpasRrtmgLwBand2F32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwBand2F32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;
'''
(root/f'woof/metal/shaders/{name}.metal').write_text(header+body+'\n}\n')
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/core/kernels/rrtmg_lw_taugb02_10_11_12.cu','cuda_symbol':'rlw_taugb2','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
