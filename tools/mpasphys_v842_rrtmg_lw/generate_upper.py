"""Generate isolated native LW gas-band kernels from the selected source."""
from pathlib import Path
import json,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from tools.mpasphys_v842_rrtmg_lw.band_source import band_source

for band in map(int,sys.argv[1:]):
    if band not in range(9,17):raise ValueError('upper LW band must be 9 through 16')
    macros,body,tables=band_source(band);macros=macros.replace('long long','long')
    body=body.replace('TAUGB_PROLOGUE','''long tid=long(gid);if(tid>=long(ncol)*nl)return;
    int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];''')
    for i,table in enumerate(tables):
        old=f'const float* {table} = tabs[{i}];'
        assert old in body
        body=body.replace(old,f'device const float* {table} = r.{table};')
    assert 'tabs[' not in body
    body=body.replace('fmodf(', 'lw_upper_fmod(')
    name=f'hexv842_mpas_rrtmg_lw_band{band}_f32'
    resources=[['laytrop_v','int32','read'],['fs','float','read'],['isv','int32','read'],['wx','float','read'],['chi_mls','float','read']]+[[n,'float','read'] for n in tables]+[['gas_tau','float','read_write'],['fracs','float','read_write']]
    params=[['ncol','int32'],['nl','int32'],['oneminus','float32']]
    header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "'''+name+'''_bindings.metalh"
#pragma clang fp contract(off)
#define NGPTLW 140
#define MAXXSEC 4
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
'''+macros+'\n#include "hexv842_mpas_rrtmg_lw_upper_math.metalh"\n'+f'''
kernel void {name}(constant Hexv842MpasRrtmgLwBand{band}F32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwBand{band}F32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {{
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;
'''
    (root/f'woof/metal/shaders/{name}.metal').write_text(header+body.rstrip()+'\n}\n')
    source_file='rrtmg_lw_taugb06_09.cu' if band==9 else 'rrtmg_lw_taugb02_10_11_12.cu' if band<=12 else 'rrtmg_lw_taugb13_16.cu'
    spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':f'woof/core/kernels/{source_file}','cuda_symbol':f'rlw_taugb{band}','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
    (root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
