"""Translate native LW gas bands 6–8 and their shared interpolation helpers."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from tools.mpasphys_v842_rrtmg_lw.band_source import band_source
helper=(root/'woof/core/kernels/rrtmg_lw_taugb06_09.cu').read_text()
helper=helper[helper.index('#define KA3_0609'):helper.index('// ---------------------------------------------------------------------\n// Band 6:')]
helper=helper.replace('__device__','static inline').replace('const float* absa','device const float* absa')
helper=helper.replace('float p = -fsl;','float p = as_type<float>(as_type<uint>(fsl)^0x80000000u);')
remainder=(root/'woof/metal/shaders/climate_diagnostics_exact.metal').read_text()
remainder=remainder[remainder.index('static uint climate_fmod('):remainder.index('static F divf(')]
for band in (6,7,8):
    macros,body,tables=band_source(band)
    macros=macros.replace('long long','long')
    body=body.replace('TAUGB_PROLOGUE','''long tid=long(gid);if(tid>=long(ncol)*nl)return;
    int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];''')
    for index,table in enumerate(tables):
        pattern=r'const float\s*\*\s*'+table+r'\s*=\s*tabs\['+str(index)+r'\];'
        body,count=re.subn(pattern,f'device const float* {table} = r.{table};',body)
        if count!=1:raise ValueError((band,table,count))
    if 'tabs[' in body:raise ValueError('unmapped native band table')
    name=f'hexv842_mpas_rrtmg_lw_band{band}_f32';typ=f'Hexv842MpasRrtmgLwBand{band}F32'
    resources=[['laytrop_v','int32','read'],['fs','float','read'],['isv','int32','read'],['wx','float','read'],['chi_mls','float','read']]+[[n,'float','read'] for n in tables]+[['gas_tau','float','read_write'],['fracs','float','read_write']]
    params=[['ncol','int32'],['nl','int32'],['oneminus','float32']]
    header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "soft_gfk_pow.metalh"
#pragma clang fp contract(off)
#define NGPTLW 140
#define MAXXSEC 4
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#define rlw_pow(x,y) as_type<float>(soft_gfk_pow_bits(as_type<uint>(x),as_type<uint>(y)))
#define fmodf(x,y) as_type<float>(climate_fmod(as_type<uint>(x),as_type<uint>(y)))
static inline float rlw_pow4(float x){float x2=mpas_mul(x,x);return mpas_mul(x2,x2);}
'''+f'#include "{name}_bindings.metalh"\n'+remainder+'\n'+macros+'\n'+helper+'\n'
    header+=f'kernel void {name}(constant {typ}Resources &r [[buffer(0)]],constant {typ}Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {{\n'
    header+=''' int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;
'''
    (root/f'woof/metal/shaders/{name}.metal').write_text(header+body+'\n}\n')
    spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/core/kernels/rrtmg_lw_taugb06_09.cu','cuda_symbol':f'rlw_taugb{band}','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
    (root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
