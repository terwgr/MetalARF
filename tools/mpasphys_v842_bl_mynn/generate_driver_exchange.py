"""Generate the native MYNN exchange-coefficient stage."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source
m=re.search(r'__device__ __noinline__ void f_retrieve_exchange_coeffs\s*\(.*?\)\s*\{',source,re.S);assert m
s=source[m.end():];a=s.index('// bl_mynn_subroutines.F90:4331');b=s.index('// MPAS physics_mmm/bl_mynn_subroutines.F90:4344',a);body=s[a:b].strip();assert body.endswith('}');body=body[:-1]
name='hexv842_mpas_bl_mynn_driver_exchange_f32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_level2_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_level2_f32_bindings.metalh',name+'_bindings.metalh')
inputs=('dfm','dfh','dz');outputs=('k_m','k_h')
macros=''.join(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]\n' for n in inputs+outputs)
undef=''.join(f'#undef {n}\n' for n in inputs+outputs)
text=header+f'''\nkernel void {name}(constant Hexv842MpasBlMynnDriverExchangeF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDriverExchangeF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]){{
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl,k;float dzk;
'''+macros+body+'\n'+undef+'}\n'
resources=[[n,'float','read'] for n in inputs]+[[n,'float','read_write'] for n in outputs]
params=[['nc','int32'],['nl','int32']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_retrieve_exchange_coeffs','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/shaders/{name}.metal').write_text(text);(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
