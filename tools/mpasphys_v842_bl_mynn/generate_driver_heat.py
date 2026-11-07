"""Generate the native MYNN driver dissipation-heat stage."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source
m=re.search(r'__device__ __noinline__ void f_bl_mynn_run\s*\(.*?\)\s*\{',source,re.S);assert m
s=source[m.end():];a=s.index('// bl_mynn.F90:973');b=s.index('// bl_mynn.F90:981',a);body=s[a:b]
name='hexv842_mpas_bl_mynn_driver_heat_f32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_condensation2_f32.metal').read_text().split('constant float j0')[0]
header=header.replace('hexv842_mpas_bl_mynn_condensation2_f32_bindings.metalh',name+'_bindings.metalh').replace('#include "hexv842_mpas_bl_mynn_condensation_exp_f32_bindings.metalh"','')
inputs=('qke1','el_pbl1','p1');outputs=('diss_heat',)
macros=''.join(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]\n' for n in inputs+outputs)
undef=''.join(f'#undef {n}\n' for n in inputs+outputs)
text=header+f'''\nkernel void {name}(constant Hexv842MpasBlMynnDriverHeatF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDriverHeatF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]){{
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl,k;float cp=cfg.cp,b1=24.f;
'''+macros+body+'\n'+undef+'}\n'
resources=[[n,'float','read'] for n in inputs]+[[n,'float','read_write'] for n in outputs]
params=[['nc','int32'],['nl','int32'],['cp','float32']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_bl_mynn_run','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/shaders/{name}.metal').write_text(text);(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
