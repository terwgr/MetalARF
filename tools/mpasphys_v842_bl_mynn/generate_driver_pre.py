"""Generate the native MYNN driver preparation for non-cycling columns."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source
m=re.search(r'__device__ __noinline__ void f_bl_mynn_run\s*\(.*?\)\s*\{',source,re.S);assert m
s=source[m.end():]
def region(start,end):
    a=s.index('// bl_mynn.F90:'+str(start));b=s.index('// bl_mynn.F90:'+str(end),a)
    return s[a:b]
# Mixing/specific conversion, native interface heights and thermodynamics.
prep=region(634,664)+region(681,696)
# The cold, non-cycling branch initializes these persistent columns and qke.
reset=region(710,722)
qke=region(730,733)
name='hexv842_mpas_bl_mynn_driver_pre_f32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_level2_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_level2_f32_bindings.metalh',name+'_bindings.metalh')
inputs=('dz1','th1','sqv1','sqc1','sqi1','sqs1','ex1')
outputs=('qv1','qc1','qi1','qs1','sqw','thl','thetav','qke1','sh1','sm1','el_pbl1','tsq1','qsq1','cov1','cldfra_bl1','qc_bl1','qi_bl1')
macros=''.join(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]\n' for n in inputs+outputs)
macros+='#define zw(i) r.zw[ulong(col)*(nl+1)+((i)-1)]\n'
undef=''.join(f'#undef {n}\n' for n in inputs+outputs+('zw',))
text=header+f'''\nkernel void {name}(constant Hexv842MpasBlMynnDriverPreF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDriverPreF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl,k;float ust1=r.ust1[col],xlvcp=cfg.xlvcp,xlscp=cfg.xlscp;
'''+macros+prep+'\nif(cfg.cold){\n'+reset+qke+'\n}\n'+undef+'}\n'
resources=[[n,'float','read'] for n in inputs+('ust1',)]+[[n,'float','read_write'] for n in outputs+('zw',)]
params=[['nc','int32'],['nl','int32'],['xlvcp','float32'],['xlscp','float32'],['cold','int32']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_bl_mynn_run','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/shaders/{name}.metal').write_text(text);(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
