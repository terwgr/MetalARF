"""Generate the native MYNN level-2 stability operator."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source
m=re.search(r'__device__ __noinline__ void f_mym_level2\s*\(.*?\)\s*\{',source,re.S);assert m
start=m.end();j=start;depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
body=source[start:j-1]
body=re.sub(r'MynnFrame mynn_frame\(arena\);\n','',body)
body=re.sub(r'\w+\.rebase\(\{kts\}\);\n','',body)
body=body.replace('mynn_ipow<2>','mynn_square')
assert 'Mynn' not in body
names=('dz','u','v','thl','thetav','qw','ql','vt','vq','dtl','dqw','dtv','gm','gh','sm','sh')
constants=('b1','b2','pr','g1','c2','c3','c5','a1','c1','a2','g2','ckmod')
const=[]
for key in constants:
    match=re.search(r'static constexpr float '+key+r'\s*=.*?;',source);assert match
    const.append(match.group().replace('static constexpr','constexpr'))
name='hexv842_mpas_bl_mynn_level2_f32';typename='Hexv842MpasBlMynnLevel2F32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_adapter_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_adapter_f32_bindings.metalh',name+'_bindings.metalh')
header+='''
static inline float mynn_neg(float a){return as_type<float>(as_type<uint>(a)^0x80000000u);}
static inline float mynn_min(float a,float b){return soft32_less(as_type<uint>(a),as_type<uint>(b))?a:b;}
static inline float mynn_max(float a,float b){return soft32_greater(as_type<uint>(a),as_type<uint>(b))?a:b;}
static inline float mynn_sqrt(float a){return mynn_result(a,a,cu_sqrt(a));}
static inline float mynn_square(float a){return mynn_mul(a,a);}
'''
macros=''.join(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]\n' for n in names)
undef=''.join(f'#undef {n}\n' for n in names)
text=header+f'\nkernel void {name}(constant {typename}Resources &r [[buffer(0)]],constant {typename}Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{\nif(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;float gtr=mynn_div(cfg.gravity,300.f),tv0=mynn_mul(cfg.ep1,300.f);\n'+''.join(x+'\n' for x in const)+macros+body+'\n'+undef+'}\n'
resources=[[n,'float','read' if n in names[:9] else 'read_write'] for n in names]
params=[['nc','int32'],['nl','int32'],['gravity','float32'],['ep1','float32']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_mym_level2','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/shaders/{name}.metal').write_text(text)
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
