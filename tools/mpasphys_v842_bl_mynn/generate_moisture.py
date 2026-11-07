"""Generate native MYNN tendency moisture-conservation correction."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source
m=re.search(r'__device__ __noinline__ void f_moisture_check\s*\(.*?\)\s*\{',source,re.S);assert m
j=m.end();depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
s=source[m.end():j-1];s=re.sub(r'MynnFrame mynn_frame\(arena\);\n','',s);s=re.sub(r'\w+\.rebase\(\{1\}\);\n','',s)
for a,b in {'((dqv2) > (1.e-20f))':'soft32_greater(as_type<uint>(dqv2),as_type<uint>(1.e-20f))','((qv(k)) > (mpas_mul(2.0f, qvmin)))':'soft32_greater(as_type<uint>(qv(k)),as_type<uint>(mpas_mul(2.f,qvmin)))','((aa) < (0.5f))':'soft32_less(as_type<uint>(aa),as_type<uint>(0.5f))'}.items():
    assert a in s;s=s.replace(a,b)
assert 'Mynn' not in s
name='hexv842_mpas_bl_mynn_moisture_f32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_level2_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_level2_f32_bindings.metalh',name+'_bindings.metalh')
inputs=('dp','exner');outputs=('qv','qc','qi','qs','th','dqv','dqc','dqi','dqs','dth')
macros=''.join(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]\n' for n in inputs+outputs)
undef=''.join(f'#undef {n}\n' for n in inputs+outputs)
text=header+f'\nkernel void {name}(constant Hexv842MpasBlMynnMoistureF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnMoistureF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{\nif(col>=uint(cfg.nc))return;int nl=cfg.nl,kte=nl;float delt=cfg.delt,xlvcp=cfg.xlvcp,xlscp=cfg.xlscp;\n'+macros+s+'\n'+undef+'}\n'
resources=[[n,'float','read'] for n in inputs]+[[n,'float','read_write'] for n in outputs];params=[['nc','int32'],['nl','int32'],['delt','float32'],['xlvcp','float32'],['xlscp','float32']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_moisture_check','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/shaders/{name}.metal').write_text(text);(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
