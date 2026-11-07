"""Generate the native MMM MYNN prediction routine with internal solves."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source

def body(name):
    m=re.search(r'__device__ __noinline__ void '+name+r'\s*\(.*?\)\s*\{',source,re.S);assert m
    start=m.end();j=start;depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    b=source[start:j-1]
    b=re.sub(r'MynnFrame mynn_frame\(arena\);\n','',b)
    b=re.sub(r'\w+\.rebase\(\{(?:kts|1)\}\);\n','',b)
    b=re.sub(r'MynnArray<float, 1> \w+ = mynn_array<float, 1>\(arena, .*?;\n','',b)
    return b
predict=body('f_mym_predict');solve=body('f_tridiag2')
assert 'Mynn' not in predict and 'Mynn' not in solve
call='f_tridiag2(kte, a, b, c, d, x);';assert predict.count(call)==4
predict=predict.replace(call,'{ int n=kte;\n'+solve+'\n}')
predict=predict.replace('for (int mynn_element=0; mynn_element < (tke_up).size(); ++mynn_element) { tke_up.linear(mynn_element) = mpas_mul(0.5f, qke.linear(mynn_element)); }', 'for (int mynn_element=1; mynn_element <= nl; ++mynn_element) { tke_up(mynn_element) = mpas_mul(0.5f, qke(mynn_element)); }')
predict=re.sub(r'\(\w+\)\.size\(\)', 'nl', predict)
predict=re.sub(r'(\w+)\.linear\(mynn_element\)', r'\1(mynn_element+1)', predict)
assert '.linear(' not in predict and '.size(' not in predict
predict=predict.replace('mynn_ipow<2>','mynn_square').replace('mynn_ipow<3>','mynn_cube')
inputs=('dz','el','dfq','rho','s_aw','s_awqke')
outputs=('pdk','pdt','pdq','pdc','qke','tsq','qsq','cov','qwt1d','qdiss1d')
scratch=('tke_up','dzinv','qkw','bp','rp','df3q','dtz','a','b','c','d','x','rhoinv','rhoz','kqdz','kmdz','cp','dp')
name='hexv842_mpas_bl_mynn_predict_f32';typename='Hexv842MpasBlMynnPredictF32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_level2_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_level2_f32_bindings.metalh',name+'_bindings.metalh')
header+='\nstatic inline float mynn_cube(float a){return mynn_mul(mynn_square(a),a);}\n'
macros=''.join(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]\n' for n in inputs+outputs)
macros+=''.join(f'#define {n}(i) r.scratch[(ulong(col)*{len(scratch)}+{idx})*(nl+1)+((i)-1)]\n' for idx,n in enumerate(scratch))
undef=''.join(f'#undef {n}\n' for n in inputs+outputs+scratch)
params=[['nc','int32'],['nl','int32']]+[[n,'float32'] for n in ('closure','delt','ust','flt','flq','pmz','phh','karman')]+[[n,'int32'] for n in ('bl_mynn_edmf_tke','bl_mynn_tkebudget')]
locals=''.join(f'float {n}=cfg.{n};' for n in ('closure','delt','ust','flt','flq','pmz','phh','karman'))+''.join(f'bool {n}=cfg.{n}!=0;' for n in ('bl_mynn_edmf_tke','bl_mynn_tkebudget'))
text=header+f'\nkernel void {name}(constant {typename}Resources &r [[buffer(0)]],constant {typename}Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{\nif(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;'+locals+'constexpr float b1=24.f,b2=15.f,sqfac=3.f;\n'+macros+predict+'\n'+undef+'}\n'
resources=[[n,'float','read'] for n in inputs]+[[n,'float','read_write'] for n in outputs]+[['scratch','float','read_write']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_mym_predict','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/shaders/{name}.metal').write_text(text)
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
