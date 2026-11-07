"""Generate native MYNN turbulence tendencies with their internal solves."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source

def extract(name):
    m=re.search(r'__device__ __noinline__ void '+name+r'\s*\(.*?\)\s*\{',source,re.S);assert m
    j=m.end();depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    return m.group(),source[m.end():j-1]
def clean(s):
    s=re.sub(r'MynnFrame mynn_frame\(arena\);\n','',s)
    s=re.sub(r'\w+\.rebase\(\{(?:kts|1)\}\);\n','',s)
    s=re.sub(r'MynnArray<float, 1> \w+ = mynn_array<float, 1>\(arena, .*?;\n','',s)
    return s
signature,raw=extract('f_mynn_tendencies');turb=clean(raw);solve=clean(extract('f_tridiag2')[1]);moist=clean(extract('f_moisture_check')[1])
profiles=re.findall(r'MynnArray<float, 1> (\w+)',signature)
assert len(profiles)==len(set(profiles))
scratch=re.findall(r'MynnArray<float, 1> (\w+) = mynn_array',raw)+['cp','dp']
assert len(scratch)==len(set(scratch))
# Each source solve has the same coefficients; only its output changes.
pat=r'f_tridiag2\(kte, a, b, c, d, (\w+)\);'
def insert_solve(m):
    output=m.group(1);return '{ int n=kte;\n'+re.sub(r'\bx\(',output+'(',solve)+'\n}'
turb,n=re.subn(pat,insert_solve,turb);assert n==14
mapping=dict(dp='delp',qv='sqv2',qc='sqc2',qi='sqi2',qs='sqs2',th='thl')
moist=re.sub(r'\b('+'|'.join(mapping)+r')\(',lambda m:mapping[m.group(1)]+'(',moist)
for a,b in {'((dqv2) > (1.e-20f))':'soft32_greater(as_type<uint>(dqv2),as_type<uint>(1.e-20f))','((qv(k)) > (mpas_mul(2.0f, qvmin)))':'soft32_greater(as_type<uint>(sqv2(k)),as_type<uint>(mpas_mul(2.f,qvmin)))','((aa) < (0.5f))':'soft32_less(as_type<uint>(aa),as_type<uint>(0.5f))'}.items():
    a=re.sub(r'\bqv\(', 'sqv2(',a);assert a in moist,a;moist=moist.replace(a,b)
assert moist.count('return;')==1;moist=moist.replace('return;','')
call='f_moisture_check(kte, delt, delp, exner, sqv2, sqc2, sqi2, sqs2, thl, dqv, dqc, dqi, dqs, dth);';assert call in turb;turb=turb.replace(call,'{\n'+moist+'\n}')
turb=turb.replace('f_qsat_blend(t, t0c, tice, p(k), {})','f_qsat_blend(t, t0c, tice, p(k), 0)')
# Native array-section assignments are one-based elementwise operations.
turb=re.sub(r'for \(int mynn_element=0; mynn_element < \((\w+)\.slice\(MynnRange\{.*?\}\)\)\.size\(\); \+\+mynn_element\) \{ \1\.slice\(MynnRange\{.*?\}\)\.linear\(mynn_element\) = 0.0f; \}',r'for(int i=1;i<=nl;i++) { \1(i)=0.0f; }',turb)
turb=re.sub(r'\(\w+\)\.size\(\)', 'nl',turb);turb=re.sub(r'(\w+)\.linear\(mynn_element\)',r'\1(mynn_element+1)',turb)
turb=turb.replace('mynn_ipow<2>','mynn_square')
# The source debug option is a compile-time false; remove its print-only block.
i=turb.index('if (debug_code)');o=turb.index('{',i);j=o+1;depth=1
while depth:
    if turb[j]=='{':depth+=1
    elif turb[j]=='}':depth-=1
    j+=1
turb=turb[:i]+turb[j:]
for a,b in {'((qvflux) < (0.0f))':'soft32_less(as_type<uint>(qvflux),0u)','((sqc(k)) > (0.0f))':'soft32_greater(as_type<uint>(sqc(k)),0u)','((sqi(k)) > (0.0f))':'soft32_greater(as_type<uint>(sqi(k)),0u)','((mpas_add(mpas_mul(dozone(k), delt), ozone(k))) < (0.f))':'soft32_less(as_type<uint>(mpas_add(mpas_mul(dozone(k),delt),ozone(k))),0u)'}.items():
    assert a in turb,a;turb=turb.replace(a,b)
assert 'Mynn' not in turb and 'f_tridiag2' not in turb and 'f_moisture_check' not in turb and '.linear(' not in turb and '.size(' not in turb
name='hexv842_mpas_bl_mynn_tendencies_f32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_condensation2_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_condensation2_f32_bindings.metalh',name+'_bindings.metalh').replace('#include "hexv842_mpas_bl_mynn_condensation_exp_f32_bindings.metalh"','')
for suffix in ('pack0','pack1','scatter'):
    header+=f'#include "hexv842_mpas_bl_mynn_tendencies_{suffix}_f32_bindings.metalh"\n'
macros=''.join(f'#define {n}(i) r.packed[(ulong({j})*cfg.nc+col)*nl+((i)-1)]\n' for j,n in enumerate(profiles))
macros+=''.join(f'#define {n}(i) r.scratch[(ulong(col)*{len(scratch)}+{j})*(nl+1)+((i)-1)]\n' for j,n in enumerate(scratch))
undef=''.join(f'#undef {n}\n' for n in profiles+scratch)
columns=('psfc','ust','flt','flq','flqv','flqc','wspd','uoce','voce')
constants=('delt','r_d','p608','xlvcp','xlscp','t0c')
flags=('flag_qc','flag_qi','flag_qnc','flag_qni','flag_qs','flag_qnwfa','flag_qnifa','flag_qnbca','flag_ozone','bl_mynn_cloudmix','bl_mynn_mixqt','bl_mynn_edmf_mom','bl_mynn_mixscalars')
text=header+f'\nkernel void {name}(constant Hexv842MpasBlMynnTendenciesF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnTendenciesF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{\nif(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;constexpr float tice=240.f;\n'+''.join('float '+n+'=r.'+n+'[col];\n' for n in columns)+''.join('float '+n+'=cfg.'+n+';\n' for n in constants)+''.join('bool '+n+'=cfg.'+n+'!=0;\n' for n in flags)+macros+turb+'\n'+undef+'}\n'
resources=[['packed','float','read_write']]+[[n,'float','read'] for n in columns]+[['scratch','float','read_write']];params=[['nc','int32'],['nl','int32']]+[[n,'float32'] for n in constants]+[[n,'int32'] for n in flags]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_mynn_tendencies','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
outputs=('thl','du','dv','dth','dqv','dqc','dqi','dqs','dqnc','dqni','dqnwfa','dqnifa','dqnbca','dozone')
for suffix,fields,scatter in (('pack0',profiles[:40],False),('pack1',profiles[40:],False),('scatter',outputs,True)):
    kernel=f'hexv842_mpas_bl_mynn_tendencies_{suffix}_f32'
    typename='Hexv842MpasBlMynnTendencies'+suffix.title()+'F32'
    rs=[['packed','float','read' if scatter else 'read_write']]+[[n,'float','read_write' if scatter else 'read'] for n in fields]
    p=[['nc','int32'],['nl','int32']]
    spec['kernels'].append({'name':kernel,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_mynn_tendencies','resources':rs,'shapes':{x[0]:[['any_1d']] for x in rs},'parameters':p,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']})
    text+=f'\nkernel void {kernel}(constant {typename}Resources &r [[buffer(0)]],constant {typename}Params &cfg [[buffer(1)]],uint i [[thread_position_in_grid]]){{uint n=uint(cfg.nc*cfg.nl);if(i>=n)return;\n'
    for n in fields:
        j=profiles.index(n);packed=f'r.packed[ulong({j})*n+i]'
        text+=(f'r.{n}[i]={packed};\n' if scatter else f'{packed}=r.{n}[i];\n')
    text+='}\n'
(root/f'woof/metal/shaders/{name}.metal').write_text(re.sub(r'(?m)^[ \t]+$','',text));(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
