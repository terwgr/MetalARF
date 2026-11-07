"""Generate the native MYNN five-pass initialization, selector 2 only."""
from pathlib import Path
import json, re, sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source

def body(name):
    m=re.search(r'__device__ __noinline__ void '+name+r'\s*\(.*?\)\s*\{',source,re.S);assert m
    j=m.end();depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    return source[m.end():j-1]

def clean(s):
    s=re.sub(r'MynnFrame mynn_frame\(arena\);\n','',s)
    s=re.sub(r'\w+\.rebase\(\{kts\}\);\n','',s)
    s=re.sub(r'MynnArray<float, 1> \w+ = mynn_array<float, 1>\(arena, .*?;\n','',s)
    return s.replace('mynn_ipow<2>','mynn_square')
level=clean(body('f_mym_level2'))
length=body('f_mym_length');switch=length.index('switch (bl_mynn_mixlength)');case=length.index('case 2: {',switch)+len('case 2: {');end=length.index('break; } }',case)
length=clean(length[:switch]+length[case:end])
for a,b in {'((zwk) <= (pblh_plus_ent))':'soft32_less_equal(as_type<uint>(zwk),as_type<uint>(pblh_plus_ent))','((dtv(k)) > (0.0f))':'soft32_greater(as_type<uint>(dtv(k)),0u)','((rmo) > (0.0f))':'soft32_greater(as_type<uint>(rmo),0u)'}.items():
    assert a in length;length=length.replace(a,b)
init=clean(body('f_mym_initialize'))
init=re.sub(r'f_mym_level2\(.*?\);',lambda _: '{\n'+level+'\n}',init,count=1)
init=re.sub(r'f_mym_length\(.*?\);',lambda _: '{\n'+length+'\n}',init,count=1)
init=init.replace('((qke(k)) <= (0.0f))','soft32_less_equal(as_type<uint>(qke(k)),0u)')
assert 'Mynn' not in init and 'f_mym_' not in init
name='hexv842_mpas_bl_mynn_initialize_f32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_length2_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_length2_f32_bindings.metalh',name+'_bindings.metalh')
header=header.replace('#include "hexv842_mpas_bl_mynn_length2_validate_f32_bindings.metalh"','')
inputs=('dz','u','v','thl','qw','theta','thetav','cldfra_bl1d','edmf_w1','edmf_a1')
outputs=('sh','sm','el','qke','tsq','qsq','cov')
scratch=('ql','pdk','pdt','pdq','pdc','dtl','dqw','dtv','gm','gh','qkw','vt','vq','qtke')
macros=''.join(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]\n' for n in inputs+outputs)
macros+=''.join(f'#define {n}(i) r.scratch[(ulong(col)*{len(scratch)}+{j})*nl+((i)-1)]\n' for j,n in enumerate(scratch))
macros+='#define u1(i) u(i)\n#define v1(i) v(i)\n#define zw(i) r.zw[ulong(col)*(nl+1)+((i)-1)]\n'
undef=''.join(f'#undef {n}\n' for n in inputs+outputs+scratch+('u1','v1','zw'))
constants=('b1','b2','twothirds','qkemin','pr','g1','c2','c3','c5','a1','c1','a2','g2','ckmod')
const=[]
for key in constants:
    m=re.search(r'static constexpr float '+key+r'\s*=.*?;',source);assert m
    const.append(m.group().replace('static constexpr','constexpr'))
text=header+f'''\nkernel void {name}(constant Hexv842MpasBlMynnInitializeF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnInitializeF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;float grav=cfg.gravity,karman=cfg.karman,gtr=mynn_div(grav,300.f),tv0=mynn_mul(cfg.ep1,300.f);float zi=r.zi[col],rmo=r.rmo[col],ust=r.ust[col],psig_bl=r.psig_bl[col];bool initialize_qke=cfg.initialize_qke;constexpr float qmin=0.f,zmax=1.f,onethird=1.f/3.f;
'''+''.join(c+'\n' for c in const)+macros+init+'\n'+undef+'}\n'
resources=[[n,'float','read'] for n in inputs+('zw','zi','rmo','ust','psig_bl')]+[[n,'float','read_write'] for n in outputs+('scratch',)]
params=[['nc','int32'],['nl','int32'],['gravity','float32'],['karman','float32'],['ep1','float32'],['initialize_qke','int32']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_mym_initialize','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/shaders/{name}.metal').write_text(text)
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
