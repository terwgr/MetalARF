"""Generate the default native MYNN closure 2.5 turbulence stage."""
from pathlib import Path
import json,re,struct,sys
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

def block_end(s,opening):
    assert s[opening]=='{';depth=1;j=opening+1
    while depth:
        if s[j]=='{':depth+=1
        elif s[j]=='}':depth-=1
        j+=1
    return j

def remove_block(s,needle,keep_else=False):
    i=s.index(needle);o=s.index('{',i);end=block_end(s,o)
    if keep_else:
        tail=re.match(r'\s*else\s*\{',s[end:]);assert tail
        o2=end+tail.end()-1;end2=block_end(s,o2)
        return s[:i]+s[o2+1:end2-1]+s[end2:]
    return s[:i]+s[end:]

level=clean(body('f_mym_level2'))
length=body('f_mym_length');switch=length.index('switch (bl_mynn_mixlength)');case=length.index('case 2: {',switch)+len('case 2: {');end=length.index('break; } }',case)
length=clean(length[:switch]+length[case:end])
for a,b in {'((zwk) <= (pblh_plus_ent))':'soft32_less_equal(as_type<uint>(zwk),as_type<uint>(pblh_plus_ent))','((dtv(k)) > (0.0f))':'soft32_greater(as_type<uint>(dtv(k)),0u)','((rmo) > (0.0f))':'soft32_greater(as_type<uint>(rmo),0u)'}.items():
    assert a in length;length=length.replace(a,b)
turb=clean(body('f_mym_turbulence'))
turb=remove_block(turb,'if (((closure) >= (3.0f)))',True)
turb=remove_block(turb,'if (bl_mynn_tkebudget)')
turb=remove_block(turb,'if (((spp_pbl) == (1)))')
while 'if (debug_code)' in turb:turb=remove_block(turb,'if (debug_code)')
turb=re.sub(r'f_mym_level2\(.*?\);',lambda _: '{\n'+level+'\n}',turb,count=1)
turb=re.sub(r'f_mym_length\(.*?\);',lambda _: '{\n'+length+'\n}',turb,count=1)
turb=turb.replace('double ','ulong ').replace('((double)(','((')
# Double constants are represented as exact IEEE-754 words.
for value in ('1.0e-20','1e-8'):
    bits=struct.unpack('<Q',struct.pack('<d',float(value)))[0]
    turb=turb.replace(value,f'0x{bits:016x}ul')
turb=turb.replace('mynn_max(((mynn_ddiv(gmel, q3sq))), ((0x3e45798ee2308c3aul)))','mynn_max64(mynn_ddiv(gmel, q3sq),0x3e45798ee2308c3aul)')
turb=turb.replace('mynn_max(((eden)), ((0x3bc79ca10c924223ul)))','mynn_max64(eden,0x3bc79ca10c924223ul)')
turb=turb.replace('((mynn_ddiv(q3sq, dlsq)) < (mynn_neg(gh(k))))','soft64_less(mynn_ddiv(q3sq,dlsq),mynn_real64(mynn_neg(gh(k))))')
turb=turb.replace('((q3sq) < (q2sq))','soft64_less(q3sq,q2sq)').replace('q3sq < q2sq','soft64_less(q3sq,q2sq)')
turb=turb.replace('((sh(k)) > (sh25max))','soft32_greater(as_type<uint>(sh(k)),as_type<uint>(sh25max))').replace('((sh(k)) < (sh25min))','soft32_less(as_type<uint>(sh(k)),as_type<uint>(sh25min))')
turb=turb.replace('((edmf_a1(k)) > (0.001f))','soft32_greater(as_type<uint>(edmf_a1(k)),as_type<uint>(0.001f))').replace('((cldavg) > (0.02f))','soft32_greater(as_type<uint>(cldavg),as_type<uint>(0.02f))')
assert 'Mynn' not in turb and 'f_mym_' not in turb and '(double)' not in turb
name='hexv842_mpas_bl_mynn_turbulence25_f32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_length2_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_length2_f32_bindings.metalh',name+'_bindings.metalh').replace('#include "hexv842_mpas_bl_mynn_length2_validate_f32_bindings.metalh"','')
header+='''\nstatic inline ulong mynn_result64(ulong a,ulong b,ulong r){if(!soft64_is_nan(r))return r;if(soft64_is_nan(a))return soft64_quiet_nan(a);if(soft64_is_nan(b))return soft64_quiet_nan(b);return 0xfff8000000000000ul;}
static inline ulong mynn_real64(float a){return soft64_from_f32_bits(as_type<uint>(a));}
static inline float mynn_real32(ulong a){return as_type<float>(soft64_to_f32_bits(a));}
static inline ulong mynn_dadd(ulong a,ulong b){return mynn_result64(a,b,soft64_add(a,b));}
static inline ulong mynn_dsub(ulong a,ulong b){return mynn_result64(a,b,soft64_sub(a,b));}
static inline ulong mynn_dmul(ulong a,ulong b){return mynn_result64(a,b,soft64_mul(a,b));}
static inline ulong mynn_ddiv(ulong a,ulong b){return mynn_result64(a,b,soft64_div(a,b));}
static inline ulong mynn_sqrt(ulong a){return mynn_result64(a,a,soft64_sqrt(a));}
static inline ulong mynn_neg(ulong a){return soft64_negate_bits(a);}
static inline ulong mynn_square(ulong a){return mynn_dmul(a,a);}
static inline ulong mynn_max64(ulong a,ulong b){return soft64_greater(a,b)?a:b;}
'''
inputs=('dz','u','v','thl','thetav','qw','ql','vt','vq','qke','tsq','qsq','cov','theta','cldfra_bl1d','edmf_w1','edmf_a1','tkeprodtd')
outputs=('sh','sm','el','dfm','dfh','dfq','tcd','qcd','pdk','pdt','pdq','pdc','qshear1d','qbuoy1d')
scratch=('qkw','dtl','dqw','dtv','gm','gh','qtke')
macros=''.join(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]\n' for n in inputs+outputs)
macros+=''.join(f'#define {n}(i) r.scratch[(ulong(col)*{len(scratch)}+{j})*nl+((i)-1)]\n' for j,n in enumerate(scratch))
macros+='#define u1(i) u(i)\n#define v1(i) v(i)\n#define zw(i) r.zw[ulong(col)*(nl+1)+((i)-1)]\n'
undef=''.join(f'#undef {n}\n' for n in inputs+outputs+scratch+('u1','v1','zw'))
constants=('b1','b2','twothirds','qkemin','pr','g1','c2','c3','c5','a1','c1','a2','g2','ckmod','cc2','cc3','e1c','e2c','e3c','e4c','e5c')
const=[]
for key in constants:
    m=re.search(r'static constexpr float '+key+r'\s*=.*?;',source);assert m
    const.append(m.group().replace('static constexpr','constexpr'))
text=header+f'''\nkernel void {name}(constant Hexv842MpasBlMynnTurbulence25F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnTurbulence25F32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;float grav=cfg.gravity,karman=cfg.karman,gtr=mynn_div(grav,300.f),tv0=mynn_mul(cfg.ep1,300.f);float zi=r.zi[col],rmo=r.rmo[col],fltv=r.fltv[col],psig_bl=r.psig_bl[col];constexpr float qmin=0.f,zmax=1.f,onethird=1.f/3.f;
'''+''.join(c+'\n' for c in const)+macros+turb+'\n'+undef+'}\n'
resources=[[n,'float','read'] for n in inputs+('zw','zi','rmo','fltv','psig_bl')]+[[n,'float','read_write'] for n in outputs+('scratch',)]
params=[['nc','int32'],['nl','int32'],['gravity','float32'],['karman','float32'],['ep1','float32']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_mym_turbulence','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/shaders/{name}.metal').write_text(text)
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
