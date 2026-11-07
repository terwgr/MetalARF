"""Generate native MYNN default cloud-PDF selector 2 condensation."""
from pathlib import Path
import json,re,sys
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
def remove_block(s,needle):
    i=s.index(needle);o=s.index('{',i);j=o+1;depth=1
    while depth:
        if s[j]=='{':depth+=1
        elif s[j]=='}':depth-=1
        j+=1
    return s[:i]+s[j:]
def extract_function(name):
    m=re.search(r'__device__ __noinline__ float '+name+r'\s*\(.*?\)\s*\{',source,re.S);assert m
    j=m.end();depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    return source[m.start():j].replace('__device__ __noinline__','static inline').replace('MynnFrame mynn_frame(arena);\n','')
qs=extract_function('f_qsat_blend').replace('MynnOptional<MynnText> waterice','int waterice').replace('MynnText wrt;\n','')
qs=remove_block(qs,'if ((!((waterice.present()))))')
# remove the unreachable optional-water branch, leaving native default blend
qs=remove_block(qs,'else {') if 'else {' in qs[:qs.index('xc =')] else qs
qs=qs.replace('(((((t) >= (t0c))) || (((wrt) == ("w")))))','(soft32_greater_equal(as_type<uint>(t),as_type<uint>(t0c)))')
xl=extract_function('f_xl_blend')
for fn in ('qs','xl'):
    text=locals()[fn].replace('((t) >= (t0c))','soft32_greater_equal(as_type<uint>(t),as_type<uint>(t0c))').replace('((t) <= (tice))','soft32_less_equal(as_type<uint>(t),as_type<uint>(tice))')
    locals()[fn]=text
s=body('f_mym_condensation');sw=s.index('switch (bl_mynn_cloudpdf)');case=s.index('case 2: case (-(2)): {',sw)+len('case 2: case (-(2)): {');end=s.index('break; } }',case)
s=s[:sw]+s[case:end]+s[end+len('break; } }'):]
s=remove_block(s,'if (((bl_mynn_cloudpdf) < (0)))')
s=re.sub(r'MynnFrame mynn_frame\(arena\);\n','',s);s=re.sub(r'\w+\.rebase\(\{kts\}\);\n','',s);s=re.sub(r'MynnArray<float, 1> \w+ = mynn_array<float, 1>\(arena, .*?;\n','',s)
s=s.replace('goto mynn_label_86;','break;').replace('mynn_label_86: ;','').replace('mynn_ipow<2>','mynn_square').replace('f_qsat_blend(t, t0c, tice, p(k), {})','f_qsat_blend(t, t0c, tice, p(k), 0)').replace('double ','ulong ')
replacements={
'((mpas_div(mpas_sub(theta2, theta1), mpas_sub(ht2, ht1))) < (mpas_div(10.f, 1500.f)))':'soft32_less(as_type<uint>(mpas_div(mpas_sub(theta2,theta1),mpas_sub(ht2,ht1))),as_type<uint>(mpas_div(10.f,1500.f)))',
'((ht1) < (19000.f))':'soft32_less(as_type<uint>(ht1),as_type<uint>(19000.f))','((ht1) > (4000.f))':'soft32_greater(as_type<uint>(ht1),as_type<uint>(4000.f))',
'((mpas_add(qi(k), qs(k))) > (1.e-9f))':'soft32_greater(as_type<uint>(mpas_add(qi(k),qs(k))),as_type<uint>(1.e-9f))','((zagl) > (pblh2))':'soft32_greater(as_type<uint>(zagl),as_type<uint>(pblh2))',
'((qc(k)) > (1.e-6f))':'soft32_greater(as_type<uint>(qc(k)),as_type<uint>(1.e-6f))','((q1k) < (0.f))':'soft32_less(as_type<uint>(q1k),0u)','((q1k) > (2.f))':'soft32_greater(as_type<uint>(q1k),as_type<uint>(2.f))',
'((cldfra_bl1d(k)) < (0.001f))':'soft32_less(as_type<uint>(cldfra_bl1d(k)),as_type<uint>(0.001f))','((mpas_sub(xland, 1.5f)) >= (0))':'soft32_greater_equal(as_type<uint>(mpas_sub(xland,1.5f)),0u)',
'((q1k) >= (1.0f))':'soft32_greater_equal(as_type<uint>(q1k),as_type<uint>(1.f))','((q1k) < (1.0f))':'soft32_less(as_type<uint>(q1k),as_type<uint>(1.f))',
'((q1k) >= (mynn_neg(1.7f)))':'soft32_greater_equal(as_type<uint>(q1k),as_type<uint>(mynn_neg(1.7f)))','((q1k) < (mynn_neg(1.7f)))':'soft32_less(as_type<uint>(q1k),as_type<uint>(mynn_neg(1.7f)))','((q1k) >= (mynn_neg(2.5f)))':'soft32_greater_equal(as_type<uint>(q1k),as_type<uint>(mynn_neg(2.5f)))'}
for a,b in replacements.items():assert a in s,a;s=s.replace(a,b)
assert 'Mynn' not in s and 'goto' not in s and 'switch' not in s
name='hexv842_mpas_bl_mynn_condensation2_f32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_length2_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_length2_f32_bindings.metalh',name+'_bindings.metalh').replace('#include "hexv842_mpas_bl_mynn_length2_validate_f32_bindings.metalh"', '#include "hexv842_mpas_bl_mynn_condensation_exp_f32_bindings.metalh"')
header+='''\n#include "hexv842_mpas_bl_mynn_log10.metalh"\nstatic inline int mynn_max(int a,int b){return a>b?a:b;}\nstatic inline ulong mynn_real64(float a){return soft64_from_f32_bits(as_type<uint>(a));}\nstatic inline float mynn_real32(ulong a){return as_type<float>(soft64_to_f32_bits(a));}\nstatic inline ulong mynn_sqrt(ulong a){ulong r=soft64_sqrt(a);return soft64_is_nan(r)&&!soft64_is_nan(a)?0xfff8000000000000ul:r;}\nstatic inline float mynn_exp(float a){
uint bits=as_type<uint>(a),abstop=(bits>>20u)&0x7ffu;
if(abstop>=0x42bu){if(bits==0xff800000u)return 0.f;if(abstop>=0x7f8u)return mynn_result(a,a,as_type<float>(soft32_add(bits,bits)));if((bits&0x80000000u)==0u&&bits>0x42b17218u)return as_type<float>(0x7f800000u);if((bits&0x80000000u)!=0u&&(bits&0x7fffffffu)>0x42cff1b4u)return 0.f;}
ulong xd=soft64_from_f32_bits(bits),inv=0x40471547652b82feul,shift=0x4338000000000000ul;
ulong kd=soft64_sub(soft64_fma(inv,xd,shift),shift),r=soft64_fma(inv,xd,soft64_negate_bits(kd));
ulong scale=soft_gfk_exp2_core(kd,shift,0ul,0ul,0ul,0ul),z=soft64_fma(0x3ebc6af84b912394ul,r,0x3f2ebfce50fac4f3ul),r2=soft64_mul(r,r);
ulong y=soft64_fma(0x3f962e42ff0c52d6ul,r,0x3ff0000000000000ul);y=soft64_mul(soft64_fma(z,r2,y),scale);
return as_type<float>(soft64_to_f32_bits(y));}\nstatic inline float mynn_log10(float a){return mynn_result(a,a,as_type<float>(mynn_log10_bits(as_type<uint>(a))));}\n'''
constants=['j'+str(i) for i in range(9)]+['k'+str(i) for i in range(9)]
for key in constants:
    m=re.search(r'static constexpr float '+key+r'\s*=.*?;',source);assert m
    header+=m.group().replace('static constexpr','constant')+'\n'
header+=qs+'\n'+xl+'\n'
inputs=('dz','zw','thl','qw','qv','qc','qi','qs','p','exner','tsq','qsq','cov','sh','el','th','rstoch_col')
outputs=('qc_bl1d','qi_bl1d','cldfra_bl1d','vt','vq','sgm')
scratch=('alp','a','bet','b','ql','q1','rh')
macros=''.join(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]\n' for n in inputs+outputs)
macros+=''.join(f'#define {n}(i) r.scratch[(ulong(col)*{len(scratch)}+{j})*nl+((i)-1)]\n' for j,n in enumerate(scratch))
undef=''.join(f'#undef {n}\n' for n in inputs+outputs+scratch)
scalars=('cp','cpv','cice','cliq','ep_2','r_d','r_v','xls','xlv','t0c','xlvcp','ep1')
text=header+f'''\nkernel void {name}(constant Hexv842MpasBlMynnCondensation2F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnCondensation2F32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl,spp_pbl=0;float xland=r.xland[col],pblh1=r.pblh1[col],hfx1=r.hfx1[col],rmo=r.rmo[col];constexpr float tice=240.f,tliq=269.f;
'''+''.join('float '+n+'=cfg.'+n+';\n' for n in scalars)+'''float tv0=mynn_mul(ep1,300.f);\n'''+macros+s+'\n'+undef+'}\n'
text+='''\nkernel void hexv842_mpas_bl_mynn_condensation_exp_f32(constant Hexv842MpasBlMynnCondensationExpF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnCondensationExpF32Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]){if(i<uint(p.count))r.output[i]=mynn_exp(r.input[i]);}\n'''
resources=[[n,'float','read'] for n in inputs+('xland','pblh1','hfx1','rmo')]+[[n,'float','read_write'] for n in outputs+('scratch',)]
params=[['nc','int32'],['nl','int32']]+[[n,'float32'] for n in scalars]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_mym_condensation','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
expname='hexv842_mpas_bl_mynn_condensation_exp_f32';expres=[['input','float','read'],['output','float','read_write']]
spec['kernels'].append({'name':expname,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/core/kernels/glibc_flt32.cuh','cuda_symbol':'gfk_exp','resources':expres,'shapes':{x[0]:[['any_1d']] for x in expres},'parameters':[['count','int32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']})
(root/f'woof/metal/shaders/{name}.metal').write_text(re.sub(r'(?m)^[ \t]+$', '', text))
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
