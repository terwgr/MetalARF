"""Generate the default native MMM MYNN mixing-length selector 2."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source
m=re.search(r'__device__ __noinline__ void f_mym_length\s*\(.*?\)\s*\{',source,re.S);assert m
start=m.end();j=start;depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
full=source[start:j-1];switch=full.index('switch (bl_mynn_mixlength)');case=full.index('case 2: {',switch)+len('case 2: {');end=full.index('break; } }',case)
body=full[:switch]+full[case:end]
body=re.sub(r'MynnFrame mynn_frame\(arena\);\n','',body)
body=re.sub(r'\w+\.rebase\(\{kts\}\);\n','',body)
body=re.sub(r'MynnArray<float, 1> \w+ = mynn_array<float, 1>\(arena, .*?;\n','',body)
body=body.replace('mynn_ipow<2>','mynn_square')
for old,new in {
 '((zwk) <= (pblh_plus_ent))':'soft32_less_equal(as_type<uint>(zwk),as_type<uint>(pblh_plus_ent))',
 '((dtv(k)) > (0.0f))':'soft32_greater(as_type<uint>(dtv(k)),0u)',
 '((rmo) > (0.0f))':'soft32_greater(as_type<uint>(rmo),0u)',
}.items():
    assert old in body,old
    body=body.replace(old,new)
assert 'Mynn' not in body and 'f_boulac' not in body
name='hexv842_mpas_bl_mynn_length2_f32';vname='hexv842_mpas_bl_mynn_length2_validate_f32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_stability_f32.metal').read_text().split('static inline float f_phim(')[0]
header=header.replace('hexv842_mpas_bl_mynn_stability_f32_bindings.metalh',name+'_bindings.metalh').replace('#include "hexv842_mpas_bl_mynn_stability_validate_f32_bindings.metalh"',f'#include "{vname}_bindings.metalh"')
header+='''\n#include "hexv842_mpas_bl_mynn_tanh.metalh"
static inline float mynn_tanh(float x){return mynn_result(x,x,as_type<float>(mynn_tanh_bits(as_type<uint>(x))));}
'''
fields=('dz','u1','v1','qke','dtv','cldfra_bl1d','edmf_w1','edmf_a1','qkw','el')
macros=''.join(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]\n' for n in fields)
macros+='#define zw(i) r.zw[ulong(col)*(nl+1)+((i)-1)]\n#define qtke(i) r.qtke[ulong(col)*nl+((i)-1)]\n'
undef=''.join(f'#undef {n}\n' for n in fields+('zw','qtke'))
params=[['nc','int32'],['nl','int32'],['gravity','float32'],['karman','float32']]
text=header+f'''\nkernel void {name}(constant Hexv842MpasBlMynnLength2F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnLength2F32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;float grav=cfg.gravity,karman=cfg.karman,gtr=mynn_div(grav,300.f);float zi=r.zi[col],rmo=r.rmo[col],fltv=r.fltv[col],psig_bl=r.psig_bl[col];constexpr float qmin=0.f,zmax=1.f,onethird=1.f/3.f;
'''+macros+body+'\n'+undef+'}\n'
text+=f'''\nkernel void {vname}(constant Hexv842MpasBlMynnLength2ValidateF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnLength2ValidateF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{
if(col>=uint(cfg.nc))return;int nl=cfg.nl;bool bad=false,stop=false;float zi=r.zi[col];
uint zbits=as_type<uint>(zi),rmobits=as_type<uint>(r.rmo[col]),fbits=as_type<uint>(r.fltv[col]),pbits=as_type<uint>(r.psig_bl[col]);
bad|=(zbits&0x7fffffffu)>=0x7f800000u || soft32_less(zbits,0u) || (rmobits&0x7fffffffu)>=0x7f800000u || (fbits&0x7fffffffu)>=0x7f800000u || (pbits&0x7fffffffu)>=0x7f800000u;
float zi2=mynn_max(zi,300.f),h1=mynn_min(mynn_max(mynn_mul(.3f,zi2),300.f),600.f),threshold=mynn_max(mynn_add(zi,h1),100.f);
for(int k=0;k<=nl;k++){{uint z=as_type<uint>(r.zw[ulong(col)*(nl+1)+k]);bad|=(z&0x7fffffffu)>=0x7f800000u;if(k>=1&&soft32_greater(z,as_type<uint>(threshold)))stop=true;
if(k<nl){{ulong j=ulong(col)*nl+k;uint d=as_type<uint>(r.dz[j]);bad|=d<0x00800000u || d>=0x7f800000u;
'''+''.join(f'bad|=(as_type<uint>(r.{n}[j])&0x7fffffffu)>=0x7f800000u;\n' for n in fields[1:8])+f'''}}}}
r.invalid[col]=int(bad||!stop);
}}\n'''
resources=[[n,'float','read'] for n in fields[:8]]+[['zw','float','read']]+[[n,'float','read'] for n in ('zi','rmo','fltv','psig_bl')]+[[n,'float','read_write'] for n in ('qkw','el','qtke')]
vr=[[n,'float','read'] for n in fields[:8]]+[['zw','float','read']]+[[n,'float','read'] for n in ('zi','rmo','fltv','psig_bl')]+[['invalid','int32','read_write']]
def spec(k,r):return {'name':k,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_mym_length','resources':r,'shapes':{x[0]:[['any_1d']] for x in r},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}
(root/f'woof/metal/shaders/{name}.metal').write_text(text)
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps({'kernels':[spec(name,resources),spec(vname,vr)]},indent=2)+'\n')
