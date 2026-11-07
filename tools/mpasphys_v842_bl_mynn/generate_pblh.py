"""Generate the native MMM MYNN boundary-layer height calculation."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source
m=re.search(r'__device__ __noinline__ void f_get_pblh\s*\(.*?\)\s*\{',source,re.S);assert m
start=m.end();j=start;depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
body=source[start:j-1]
body=re.sub(r'MynnFrame mynn_frame\(arena\);\n','',body)
body=re.sub(r'\w+\.rebase\(\{kts\}\);\n','',body)
replacements={
 '((zw1d(k)) <= (200.f))':'soft32_less_equal(as_type<uint>(zw1d(k)),as_type<uint>(200.f))',
 '((minthv) > (thetav1d(k)))':'soft32_greater(as_type<uint>(minthv),as_type<uint>(thetav1d(k)))',
 '((mpas_sub(landsea, 1.5f)) >= (0))':'soft32_greater_equal(as_type<uint>(mpas_sub(landsea,1.5f)),0u)',
 '((thetav1d(k)) >= (mpas_add(minthv, delt_thv)))':'soft32_greater_equal(as_type<uint>(thetav1d(k)),as_type<uint>(mpas_add(minthv,delt_thv)))',
 '((zi) != (0.0f))':'(!soft32_equal(as_type<uint>(zi),0u))',
 '((qtke) <= (tkeeps))':'soft32_less_equal(as_type<uint>(qtke),as_type<uint>(tkeeps))',
 '((pblh_tke) != (0.f))':'(!soft32_equal(as_type<uint>(pblh_tke),0u))',
 '((maxqke) <= (0.05f))':'soft32_less_equal(as_type<uint>(maxqke),as_type<uint>(0.05f))',
 '((zw1d(k)) >= (zi))':'soft32_greater_equal(as_type<uint>(zw1d(k)),as_type<uint>(zi))',
}
for old,new in replacements.items():
    assert old in body,old
    body=body.replace(old,new)
assert 'Mynn' not in body
name='hexv842_mpas_bl_mynn_pblh_f32';vname='hexv842_mpas_bl_mynn_pblh_validate_f32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_level2_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_level2_f32_bindings.metalh',name+'_bindings.metalh')
header+='''\n#include "hexv842_mpas_bl_mynn_pblh_validate_f32_bindings.metalh"
#include "hexv842_mpas_bl_mynn_tanh.metalh"
static inline float mynn_tanh(float x){return mynn_result(x,x,as_type<float>(mynn_tanh_bits(as_type<uint>(x))));}
'''
arrays=('thetav1d','qke1d','zw1d','dz1d')
macros=''.join(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]\n' for n in arrays)
undef=''.join(f'#undef {n}\n' for n in arrays)
text=header+f'''\nkernel void {name}(constant Hexv842MpasBlMynnPblhF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnPblhF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;float zi,landsea=r.landsea[col];int kzi;
'''+macros+body+'\n'+undef+'r.zi[col]=zi;r.kzi[col]=kzi;\n}\n'
text+=f'''\nkernel void {vname}(constant Hexv842MpasBlMynnPblhValidateF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnPblhValidateF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{
if(col>=uint(cfg.nc))return;int nl=cfg.nl;bool stop=false,bad=(as_type<uint>(r.landsea[col])&0x7fffffffu)>=0x7f800000u;
for(int k=0;k<nl;k++){{ulong j=ulong(col)*nl+k;uint z=as_type<uint>(r.zw1d[j]);
 bad|=((z&0x7fffffffu)>=0x7f800000u)||((as_type<uint>(r.thetav1d[j])&0x7fffffffu)>=0x7f800000u)||((as_type<uint>(r.qke1d[j])&0x7fffffffu)>=0x7f800000u)||((as_type<uint>(r.dz1d[j])&0x7fffffffu)>=0x7f800000u);
 if(k>=1 && soft32_greater(z,as_type<uint>(200.f)))stop=true;}}
r.invalid[col]=int(bad||!stop);
}}\n'''
params=[['nc','int32'],['nl','int32']]
resources=[[n,'float','read'] for n in arrays]+[['landsea','float','read'],['zi','float','read_write'],['kzi','int32','read_write']]
vr=[[n,'float','read'] for n in arrays]+[['landsea','float','read'],['invalid','int32','read_write']]
def spec(k,r):return {'name':k,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_get_pblh','resources':r,'shapes':{x[0]:[['any_1d']] for x in r},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}
(root/f'woof/metal/shaders/{name}.metal').write_text(text)
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps({'kernels':[spec(name,resources),spec(vname,vr)]},indent=2)+'\n')
