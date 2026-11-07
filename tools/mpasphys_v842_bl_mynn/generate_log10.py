"""Translate the vendored CORE-MATH log10f tables to Metal binary64 words."""
from pathlib import Path
import json,re,struct
root=Path(__file__).resolve().parents[2]
src=(root/'woof/hex/data/mpasphys-maths/core-math/log10f.c').read_text()
def bits(x):
    x=x.strip();value=float.fromhex(x) if '0x' in x else float(x)
    return struct.unpack('<Q',struct.pack('<d',value))[0]
def table(name):
    match=re.search(r'static const double '+name+r'\[\] =\s*\{(.*?)\};',src,re.S);assert match
    return [bits(x) for x in match.group(1).replace('\n',' ').split(',') if x.strip()]
def const(x):return '0x%016xul'%bits(x)
tables={name:table(name) for name in ('tr','tl','b','c')}
name='hexv842_mpas_bl_mynn_log10_f32'
notice=src[:src.index('*/')+2]
text=notice+'''\n#ifndef HEXV842_MPAS_BL_MYNN_LOG10_METALH
#define HEXV842_MPAS_BL_MYNN_LOG10_METALH
#include "soft_binary64.metalh"
'''
for key,vals in tables.items():text+=f'constant ulong mynn_log10_{key}[{len(vals)}] = {{'+','.join(f'0x{x:016x}ul' for x in vals)+'};\n'
text+='''constant uint mynn_log10_powers[16]={0x501502f9u,0x41200000u,0x42c80000u,0u,0x447a0000u,0u,0x461c4000u,0x47c35000u,0u,0x49742400u,0u,0x4b189680u,0x4cbebc20u,0u,0x4e6e6b28u,0x3f800000u};
static inline uint mynn_log10_bits(uint ux){
uint ax=ux<<1;
if(ax==0u)return 0xff800000u;
if(ux==0x7f800000u)return ux;
if(ax>0xff000000u)return ux|0x00400000u;
if((ux&0x80000000u)!=0u)return 0xffc00000u;
if(ux==mynn_log10_powers[(ux>>24)&15u]){uint je=uint(int(ux)>>23)-126u;je=(je*0x4d104d4u)>>28;return soft64_to_f32_bits(soft64_from_u32(je));}
if(ux<0x00800000u){int n=int(clz(ux))-8;ux<<=n;ux-=uint(n<<23);}
int e=(int(ux)>>23)-127;ulong m=ulong(ux&0x7fffffu),j=(m+(1ul<<16))>>17;
ulong z=soft64_sub(soft64_mul((m<<29)|0x3ff0000000000000ul,mynn_log10_tr[j]),0x3ff0000000000000ul),z2=soft64_mul(z,z),ed=soft64_from_i32(e);
'''
text+=f'''ulong r=soft64_add(soft64_add(soft64_add(soft64_mul(ed,{const('0x1.34413509f79ffp-2')}),mynn_log10_tl[j]),soft64_mul(z,mynn_log10_b[0])),soft64_mul(z2,soft64_add(mynn_log10_b[1],soft64_mul(z,mynn_log10_b[2]))));
uint ub=soft64_to_f32_bits(r),lb=soft64_to_f32_bits(soft64_add(r,{const('0x1.af23fp-34')}));
if(ub!=lb){{
ulong f=soft64_mul(z,soft64_add(soft64_add(mynn_log10_c[0],soft64_mul(z,mynn_log10_c[1])),soft64_mul(z2,soft64_add(soft64_add(mynn_log10_c[2],soft64_mul(z,mynn_log10_c[3])),soft64_mul(z2,soft64_add(soft64_add(mynn_log10_c[4],soft64_mul(z,mynn_log10_c[5])),soft64_mul(z2,mynn_log10_c[6])))))));
f=soft64_add(f,soft64_mul({const('-0x1.0cee0ed4ca7e9p-54')},ed));f=soft64_add(f,soft64_sub(mynn_log10_tl[j],mynn_log10_tl[0]));
ulong el=soft64_mul(ed,{const('0x1.34413509f7ap-2')});r=soft64_add(el,f);ub=soft64_to_f32_bits(r);
if((r&0xffffffful)==0ul){{ulong dr=soft64_add(soft64_sub(el,r),f);r=soft64_add(r,soft64_mul(dr,0x4040000000000000ul));ub=soft64_to_f32_bits(r);}}
}}return ub;}}
#endif
'''
(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_log10.metalh').write_text(text)
shader=f'''#include <metal_stdlib>\nusing namespace metal;\n#include "{name}_bindings.metalh"\n#include "hexv842_mpas_bl_mynn_log10.metalh"\nkernel void {name}(constant Hexv842MpasBlMynnLog10F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnLog10F32Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]){{if(i<uint(p.count))r.output[i]=as_type<float>(mynn_log10_bits(as_type<uint>(r.input[i])));}}\n'''
(root/f'woof/metal/shaders/{name}.metal').write_text(shader)
r=[['input','float','read'],['output','float','read_write']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/data/mpasphys-maths/core-math/log10f.c','cuda_symbol':'cr_log10f','resources':r,'shapes':{x[0]:[['any_1d']] for x in r},'parameters':[['count','int32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
