"""Translate the native Thompson float log10 helper to integer binary64."""
from pathlib import Path
import json,re,struct
root=Path(__file__).resolve().parents[2]
src=(root/'woof/hex/data/mpasphys-maths/ubm_log10f.cuh').read_text()
def bits(x):
    x=x.strip().removesuffix('f');v=float.fromhex(x) if '0x' in x else float(x)
    return struct.unpack('<Q',struct.pack('<d',v))[0]
def table(name):
    m=re.search(r'__device__ const double ubm_l10_'+name+r'\[\d+\] = \{(.*?)\};',src,re.S)
    if not m:raise ValueError(name)
    return [bits(x) for x in m.group(1).replace('\n',' ').split(',') if x.strip()]
def c(x):return '0x%016xul'%bits(x)
name='hexv842_mpas_thompson_column_log10_f32'
text=src[:src.index('__device__ const double')]+'''\n#ifndef HEXV842_MPAS_THOMPSON_COLUMN_LOG10_METALH
#define HEXV842_MPAS_THOMPSON_COLUMN_LOG10_METALH
#include "soft_binary64.metalh"
'''
for key in ('tr','tl','b','c'):
    vals=table(key);text+=f'constant ulong native_l10_{key}[{len(vals)}]={{'+','.join(f'0x{x:016x}ul' for x in vals)+'};\n'
text+='''constant uint native_l10_st[16]={0x3f800000u,0x41200000u,0x42c80000u,0x447a0000u,0x461c4000u,0x47c35000u,0x49742400u,0x4b189680u,0x4cbebc20u,0x4e6e6b28u,0x501502f9u,0,0,0,0,0};
static inline uint native_thompson_log10_bits(uint ux){
if(ux<0x00800000u||ux>=0x7f800000u){
 if(ux==0u||ux>=0x7f800000u){
  uint ax=ux<<1;if(ux==0x7f800000u)return ux;if(ax==0u)return 0xff800000u;
  if(ax>0xff000000u)return ux|0x00400000u;return 0x7fc00000u;
 }
 int n=int(clz(ux))-8;ux<<=n;ux-=uint(n<<23);
}
uint m=ux&0x7fffffu,j=(m+(1u<<16))>>17;int e=(int(ux)>>23)-127;
uint je=uint(e+1);je=(je*0x4d104d4u)>>28;
if(ux==native_l10_st[je])return soft64_to_f32_bits(soft64_from_u32(je));
ulong ed=soft64_from_i32(e),tz=(ulong(m)| (1023ul<<23))<<29;
ulong z=soft64_sub(soft64_mul(tz,native_l10_tr[j]),0x3ff0000000000000ul),z2=soft64_mul(z,z),l=native_l10_tl[j];
'''
text+=f'''ulong r=soft64_add(soft64_add(soft64_add(soft64_mul(ed,{c('0x1.34413509f79ffp-2')}),l),soft64_mul(z,native_l10_b[0])),soft64_mul(z2,soft64_add(native_l10_b[1],soft64_mul(z,native_l10_b[2]))));
uint ub=soft64_to_f32_bits(r),lb=soft64_to_f32_bits(soft64_add(r,{c('0x1.b008p-34')}));
if(ub!=lb && ((ub|lb)<<1)!=0u){{
ulong t456=soft64_add(soft64_add(native_l10_c[4],soft64_mul(z,native_l10_c[5])),soft64_mul(z2,native_l10_c[6]));
ulong t23=soft64_add(native_l10_c[2],soft64_mul(z,native_l10_c[3])),t01=soft64_add(native_l10_c[0],soft64_mul(z,native_l10_c[1]));
ulong f=soft64_mul(z,soft64_add(t01,soft64_mul(z2,soft64_add(t23,soft64_mul(z2,t456)))));
f=soft64_sub(f,soft64_mul({c('0x1.0cee0ed4ca7e9p-54')},ed));f=soft64_add(f,soft64_sub(l,native_l10_tl[0]));
ulong el=soft64_mul(ed,{c('0x1.34413509f7ap-2')});r=soft64_add(el,f);ub=soft64_to_f32_bits(r);
if((r&0xffffffful)==0ul){{ulong dr=soft64_add(soft64_sub(el,r),f);r=soft64_add(r,soft64_mul(dr,0x4040000000000000ul));ub=soft64_to_f32_bits(r);}}
}}return ub;}}
#endif
'''
(root/'woof/metal/shaders/hexv842_mpas_thompson_column_log10.metalh').write_text(text)
shader=f'''#include <metal_stdlib>\nusing namespace metal;\n#include "{name}_bindings.metalh"\n#include "hexv842_mpas_thompson_column_log10.metalh"\nkernel void {name}(constant Hexv842MpasThompsonColumnLog10F32Resources &r [[buffer(0)]],constant Hexv842MpasThompsonColumnLog10F32Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]){{if(i<uint(p.n))r.output[i]=as_type<float>(native_thompson_log10_bits(as_type<uint>(r.input[i])));}}\n'''
(root/f'woof/metal/shaders/{name}.metal').write_text(shader)
r=[['input','float','read'],['output','float','read_write']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/data/mpasphys-maths/ubm_log10f.cuh','cuda_symbol':'ubm_log10f','resources':r,'shapes':{x[0]:[['any_1d']] for x in r},'parameters':[['n','int32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
