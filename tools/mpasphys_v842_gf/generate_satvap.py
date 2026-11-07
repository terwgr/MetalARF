"""Translate the native MPAS GF saturation-vapour helper."""
from pathlib import Path
import json
from woof.hex.mpasphys import gf_mpas_v842 as native
from tools.mp28_metal.generate import POLICIES
root=Path(__file__).resolve().parents[2]; name='hexv842_mpas_gf_satvap_f32'
source=native._SCHEME_SOURCE
start=source.index('__device__ GFReal satvap (GFWorkspace &workspace, GFReal * temp2)\n{')
end=native._body_end(source,source.index('{',start)) if hasattr(native,'_body_end') else None
if end is None:
    brace=source.index('{',start);depth=0
    for i in range(brace,len(source)):
        depth+= (source[i]=='{')-(source[i]=='}')
        if depth==0:end=i+1;break
body=source[start:end].replace('__device__ GFReal satvap (GFWorkspace &workspace, GFReal * temp2)','static inline GFReal satvap(GFReal temp2)').replace('  GFFrame frame(workspace);','').replace('*temp2','temp2').replace('  L_161:;','')
pre='''#include <metal_stdlib>
using namespace metal;
#include "soft_binary32.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_gfk_log.metalh"
struct GFReal { uint bits; GFReal()=default; GFReal(float f):bits(as_type<uint>(f)){} static GFReal word(uint x){GFReal a;a.bits=x;return a;} };
static inline GFReal gf_nan(uint r,uint a,uint b){if(!soft32_is_nan(r))return GFReal::word(r);if(soft32_is_nan(a))return GFReal::word(soft32_quiet_nan(a));if(soft32_is_nan(b))return GFReal::word(soft32_quiet_nan(b));return GFReal::word(0xffc00000u);}
static inline GFReal operator+(GFReal a,GFReal b){return gf_nan(soft32_add(a.bits,b.bits),a.bits,b.bits);}
static inline GFReal operator-(GFReal a,GFReal b){return gf_nan(soft32_sub(a.bits,b.bits),a.bits,b.bits);}
static inline GFReal operator*(GFReal a,GFReal b){return gf_nan(soft32_mul(a.bits,b.bits),a.bits,b.bits);}
static inline GFReal operator/(GFReal a,GFReal b){return gf_nan(soft32_div(a.bits,b.bits),a.bits,b.bits);}
static inline GFReal operator-(GFReal a){return GFReal::word(a.bits^0x80000000u);}
static inline bool operator<(GFReal a,GFReal b){return soft32_less(a.bits,b.bits);}
static inline GFReal gf_log(GFReal a){return gf_nan(soft_gfk_log(a.bits),a.bits,0u);}
static inline GFReal gf_pow(GFReal a,GFReal b){return gf_nan(soft_gfk_pow_bits(a.bits,b.bits),a.bits,0u);}
'''
kernel=f'''\n#include "{name}_bindings.metalh"
kernel void {name}(constant Hexv842MpasGfSatvapF32Resources &r [[buffer(0)]],constant Hexv842MpasGfSatvapF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]){{
 if(gid>=uint(p.n))return; r.output[gid]=as_type<float>(satvap(GFReal::word(as_type<uint>(r.temperature[gid]))).bits);
}}\n'''
(root/f'woof/metal/shaders/{name}.metal').write_text(pre+body+kernel)
resources=[['temperature','float','read'],['output','float','read_write']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/gf_mpas_v842.py','cuda_symbol':'satvap','resources':resources,'shapes':{r[0]:[['any_1d']] for r in resources},'parameters':[['n','int32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':POLICIES}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
