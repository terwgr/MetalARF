"""Generate the pinned CORE-MATH binary32 tanh for native MYNN."""
from pathlib import Path
import re,struct,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.resources_build import _vendored_maths
source=_vendored_maths()['core-math/tanhf.c']
license=source[:source.index('#include <stdint.h>')]
def words(name):
    m=re.search(r'static const double '+name+r'\[\]\s*=\s*\{(.*?)\};',source,re.S);assert m
    values=[float.fromhex(x.strip()) for x in m.group(1).split(',')]
    return ','.join(f'0x{struct.unpack("<Q",struct.pack("<d",x))[0]:016x}ul' for x in values)
header=license+'''\n#ifndef HEXV842_MPAS_BL_MYNN_TANH_METALH
#define HEXV842_MPAS_BL_MYNN_TANH_METALH
#include "soft_binary32.metalh"
#include "soft_binary64.metalh"
constant ulong mynn_tanh_cn[8]={'''+words('cn')+'''};
constant ulong mynn_tanh_cd[8]={'''+words('cd')+'''};
static inline uint mynn_tanh_bits(uint x) {
    uint ax=x&0x7fffffffu,e=(x>>23)&0xffu;
    if(e==255u) {if(ax>0x7f800000u)return soft32_add(x,x);return (x&0x80000000u)|0x3f800000u;}
    // For |x|<2^-25 the source fmaf rounds to x: its nonzero
    // correction is strictly less than half an ulp, including subnormal x.
    if(e<102u)return x;
    if(e<115u) {
        float f=as_type<float>(x),f2=mpas_mul(f,f);
        return as_type<uint>(fma(f,mpas_mul(-0x1.555556p-2f,f2),f));
    }
    if(ax>0x41102cb3u)return soft32_sub((x&0x80000000u)|0x3f800000u,(x&0x80000000u)|0x33000000u);
    ulong z=soft64_from_f32_bits(x),z2=soft64_mul(z,z),z4=soft64_mul(z2,z2),z8=soft64_mul(z4,z4);
    ulong n0=soft64_add(mynn_tanh_cn[0],soft64_mul(z2,mynn_tanh_cn[1]));
    ulong n2=soft64_add(mynn_tanh_cn[2],soft64_mul(z2,mynn_tanh_cn[3]));
    ulong n4=soft64_add(mynn_tanh_cn[4],soft64_mul(z2,mynn_tanh_cn[5]));
    ulong n6=soft64_add(mynn_tanh_cn[6],soft64_mul(z2,mynn_tanh_cn[7]));
    n0=soft64_add(n0,soft64_mul(z4,n2));n4=soft64_add(n4,soft64_mul(z4,n6));n0=soft64_add(n0,soft64_mul(z8,n4));
    ulong d0=soft64_add(mynn_tanh_cd[0],soft64_mul(z2,mynn_tanh_cd[1]));
    ulong d2=soft64_add(mynn_tanh_cd[2],soft64_mul(z2,mynn_tanh_cd[3]));
    ulong d4=soft64_add(mynn_tanh_cd[4],soft64_mul(z2,mynn_tanh_cd[5]));
    ulong d6=soft64_add(mynn_tanh_cd[6],soft64_mul(z2,mynn_tanh_cd[7]));
    d0=soft64_add(d0,soft64_mul(z4,d2));d4=soft64_add(d4,soft64_mul(z4,d6));d0=soft64_add(d0,soft64_mul(z8,d4));
    return soft64_to_f32_bits(soft64_div(soft64_mul(z,n0),d0));
}
#endif
'''
(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_tanh.metalh').write_text(header)
