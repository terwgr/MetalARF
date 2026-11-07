import os
from pathlib import Path
p=Path(__file__).resolve().parent
root=Path(os.environ.get('WOOF_SOURCE_ROOT', Path.cwd())) / 'woof/metal/shaders'
s=(root/'soft_gfk_pow.metalh').read_text().replace('WOOF_SOFT_GFK_POW_METALH','RRFS_MYNN_POW_FMA_METALH')
s=s.replace('soft_gfk_powf_log2_bits','rrfs_mynn_powf_log2_bits').replace('soft_gfk_pow_bits','rrfs_mynn_pow_core')
s=s.replace('soft64_sub(soft64_mul(z, soft_gfk_powf_invc[i]), 0x3ff0000000000000ul)','soft64_fma(z, soft_gfk_powf_invc[i], 0xbff0000000000000ul)')
s=s.replace('soft64_add(soft64_mul(soft_gfk_powf_a[0], r), soft_gfk_powf_a[1])','soft64_fma(soft_gfk_powf_a[0], r, soft_gfk_powf_a[1])')
s=s.replace('soft64_add(soft64_mul(soft_gfk_powf_a[2], r), soft_gfk_powf_a[3])','soft64_fma(soft_gfk_powf_a[2], r, soft_gfk_powf_a[3])')
s=s.replace('soft64_add(soft64_mul(soft_gfk_powf_a[4], r), y0)','soft64_fma(soft_gfk_powf_a[4], r, y0)')
s=s.replace('soft64_add(soft64_mul(p, r2), q)','soft64_fma(p, r2, q)')
s=s.replace('soft64_add(soft64_mul(y, r4), q)','soft64_fma(y, r4, q)')
s=s.replace('soft_gfk_exp2_core(', 'rrfs_mynn_exp2_core(')
pos=s.index('// Preserve the pinned')
fma='''#include "soft_gfk_exp.metalh"
static inline ulong rrfs_mynn_exp2_core(ulong xd,ulong shift,ulong p0,ulong p1,ulong p2,ulong bias) {
    ulong kd=soft64_add(xd,shift),ki=kd;
    kd=soft64_sub(kd,shift);
    ulong r=soft64_sub(xd,kd);
    ulong scale=SOFT_GFK_EXP2F_TAB[uint(ki&31ul)]+((ki+bias)<<47u);
    ulong z=soft64_fma(p0,r,p1),r2=soft64_mul(r,r);
    ulong y=soft64_fma(p2,r,0x3ff0000000000000ul);
    return soft64_mul(soft64_fma(z,r2,y),scale);
}
'''
s=s[:pos]+fma+s[pos:]
s=s.replace('// Preserve the pinned source\'s separate RN-even multiply and add/subtract;\n// using soft64_fma here changes the binary64 rounding graph.','// FMA operations match the retained MYNN CUDA math, not the generic GFK header.')
s=s.replace('// Faithful operation graph of pinned glibc 2.39 powf log2_inline.','// RRFS MYNN CUDA uses FMA in this reduction.')
assert s.count('soft64_fma(')>=9
(p/'mynn_pow_fma.metalh').write_text(s)


import os
from pathlib import Path
import re,struct
p=Path(__file__).resolve().parent
s=(Path(os.environ.get('WOOF_SOURCE_ROOT', Path.cwd())) / 'woof/hex/glibc_erff.cuh').read_text()
table=s[s.index('hex_erff_coefficients[56][8]'):s.index('};',s.index('hex_erff_coefficients[56][8]'))]
vals=re.findall(r'-?0x[\da-f]+\.[\da-f]+p[+-]\d+',table,re.I);assert len(vals)==448
bits=lambda x:'0x%016xul'%struct.unpack('>Q',struct.pack('>d',float.fromhex(x)))[0]
small=['0x1.20dd750429b6dp+0','-0x1.812746b0375fbp-2','0x1.ce2f219fd6f45p-4','-0x1.b82ce2cbf0838p-6','0x1.565bb655adb85p-8','-0x1.c025bfc879c94p-11','0x1.f81718f61309cp-14','-0x1.cc67bd88f5867p-17']
text=s[:s.index('*/')+2]+'\nconstant ulong rrfs_erf_coefficients[448]={\n'+',\n'.join(','.join(map(bits,vals[i:i+8])) for i in range(0,448,8))+'\n};\n'
text+='constant ulong rrfs_erf_small[8]={'+','.join(map(bits,small))+'};\n'
text+='''static inline Exact32 mynn_erf(Exact32 a) {
    uint x=a.bits,ax=x&0x7fffffffu,sign=x&0x80000000u;
    if(ax>0x407ad444u) {
        if(ax>0x7f800000u)return Exact32::from_bits(x|0x00400000u);
        uint one=sign|0x3f800000u;
        if(ax==0x7f800000u)return Exact32::from_bits(one);
        return Exact32::from_bits(soft32_sub(one,sign|0x33000000u));
    }
    ulong s=soft64_from_f32_bits(x),z=soft64_from_f32_bits(ax);
    int i=soft64_to_i32_rz(soft64_mul(0x4030000000000000ul,z)).value;
    ulong c0,c2,c4,c6,z2,z4;
    if(ax<0x3ee00000u) {
        z2=soft64_mul(s,s);z4=soft64_mul(z2,z2);ulong z8=soft64_mul(z4,z4);
        c0=soft64_add(rrfs_erf_small[0],soft64_mul(z2,rrfs_erf_small[1]));
        c2=soft64_add(rrfs_erf_small[2],soft64_mul(z2,rrfs_erf_small[3]));
        c4=soft64_add(rrfs_erf_small[4],soft64_mul(z2,rrfs_erf_small[5]));
        c6=soft64_add(rrfs_erf_small[6],soft64_mul(z2,rrfs_erf_small[7]));
        c0=soft64_add(c0,soft64_mul(z4,c2));c4=soft64_add(c4,soft64_mul(z4,c6));
        c0=soft64_add(c0,soft64_mul(z8,c4));
        return Exact32::from_bits(soft64_to_f32_bits(soft64_mul(s,c0)));
    }
    z=soft64_sub(soft64_sub(z,0x3fa0000000000000ul),soft64_mul(0x3fb0000000000000ul,soft64_from_i32(i)));
    int j=(i-7)*8;
    z2=soft64_mul(z,z);z4=soft64_mul(z2,z2);
    c0=soft64_add(rrfs_erf_coefficients[j],soft64_mul(z,rrfs_erf_coefficients[j+1]));
    c2=soft64_add(rrfs_erf_coefficients[j+2],soft64_mul(z,rrfs_erf_coefficients[j+3]));
    c4=soft64_add(rrfs_erf_coefficients[j+4],soft64_mul(z,rrfs_erf_coefficients[j+5]));
    c6=soft64_add(rrfs_erf_coefficients[j+6],soft64_mul(z,rrfs_erf_coefficients[j+7]));
    c0=soft64_add(c0,soft64_mul(z2,c2));c4=soft64_add(c4,soft64_mul(z2,c6));
    c0=soft64_add(c0,soft64_mul(z4,c4));
    return Exact32::from_bits(soft64_to_f32_bits((c0&0x7ffffffffffffffful)|(s&0x8000000000000000ul)));
}
'''
(p/'mynn_erf.metalh').write_text(text)
