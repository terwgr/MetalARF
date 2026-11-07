"""Generate the fused MYNN libm graph used by the native sea-ice source."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
pow_source=(ROOT/'woof/metal/shaders/soft_gfk_pow.metalh').read_text()
body=pow_source[pow_source.index('static inline uint soft_gfk_pow_bits('):pow_source.rindex('\n#endif')]
body=body.replace('soft_gfk_pow_bits(', 'seaice_pow_bits(').replace('soft_gfk_powf_log2_bits(ix)', 'seaice_pow_log2(ix)')
start=body.index('    return soft64_to_f32_bits(soft_gfk_exp2_core(')
body=body[:start]+'''    return soft64_to_f32_bits(seaice_exp2(ylogx, ulong(sign_bias)));
}
'''
text='''// Native MPAS sea-ice mynn_libm.cuh: fused binary64 operation graph.
#ifndef HEXV842_SEAICE_MATH
#define HEXV842_SEAICE_MATH
#include "soft_gfk_pow.metalh"
static inline ulong seaice_pow_log2(uint ix) {
    uint tmp=ix-0x3f330000u, top=tmp&0xff800000u, iz=ix-top;
    int i=int((tmp>>19)&15u), k=as_type<int>(top)>>23;
    ulong r=soft64_fma(soft64_from_f32_bits(iz),soft_gfk_powf_invc[i],0xbff0000000000000ul);
    ulong y0=soft64_add(soft_gfk_powf_logc[i],soft64_from_i32(k));
    ulong r2=soft64_mul(r,r),r4=soft64_mul(r2,r2);
    ulong y=soft64_fma(soft_gfk_powf_a[0],r,soft_gfk_powf_a[1]);
    ulong p=soft64_fma(soft_gfk_powf_a[2],r,soft_gfk_powf_a[3]);
    ulong q=soft64_fma(soft_gfk_powf_a[4],r,y0);
    q=soft64_fma(p,r2,q);
    return soft64_fma(y,r4,q);
}
static inline ulong seaice_exp2(ulong x,ulong bias) {
    ulong shift=0x42e8000000000000ul,kd=soft64_add(x,shift),ki=kd;
    kd=soft64_sub(kd,shift);
    ulong r=soft64_sub(x,kd),s=SOFT_GFK_EXP2F_TAB[uint(ki&31ul)]+((ki+bias)<<47u);
    ulong z=soft64_fma(0x3fac6af84b912394ul,r,0x3fcebfce50fac4f3ul);
    ulong r2=soft64_mul(r,r);
    ulong y=soft64_fma(0x3fe62e42ff0c52d6ul,r,0x3ff0000000000000ul);
    return soft64_mul(soft64_fma(z,r2,y),s);
}
'''+body+'''
static inline float seaice_pow(float x,float y) {
    return as_type<float>(seaice_pow_bits(as_type<uint>(x),as_type<uint>(y)));
}
'''
# expf has the same selected source FMA graph as the v842 preparation helper.
prep=(ROOT/'woof/metal/shaders/hexv842_mpas_prep_interfaces_f32.metal').read_text()
exp=prep[prep.index('static inline float mpas_v842_exp_fma('):prep.index('\nkernel void hexv842_mpas_prep_interfaces_f32')]
exp=exp.replace('mpas_v842_exp_fma','seaice_exp').replace(
    'if (abstop>=0x42bu) return as_type<float>(soft_gfk_exp(bits));',
    '''if (abstop>=0x42bu) {
        if (bits==0xff800000u) return 0.0f;
        if (abstop>=0x7f8u) return mpas_add(value,value);
        if (value>as_type<float>(0x42b17218u)) return as_type<float>(0x7f800000u);
        if (value<as_type<float>(0xc2cff1b4u)) return 0.0f;
    }''')
text+=exp+'\n#endif\n'
(ROOT/'woof/metal/shaders/hexv842_seaice_math.metalh').write_text(text)
