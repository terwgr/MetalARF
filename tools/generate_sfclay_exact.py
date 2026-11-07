"""Translate the selected revised MM5 source graph to software binary32."""
from pathlib import Path
import re
from woof.core.constants import CUDA_DEFINES

root = Path(__file__).resolve().parents[1]
src = (root / 'woof/core/kernels/sfclay.cu').read_text()
helpers = src[src.index('__device__ __forceinline__ real sf_psim_classic_full'):src.index('extern "C" __global__')]
body = src[src.index('    const real karman = 0.4f'):src.rindex('\n}')]
inputs = ('u','v','t','qv','p','dz8w','psfc','tsk','pblh','mavail','xland','lakemask')
outputs = ('znt','ust','ustm','mol','hfx','qfx','qsfc','zol','regime','psim','psih','fm','fh','lh','u10','v10','th2','t2','q2','chs','chs2','cqs2','flhc','flqc','qgh','rmol','wspd','br','gz1oz0','cpm','ck','cka','cd','cda')
source_names = dict(zip(('znt','ust','ustm','mol','hfx','qfx','qsfc','zol_o','regime_o','psim_o','psih_o','fm_o','fh_o','lh_o','u10_o','v10_o','th2_o','t2_o','q2_o','chs_o','chs2_o','cqs2_o','flhc_o','flqc_o','qgh_o','rmol_o','wspd_o','br_o','gz1_o','cpm_o','ck_o','cka_o','cd_o','cda_o'),outputs))
for name, field in source_names.items():
    body=re.sub(r'\b'+name+r'\[idx\]\s*=\s*([^;]+);',lambda m: 'r.'+field+'[idx] = float('+m.group(1)+');',body)
for name, field in source_names.items():
    body=re.sub(r'(?<![.\w])'+name+r'\[idx\]', 'F(r.'+field+'[idx])',body)
for name in inputs:
    body=re.sub(r'(?<![.\w])'+name+r'\[idx\]', 'F(r.'+name+'[idx])',body)
def convert(s):
    s=s.replace('__device__ __forceinline__','static inline')
    s=re.sub(r'\breal\b','F',s).replace('F &zeta','thread F &zeta')
    s=s.replace('(F)(n + 1)', 'F(float(n + 1))').replace('(F)n', 'F(float(n))')
    for old,new in [('fminf(', 'sf_min('),('fmaxf(', 'sf_max('),('sqrtf(', 'sf_sqrt('),('fabsf(', 'fabs('),('copysignf(', 'sf_copysign(')]:
        s=s.replace(old,new)
    return s
helpers=convert(helpers).replace('static inline F sf_', 'static __attribute__((noinline)) F sf_')
body=convert(body)
constants='\n'.join('#define '+name+' F('+repr(float(CUDA_DEFINES[name]))+'f)' for name in ('G','RD','RV','CP','P0','RCP','XLV','SVP1','SVP2','SVP3','SVPT0','EP2'))
prefix='''#include <metal_stdlib>
#include "sfclay_column_exact_bindings.metalh"
#include "sfclay_preflight_exact_bindings.metalh"
#include "soft_word32.metalh"
#include "soft_gfk_atan.metalh"
using namespace metal;
static inline F operator/(F a,F b) { return aa_div_impl(a,b); }
static inline thread F &operator*=(thread F &a,F b) { a=a*b; return a; }
static inline F sf_min(F a,F b) { if (soft32_is_nan(a.bits)) return b; if (soft32_is_nan(b.bits)) return a; if (((a.bits|b.bits)&0x7fffffffu)==0) return f32_word(a.bits|b.bits); return a<b?a:b; }
static inline F sf_max(F a,F b) { if (soft32_is_nan(a.bits)) return b; if (soft32_is_nan(b.bits)) return a; if (((a.bits|b.bits)&0x7fffffffu)==0) return f32_word(a.bits&b.bits); return a>b?a:b; }
static inline F sf_sqrt(F a) { return f32_word(soft32_sqrt(a.bits)); }
static __attribute__((noinline)) F sf_log(F a) { return gfk_log(a); }
static __attribute__((noinline)) F sf_exp(F a) { return gfk_exp(a); }
static __attribute__((noinline)) F sf_pow(F a,F b) { return gfk_pow(a,b); }
static __attribute__((noinline)) F sf_atan(F a) { return f32_word(soft_gfk_atanf(a.bits)); }
static inline F sf_copysign(F a,F b) { return f32_word((a.bits&0x7fffffffu)|(b.bits&0x80000000u)); }
static inline bool surface_le_zero(F a) { return a <= F(0.0f); }
static inline bool surface_lt_zero(F a) { return a < F(0.0f); }
static inline F sf_log_zratio(F num,F z0) { return sf_log(num / z0); }
#define SF_TWO_ATAN1 F(1.57079637050628662109375f)
#define SF_SQRT3 F(1.73205077648162841796875f)
#define SF_FOUR_ATAN1_OVER_SQRT3 F(1.81379950046539306640625f)
#define SF_INV_1P1 F(9.0909087657928466796875e-1f)
#define SF_INV_2P5 F(4.000000059604644775390625e-1f)
#define SF_SQRT_0P71 F(8.42614948749542236328125e-1f)
#define SF_SQRT_0P60 F(7.74596691131591796875e-1f)
'''
preflight='''
kernel void sfclay_preflight_exact(constant SfclayPreflightExactResources &r [[buffer(0)]],constant SfclayPreflightExactParams &p [[buffer(1)]],uint idx [[thread_position_in_grid]]) {
    if (idx>=uint(p.n)) return;
    F z0=F(r.znt[idx]), za=F(0.5f)*F(r.dz8w[idx]);
    F nums[5]={za+z0,F(2.f)+z0,F(10.f)+z0,za,F(2.f)};
    bool reject=soft32_is_nan(z0.bits);
    for (uint i=0;i<5;i++) {
        uint b=sf_log(nums[i]/z0).bits;
        reject = reject || soft32_is_nan(nums[i].bits) || ((b & 0x7f800000u)==0x7f800000u);
    }
    r.status[idx]=reject?1.0f:0.0f;
}
'''
shader=prefix+constants+'\n'+helpers+preflight+'''\nkernel void sfclay_column_exact(constant SfclayColumnExactResources &r [[buffer(0)]],constant SfclayColumnExactParams &p [[buffer(1)]],uint idx [[thread_position_in_grid]]) {
    if (idx>=uint(p.n)) return;
    const int option=1, isfflx=p.isfflx, isftcflx=p.isftcflx, iz0tlnd=p.iz0tlnd;
    const F dx=F(p.dx);
'''+body+'\n}\n'
(root/'woof/metal/shaders/sfclay_exact.metal').write_text(shader)
