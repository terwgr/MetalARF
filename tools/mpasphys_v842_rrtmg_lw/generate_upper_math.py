"""Translate upper-band source helpers and exact binary32 remainder."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.rrtmg_lw_v842 import _solver_source
s=_solver_source()
def function(name):
    i=s.index('__device__ ',s.index(name)-30);j=s.index('{',i);k=j+1;depth=1
    while depth:
        if s[k]=='{':depth+=1
        elif s[k]=='}':depth-=1
        k+=1
    return s[i:k].replace('__device__ ','static inline ').replace('const float* absa','device const float* absa')
start=s.index('#define KA3_0609');stop=s.index('// specparm branch triplet',start)
macro1=s[start:stop]
start=s.index('struct RlwSpecFac0609');stop=s.index('// tau_major sum',start)
helper1=s[start:stop].replace('__device__ ','static inline ')
start=s.index('#define RLW1316_A3');stop=s.index('// The specparm branch triplet',start)
macro2=s[start:stop]
climate=(root/'woof/metal/shaders/climate_diagnostics_exact.metal').read_text()
start=climate.index('static uint climate_fmod(');stop=climate.index('static F divf(',start)
remainder=climate[start:stop].replace('climate_fmod','lw_upper_fmod_bits')
header='''#ifndef WOOF_MPAS_LW_UPPER_MATH_METALH
#define WOOF_MPAS_LW_UPPER_MATH_METALH
#include "soft_gfk_pow.metalh"
'''+remainder+'''
static inline float lw_upper_fmod(float x, float y) { return as_type<float>(lw_upper_fmod_bits(as_type<uint>(x),as_type<uint>(y))); }
static inline float rlw_pow(float x, float y) { return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x),as_type<uint>(y))); }
'''+function('rlw_pow4')+'\n'+macro1+'\n'+helper1+'\n'+function('rlw_ninemajor_0609')+'\n'+macro2+'\n'+function('rlw_nine_major_b1316')+'\n#endif\n'
(root/'woof/metal/shaders/hexv842_mpas_rrtmg_lw_upper_math.metalh').write_text(header)
