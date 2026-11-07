"""Translate the selected WOOF upper-wind limiter to integer FP32 Metal."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.mp28_metal.parse import closure, def_name, split_top
from tools.thompson_real_column_parity.host_backend import parse_kernels


def retained_source():
    from woof.core.kernels import module_source
    source='#define GPUWM_WRF_EXACT 1\n'+module_source('upper_wind_limiter')
    expanded=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],
                            input=source,text=True,capture_output=True,check=True).stdout
    return closure(split_top(expanded),('upper_wind_limiter',))[0]


def main():
    items=retained_source()
    params=parse_kernels('\n'.join(items))['upper_wind_limiter']
    item=next(x for x in items if def_name(x)=='upper_wind_limiter')
    item=re.sub(r'extern\s+"C"\s+__global__\s*','',item)
    pos=item.index('{');head,body=item[:pos].rstrip(),item[pos:]
    head=head[:-1]+', int col)\n'
    body,count=re.subn(r'const int col = blockIdx\.x \* blockDim\.x \+ threadIdx\.x;', '',body)
    if count != 1: raise ValueError('unexpected CUDA thread index')
    item=head+body
    item=item.replace('float(k-(nz-1))','Exact32(float(k-(nz-1)))')
    item=re.sub(r'\bconst float\s*\*','device const Exact32*',item)
    item=re.sub(r'\bfloat\s*\*','device Exact32*',item)
    item=re.sub(r'\bfloat\b(?!\()', 'Exact32', item)
    for old,new in {'__fadd_rn':'exact_add','__fsub_rn':'exact_sub','__fmul_rn':'exact_mul',
                    '__fdiv_rn':'exact_div','fmaxf':'exact_max','sqrtf':'exact_sqrt',
                    'gfk_pow':'exact_gfk_pow'}.items():
        item=re.sub(r'\b'+old+r'\b',new,item)
    if re.search(r'\b(?:__f[a-z]+_rn|sqrtf|fmaxf|gfk_pow|blockIdx|threadIdx)\b',item):
        raise ValueError('untranslated CUDA operation')
    base=json.loads((ROOT/'woof/metal/kernel_signatures.json').read_text())
    sig=copy.deepcopy(next(x for x in base['kernels'] if x['name']=='upper_wind_limiter'))
    sig.update(name='wrf_exact_upper_wind_limiter',module='wrf_exact_upper_wind_limiter',
               source='woof/metal/shaders/wrf_exact_upper_wind_limiter.metal',arithmetic_policies=['wrf-exact'])
    args=[]
    for p in params:
        if p.pointer:
            const='const ' if 'const' in p.text.split() else ''
            args.append('reinterpret_cast<device '+const+'Exact32*>(r.'+p.name+')')
        else:
            args.append('Exact32(p.'+p.name+')' if p.base=='float' else 'p.'+p.name)
    shader=('// Generated from WOOF CUDA upper-wind source; do not edit.\n'
            '#include <metal_stdlib>\n#include "exact_binary32.metalh"\n#include "exact_gfk_math.metalh"\n'
            '#include "wrf_exact_upper_wind_limiter_bindings.metalh"\nusing namespace metal;\n'
            'namespace wrf_exact_upper_wind_source {\n'
            'static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_greater_equal(b.bits,a.bits); }\n'
            '#define G Exact32(9.81f)\n'+item+'\n}\n'
            'kernel void wrf_exact_upper_wind_limiter(constant WrfExactUpperWindLimiterResources &r [[buffer(0)]], '
            'constant WrfExactUpperWindLimiterParams &p [[buffer(1)]], uint col [[thread_position_in_grid]]) {\n'
            '  if (col >= uint(p.ny)*uint(p.nx)) return;\n'
            '  wrf_exact_upper_wind_source::upper_wind_limiter('+', '.join(args)+', int(col));\n}\n')
    (ROOT/'woof/metal/shaders/wrf_exact_upper_wind_limiter.metal').write_text('\n'.join(x.rstrip() for x in shader.splitlines())+'\n')
    (ROOT/'woof/metal/signature_fragments/wrf_exact_upper_wind_limiter.json').write_text(json.dumps({'kernels':[sig]},indent=2)+'\n')

if __name__=='__main__': main()
