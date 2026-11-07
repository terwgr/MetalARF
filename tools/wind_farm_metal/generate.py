#!/usr/bin/env python3
"""Translate the upstream turbine kernels using the existing soft32 translator."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/sfire_metal'))
import generate as sf
from parse import split_top
from translate import untok


def main():
    source = 'woof/core/kernels/wind_farm.cu'
    text = sf.preprocessed((ROOT / source).read_text())
    text = re.sub(r'\bF\b', 'wf_force', text)
    items = split_top(text)
    out = ['// Translated from the CUDA turbine equations by tools/wind_farm_metal/generate.py.',
           '#include "wind_farm_support.metalh"']
    for item in items:
        if not item.startswith('extern "C"'):
            out.append(sf.convert_function(item))
            continue
        name, params = sf.kernel_info(item)
        wrapper, resources, scalars = sf.wrapper(name, params, [])
        out.insert(1, f'#include "{name}_bindings.metalh"')
        if name == 'wind_fitch_sparse':
            scalars.append(['n', 'int32'])
            wrapper = wrapper.replace('&p [[buffer(1)]]', '&prm [[buffer(1)]]')
            body = item[item.index('{') + 1:item.rindex('}')]
            body = body.replace('blockIdx.x', '(gid/128)').replace('threadIdx.x', '(gid%128)')
            body = body.replace('__shared__', 'threadgroup').replace('__syncthreads()', 'threadgroup_barrier(mem_flags::mem_threadgroup)')
            # A full 128-thread group is launched for every occupied column.
            aliases = []
            for p in params:
                n = p['name']
                if p['pointer']:
                    ty = 'F' if p['base'] == 'float' else p['base']
                    aliases.append(f'device {"const " if p["const"] else ""}{ty} *{n} = (device {"const " if p["const"] else ""}{ty} *)r.{n};')
                else:
                    ty = 'F' if p['base'] == 'float' else p['base']
                    aliases.append(f'{ty} {n} = {ty}(prm.{n});')
            converted = untok(sf.rewrite_pointer_locals(sf.map_types(sf.tokenize(body))))
            converted = converted.replace('threadgroup F sums [ 128 ] ;', 'threadgroup uint sums_bits[128]; threadgroup F *sums = (threadgroup F *)sums_bits;').replace('threadgroup F hs [ 7 ] ;', 'threadgroup uint hs_bits[7]; threadgroup F *hs = (threadgroup F *)hs_bits;')
            wrapper = wrapper[:wrapper.index('{')+1] + '\nif (gid >= uint(prm.n)) return;\n' + '\n'.join(aliases) + '\n' + converted + '\n}\n'
        else:
            out.append(sf.convert_kernel(item, name))
        out.append(wrapper)
        fragment = dict(name=name, cuda_symbol=name, module='wind_farm',
                        source='woof/metal/shaders/wind_farm.metal', cuda_source=source,
                        resources=resources, parameters=scalars,
                        shapes={r[0]: [['any_1d']] for r in resources},
                        threads_per_group=128, layout_version=1,
                        arithmetic_policies=['soft32-rn-nonftz', 'soft64-rn-bit-ops'])
        (ROOT / f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps({'kernels': [fragment]}, indent=2)+'\n')
    result = '\n\n'.join(out) + '\n'
    for old, new in {'hypotf':'wf_hypotf','asinf':'wf_asinf','acosf':'wf_acosf',
                     'sinf':'wf_sinf','cosf':'wf_cosf','expf':'wf_expf',
                     'atan2f':'wf_atan2f','remainderf':'wf_remainderf',
                     'sin':'fd64_sin','cos':'fd64_cos'}.items():
        result = re.sub(r'\b'+old+r'\b', new, result)
    for argument in ('theta', 'psi'):
        for func, cosine in (('wf_sinf', 'false'), ('wf_cosf', 'true')):
            result = result.replace(f'{func} ( {argument} )', f'wf_pair_trig ( {argument}, {cosine} )')
    # F compound operators in the shared support only bind thread locals.
    result = re.sub(r'(\b\w+\s*\[[^;{}]*?\])\s*([+*\-/])=\s*([^;]+);',
                    lambda m: f'{m[1]} = {m[1]} {m[2]} ({m[3]});', result)
    (ROOT / 'woof/metal/shaders/wind_farm.metal').write_text(result.rstrip()+'\n')


if __name__ == '__main__':
    main()
