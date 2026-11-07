"""Translate the selected CUDA w-damping kernel to integer binary32 Metal."""
import copy
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.mp28_metal.parse import closure, def_name, split_top
from tools.thompson_real_column_parity.host_backend import parse_kernels


def retained_source():
    from woof.core.kernels import module_source
    source = '#define GPUWM_WRF_EXACT 1\n' + module_source('openbc')
    expanded = subprocess.run(['clang++', '-E', '-P', '-std=c++17', '-x', 'c++', '-'],
                              input=source, text=True, capture_output=True, check=True).stdout
    return closure(split_top(expanded), ('w_damp',))[0]


def main():
    items = retained_source()
    params = parse_kernels('\n'.join(items))['w_damp']
    item = next(x for x in items if def_name(x) == 'w_damp')
    item = re.sub(r'extern\s+"C"\s+__global__\s*', '', item)
    pos = item.index('{')
    head, body = item[:pos].rstrip(), item[pos:]
    head = head[:-1] + ', int t)\n'
    body, count = re.subn(r'int t = blockIdx\.x \* blockDim\.x \+ threadIdx\.x;', '', body)
    if count != 1: raise ValueError('unexpected CUDA thread index')
    item = head + body
    item = re.sub(r'\b__restrict__\b', '', item)
    item = re.sub(r'\bconst real\s*\*', 'device const Exact32*', item)
    item = re.sub(r'\breal\s*\*', 'device Exact32*', item)
    item = re.sub(r'\breal\b', 'Exact32', item)
    item = re.sub(r'\bsize_t\b', 'ulong', item)
    for before, after in {'__fadd_rn': 'exact_add', '__fsub_rn': 'exact_sub',
                          '__fmul_rn': 'exact_mul', '__fdiv_rn': 'exact_div',
                          'fabsf': 'exact_abs', 'copysignf': 'exact_copysign'}.items():
        item = re.sub(r'\b' + before + r'\b', after, item)
    base = json.loads((ROOT / 'woof/metal/kernel_signatures.json').read_text())
    sig = copy.deepcopy(next(x for x in base['kernels'] if x['name'] == 'w_damp'))
    sig.update(name='wrf_exact_w_damp', module='wrf_exact_w_damp',
               source='woof/metal/shaders/wrf_exact_w_damp.metal',
               arithmetic_policies=['wrf-exact'])
    args = []
    for p in params:
        if p.pointer:
            const = 'const ' if 'const' in p.text.split() else ''
            args.append('reinterpret_cast<device ' + const + 'Exact32*>(r.' + p.name + ')')
        else:
            args.append('Exact32(p.' + p.name + ')' if p.base == 'real' else 'p.' + p.name)
    shader = ('// Generated from woof/core/kernels/openbc.cu by tools/w_damp_wrf_exact/generate.py; do not edit.\n'
              '#include <metal_stdlib>\n#include "exact_binary32.metalh"\n'
              '#include "wrf_exact_w_damp_bindings.metalh"\nusing namespace metal;\n'
              'namespace wrf_exact_w_damp_source {\n'
              'static inline Exact32 exact_copysign(Exact32 a, Exact32 b)\n'
              '{ return Exact32::from_bits((a.bits & 0x7fffffffu) | (b.bits & 0x80000000u)); }\n'
              + item + '\n}\n'
              'kernel void wrf_exact_w_damp(\n'
              '    constant WrfExactWDampResources &r [[buffer(0)]],\n'
              '    constant WrfExactWDampParams &p [[buffer(1)]],\n'
              '    uint gid [[thread_position_in_grid]]) {\n'
              '    if (ulong(gid) >= ulong(p.nz - 1) * ulong(p.ny) * ulong(p.nx)) return;\n'
              '    wrf_exact_w_damp_source::w_damp(' + ', '.join(args) + ', int(gid));\n}\n')
    (ROOT / 'woof/metal/shaders/wrf_exact_w_damp.metal').write_text(
        '\n'.join(line.rstrip() for line in shader.splitlines()) + '\n')
    (ROOT / 'woof/metal/signature_fragments/wrf_exact_w_damp.json').write_text(
        json.dumps({'kernels': [sig]}, indent=2) + '\n')


if __name__ == '__main__': main()
