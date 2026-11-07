"""Translate the selected CUDA surface-w kernel to integer binary32 Metal."""
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
    source = '#define GPUWM_WRF_EXACT 1\n' + module_source('surface_w')
    expanded = subprocess.run(['clang++', '-E', '-P', '-std=c++17', '-x', 'c++', '-'],
                              input=source, text=True, capture_output=True, check=True).stdout
    return closure(split_top(expanded), ('set_surface_w',))[0]


def main():
    items = retained_source()
    params = parse_kernels('\n'.join(items))['set_surface_w']
    item = next(x for x in items if def_name(x) == 'set_surface_w')
    item = re.sub(r'extern\s+"C"\s+__global__\s*', '', item)
    pos = item.index('{')
    head, body = item[:pos].rstrip(), item[pos:]
    head = head[:-1] + ', int col)\n'
    body, count = re.subn(r'int col = blockIdx\.x \* blockDim\.x \+ threadIdx\.x;', '', body)
    if count != 1:
        raise ValueError('unexpected CUDA thread index')
    item = head + body
    item = re.sub(r'\bconst float\s*\*', 'device const Exact32*', item)
    item = re.sub(r'\bfloat\s*\*', 'device Exact32*', item)
    item = re.sub(r'\bfloat\b', 'Exact32', item)
    for before, after in {'__fadd_rn': 'exact_add', '__fsub_rn': 'exact_sub',
                          '__fmul_rn': 'exact_mul', '__fdiv_rn': 'exact_div'}.items():
        item = re.sub(r'\b' + before + r'\b', after, item)
    base = json.loads((ROOT / 'woof/metal/kernel_signatures.json').read_text())
    sig = copy.deepcopy(next(x for x in base['kernels'] if x['name'] == 'set_surface_w'))
    sig.update(name='wrf_exact_set_surface_w', module='wrf_exact_surface_w',
               source='woof/metal/shaders/wrf_exact_surface_w.metal',
               arithmetic_policies=['wrf-exact'])
    aliases = {'mapped': 'has_msf', 'bx': 'boundary_x', 'by': 'boundary_y'}
    args = []
    for p in params:
        field = aliases.get(p.name, p.name)
        if p.pointer:
            const = 'const ' if 'const' in p.text.split() else ''
            args.append('reinterpret_cast<device ' + const + 'Exact32*>(r.' + field + ')')
        else:
            args.append('Exact32(p.' + field + ')' if p.base == 'float' else 'p.' + field)
    shader = ('// Generated from woof/core/kernels/surface_w.cu by tools/surface_w_wrf_exact/generate.py; do not edit.\n'
              '#include <metal_stdlib>\n#include "exact_binary32.metalh"\n'
              '#include "wrf_exact_surface_w_bindings.metalh"\nusing namespace metal;\n'
              'namespace wrf_exact_surface_w {\n' + item + '\n}\n'
              'kernel void wrf_exact_set_surface_w(\n'
              '    constant WrfExactSetSurfaceWResources &r [[buffer(0)]],\n'
              '    constant WrfExactSetSurfaceWParams &p [[buffer(1)]],\n'
              '    uint gid [[thread_position_in_grid]]) {\n'
              '    if (ulong(gid) >= ulong(p.ny) * ulong(p.nx)) return;\n'
              '    wrf_exact_surface_w::set_surface_w(' + ', '.join(args) + ', int(gid));\n}\n')
    (ROOT / 'woof/metal/shaders/wrf_exact_surface_w.metal').write_text(shader)
    (ROOT / 'woof/metal/signature_fragments/wrf_exact_surface_w.json').write_text(
        json.dumps({'kernels': [sig]}, indent=2) + '\n')


if __name__ == '__main__':
    main()
