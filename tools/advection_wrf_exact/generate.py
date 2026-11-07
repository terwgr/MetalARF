"""Translate the selected CUDA advection branches to integer binary32 Metal."""
from __future__ import annotations

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

KERNELS = tuple('flux_div_' + name for name in ('scalar', 'u', 'v', 'w'))
EXTENTS = {'scalar': ('nz', 'ny', 'nx'), 'u': ('nz', 'ny', 'nx + 1'),
           'v': ('nz', 'ny + 1', 'nx'), 'w': ('nz + 1', 'ny', 'nx')}


def retained_source(enabled):
    from woof.core.kernels import module_source
    # Match compiler command-line defines before the module preamble as well.
    source = (f'#define GPUWM_WRF_EXACT 1\n#define GPUWM_WRF_EXACT_C_ADVECTION {int(enabled)}\n'
              + module_source('advection'))
    result = subprocess.run(['clang++', '-E', '-P', '-std=c++17', '-x', 'c++', '-'],
                            input=source, text=True, capture_output=True, check=True)
    return closure(split_top(result.stdout), KERNELS)[0]


def translate(item):
    if def_name(item) in KERNELS:
        item = re.sub(r'extern\s+"C"\s+__global__\s*', '', item)
        pos = item.index('{')
        head, body = item[:pos].rstrip(), item[pos:]
        head = head[:-1] + ', int i, int j, int k)\n'
        body, count = re.subn(r'int [ijk] = blockIdx\.[xyz] \* blockDim\.[xyz] \+ threadIdx\.[xyz];', '', body)
        if count != 3:
            raise ValueError('unexpected CUDA index declarations')
        item = head + body
    item = re.sub(r'\b(__device__|__forceinline__|__restrict__)\b', '', item)
    item = re.sub(r'\bconst real\s*\*', 'device const Exact32*', item)
    item = re.sub(r'\breal\s*\*', 'device Exact32*', item)
    item = re.sub(r'\breal\b', 'Exact32', item)
    item = re.sub(r'\bsize_t\b', 'ulong', item)
    for before, after in {'__fdiv_rn': 'exact_div', 'fabsf': 'exact_abs',
                          'fminf': 'exact_min', 'fmaxf': 'exact_max',
                          'copysignf': 'exact_copysign'}.items():
        item = re.sub(r'\b' + before + r'\b', after, item)
    return item


def emit(enabled, registry):
    mode = 'on' if enabled else 'off'
    stem = 'wrf_exact_advection_' + mode
    items = retained_source(enabled)
    params = parse_kernels('\n'.join(items))
    definitions = [translate(item) for item in items if def_name(item) != 'real']
    if re.search(r'\b(float|double)\b', '\n'.join(definitions)):
        raise ValueError('untranslated floating type')
    signatures, wrappers = [], []
    for name in KERNELS:
        sig = copy.deepcopy(registry[name + '_vorder'])
        sig.update(name=stem + '_' + name, module=stem,
                   source='woof/metal/shaders/' + stem + '.metal',
                   arithmetic_policies=['wrf-exact'])
        signatures.append(sig)
        title = ''.join(part.title() for part in sig['name'].split('_'))
        args = []
        for param in params[name]:
            field = 'tend' if param.name == 'tend_out' else param.name
            if param.pointer:
                const = 'const ' if 'const' in param.text.split() else ''
                args.append('reinterpret_cast<device ' + const + 'Exact32*>(r.' + field + ')')
            else:
                args.append('Exact32(p.' + field + ')' if param.base in ('float', 'real') else 'p.' + field)
        ez, ey, ex = (re.sub(r'\b(n[xyz])\b', r'p.\1', e) for e in EXTENTS[name[9:]])
        wrappers.append('\n'.join([
            'kernel void ' + sig['name'] + '(',
            '    constant ' + title + 'Resources &r [[buffer(0)]],',
            '    constant ' + title + 'Params &p [[buffer(1)]],',
            '    uint gid [[thread_position_in_grid]]) {',
            f'    if (ulong(gid) >= ulong({ez}) * ulong({ey}) * ulong({ex})) return;',
            f'    const int i = int(gid % uint({ex}));',
            f'    const int j = int((gid / uint({ex})) % uint({ey}));',
            f'    const int k = int(gid / (uint({ex}) * uint({ey})));',
            '    ' + stem + '::' + name + '(' + ', '.join(args) + ', i, j, k);',
            '}']))
    shader = ('// Generated from woof/core/kernels/advection.cu by tools/advection_wrf_exact/generate.py; do not edit.\n'
              '// Flux formulations transcribed from WRF-ARW module_advect_em.F.\n'
              '#include <metal_stdlib>\n#include "exact_binary32.metalh"\n'
              f'#include "{stem}_bindings.metalh"\nusing namespace metal;\nnamespace {stem} {{\n'
              'static inline bool operator<(Exact32 a, Exact32 b)\n'
              '{ return soft32_less(a.bits, b.bits); }\n'
              'static inline Exact32 exact_copysign(Exact32 a, Exact32 b)\n'
              '{ return Exact32::from_bits((a.bits & 0x7fffffffu) | (b.bits & 0x80000000u)); }\n'
              + '\n'.join(definitions) + '\n}\n' + '\n'.join(wrappers) + '\n')
    (ROOT / 'woof/metal/shaders' / (stem + '.metal')).write_text(
        '\n'.join(line.rstrip() for line in shader.splitlines()) + '\n')
    return signatures


def main():
    base = json.loads((ROOT / 'woof/metal/kernel_signatures.json').read_text())
    registry = {s['name']: s for s in base['kernels']}
    signatures = emit(True, registry) + emit(False, registry)
    (ROOT / 'woof/metal/signature_fragments/wrf_exact_advection.json').write_text(
        json.dumps({'kernels': signatures}, indent=2) + '\n')


if __name__ == '__main__':
    main()
