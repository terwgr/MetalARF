"""Translate the selected WOOF sixth-order CUDA stencils to Exact32 Metal."""
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

NAMES = ('diff6', 'diff6_seam_u', 'diff6_seam_v')
STEM = 'wrf_exact_diff6'


def retained_source():
    from woof.core.kernels import module_source
    result = []
    for module, names in (('diff6', NAMES[:1]), ('diff6_seam', NAMES[1:])):
        source = '#define GPUWM_WRF_EXACT 1\n' + module_source(module)
        preprocessed = subprocess.run(['clang++', '-E', '-P', '-std=c++17', '-x', 'c++', '-'],
                                      input=source, text=True, capture_output=True, check=True).stdout
        result.extend(x for x in closure(split_top(preprocessed), names)[0] if def_name(x) != 'real')
    return result


def translate(item):
    name = def_name(item)
    item = re.sub(r'extern\s+"C"\s+__global__\s*', '', item)
    head, body = item[:item.index('{')].rstrip(), item[item.index('{'):]
    indices = ('i', 'j', 'k') if name == 'diff6' else ('t', 'k')
    head = head[:-1] + ', ' + ', '.join('int ' + x for x in indices) + ')\n'
    body, count = re.subn(r'int (?:i|j|k|t) = blockIdx\.[xyz] \* blockDim\.[xyz] \+ threadIdx\.[xyz];', '', body)
    if count != len(indices):
        raise ValueError(f'unexpected CUDA indices in {name}')
    item = head + body
    item = re.sub(r'\b(__device__|__forceinline__|__restrict__)\b', '', item)
    item = re.sub(r'\bconst real\s*\*', 'device const Exact32*', item)
    item = re.sub(r'\breal\s*\*', 'device Exact32*', item)
    item = re.sub(r'\breal\b', 'Exact32', item)
    item = re.sub(r'\bsize_t\b', 'ulong', item)
    for old, new in {'__fmul_rn': 'exact_mul', '__fdiv_rn': 'exact_div',
                     'fabsf': 'exact_abs', 'fmaxf': 'exact_max'}.items():
        item = re.sub(r'\b' + old + r'\b', new, item)
    return item


def main():
    items = retained_source()
    params = parse_kernels('\n'.join(items))
    definitions = [translate(item) for item in items]
    if re.search(r'\b(float|double|__f\w+|blockIdx|threadIdx)\b', '\n'.join(definitions)):
        raise ValueError('untranslated source')
    registry = {x['name']: x for x in json.loads((ROOT/'woof/metal/kernel_signatures.json').read_text())['kernels']}
    signatures, wrappers = [], []
    for name in NAMES:
        sig = copy.deepcopy(registry[name])
        sig.update(name='wrf_exact_' + name, module=STEM,
                   source=f'woof/metal/shaders/{STEM}.metal', arithmetic_policies=['wrf-exact'])
        signatures.append(sig)
        title = ''.join(x.title() for x in sig['name'].split('_'))
        args = []
        bounds = {'i_lo': '(p.boundary_x ? 3 : 0)', 'i_hi': '(p.boundary_x ? p.nx-4 : p.nxs-1)',
                  'j_lo': '(p.boundary_y ? 3 : 0)', 'j_hi': '(p.boundary_y ? p.ny-4 : p.nys-1)',
                  'bndc': 'p.boundary_cross'}
        for param in params[name]:
            if param.pointer:
                const = 'const ' if 'const' in param.text.split() else ''
                args.append(f'reinterpret_cast<device {const}Exact32*>(r.{param.name})')
            elif param.name in bounds:
                args.append(bounds[param.name])
            elif param.base == 'real':
                args.append(f'Exact32(p.{param.name})')
            else:
                args.append('p.' + param.name)
        if name == 'diff6':
            index = 'int i=int(gid%uint(p.nxs)), j=int((gid/uint(p.nxs))%uint(p.nys)), k=int(gid/(uint(p.nxs)*uint(p.nys)));'
            extra = 'i, j, k'
            count = 'ulong(p.nlev)*ulong(p.nys)*ulong(p.nxs)'
        else:
            index = 'int t=int(gid%uint(p.h1-p.h0+1)), k=int(gid/uint(p.h1-p.h0+1));'
            extra = 't, k'
            count = 'ulong(p.nlev)*ulong(p.h1-p.h0+1)'
        wrappers.append(f'kernel void {sig["name"]}(constant {title}Resources &r [[buffer(0)]], constant {title}Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n  if (ulong(gid)>={count}) return;\n  {index}\n  {STEM}_impl::{name}(' + ', '.join(args) + ', ' + extra + ');\n}')
    crop = copy.deepcopy(registry['diff6_edge_crop_add'])
    crop.update(name='wrf_exact_diff6_edge_crop', module=STEM,
                source=f'woof/metal/shaders/{STEM}.metal', arithmetic_policies=['wrf-exact'])
    signatures.append(crop)
    wrappers.append('''kernel void wrf_exact_diff6_edge_crop(constant WrfExactDiff6EdgeCropResources &r [[buffer(0)]], constant WrfExactDiff6EdgeCropParams &p [[buffer(1)]], uint flat [[thread_position_in_grid]]) {
  if (flat >= uint(p.nlev)*uint(p.ny)*uint(p.nx)) return;
  uint i=flat%uint(p.nx), j=(flat/uint(p.nx))%uint(p.ny), k=flat/(uint(p.nx)*uint(p.ny));
  uint src=(k*uint(p.ny+7)+j+3)*uint(p.nx+7)+i+3;
  reinterpret_cast<device uint*>(r.tend)[flat]=reinterpret_cast<device const uint*>(r.padded)[src];
}''')
    shader = ('// Generated from WOOF CUDA sixth-order stencils by tools/diff6_wrf_exact/generate.py; do not edit.\n'
              '#include <metal_stdlib>\n#include "exact_binary32.metalh"\n'
              f'#include "{STEM}_bindings.metalh"\n#include "{STEM}_seam_u_bindings.metalh"\n#include "{STEM}_seam_v_bindings.metalh"\n#include "{STEM}_edge_crop_bindings.metalh"\nusing namespace metal;\nnamespace {STEM}_impl {{\n'
              'static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_greater_equal(b.bits, a.bits); }\n'
              + '\n'.join(definitions) + '\n}\n' + '\n'.join(wrappers) + '\n')
    (ROOT/f'woof/metal/shaders/{STEM}.metal').write_text('\n'.join(x.rstrip() for x in shader.splitlines())+'\n')
    (ROOT/f'woof/metal/signature_fragments/{STEM}.json').write_text(json.dumps({'kernels':signatures},indent=2)+'\n')

if __name__ == '__main__': main()
