"""Translate both selected CUDA Coriolis branches to integer binary32 Metal."""
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


def retained_source(bigstep):
    from woof.core.kernels import module_source
    source = (f'#define GPUWM_WRF_EXACT 1\n#define GPUWM_WRF_EXACT_C_BIGSTEP {int(bigstep)}\n'
              + module_source('coriolis_map'))
    expanded = subprocess.run(['clang++', '-E', '-P', '-std=c++17', '-x', 'c++', '-'],
                              input=source, text=True, capture_output=True, check=True).stdout
    return closure(split_top(expanded), ('coriolis_curvature',))[0]


def translate(item):
    if def_name(item) == 'coriolis_curvature':
        item = re.sub(r'extern\s+"C"\s+__global__\s*', '', item)
        pos = item.index('{')
        head, body = item[:pos].rstrip(), item[pos:]
        head = head[:-1] + ', int tid)\n'
        body, count = re.subn(r'int tid = blockIdx\.x \* blockDim\.x \+ threadIdx\.x;', '', body)
        if count != 1: raise ValueError('unexpected CUDA thread index')
        item = head + body
    item = re.sub(r'\b(__device__|__forceinline__|__restrict__)\b', '', item)
    item = re.sub(r'\bconst real\s*\*', 'device const Exact32*', item)
    item = re.sub(r'\breal\s*\*', 'device Exact32*', item)
    item = re.sub(r'\breal\b', 'Exact32', item)
    item = re.sub(r'\bsize_t\b', 'ulong', item)
    return item


def emit(bigstep, by_name):
    mode = 'on' if bigstep else 'off'
    stem = 'wrf_exact_coriolis_' + mode
    items = retained_source(bigstep)
    params = parse_kernels('\n'.join(items))['coriolis_curvature']
    definitions = [translate(x) for x in items if def_name(x) != 'real']
    if re.search(r'\b(float|double)\b', '\n'.join(definitions)):
        raise ValueError('untranslated floating type')
    sig = copy.deepcopy(by_name['coriolis_curvature'])
    sig.update(name=stem, module=stem, source='woof/metal/shaders/' + stem + '.metal',
               arithmetic_policies=['wrf-exact'])
    signatures=[sig]
    args=[]
    for p in params:
        field={'fzm':'fnm','fzp':'fnp'}.get(p.name,p.name)
        if p.pointer:
            const='const ' if 'const' in p.text.split() else ''
            args.append('reinterpret_cast<device ' + const + 'Exact32*>(r.' + field + ')')
        else:
            args.append('Exact32(p.' + field + ')' if p.base == 'real' else 'p.' + field)
    title='WrfExactCoriolis' + mode.title()
    shader=('// Generated from woof/core/kernels/coriolis_map.cu by tools/coriolis_wrf_exact/generate.py; do not edit.\n'
            '#include <metal_stdlib>\n#include "exact_binary32.metalh"\n'
            f'#include "{stem}_bindings.metalh"\n'
            + ('#include "wrf_exact_rk_total_mu_bindings.metalh"\n' if bigstep else '')
            + f'using namespace metal;\nnamespace {stem}_source {{\n'
            + '\n'.join(definitions) + '\n}\n'
            + f'kernel void {stem}(constant {title}Resources &r [[buffer(0)]],\n'
            + f'    constant {title}Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n'
            + '    if (ulong(gid) >= ulong(p.nz) * ulong(p.ny + 1) * ulong(p.nx + 1)) return;\n'
            + f'    {stem}_source::coriolis_curvature(' + ', '.join(args) + ', int(gid));\n}\n')
    if bigstep:
        mass=copy.deepcopy(by_name['rk_total_mu'])
        mass.update(name='wrf_exact_rk_total_mu',module=stem,
                    source='woof/metal/shaders/' + stem + '.metal',arithmetic_policies=['wrf-exact'])
        signatures.append(mass)
        shader += '''kernel void wrf_exact_rk_total_mu(
    constant WrfExactRkTotalMuResources &r [[buffer(0)]],
    constant WrfExactRkTotalMuParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= ulong(p.ny) * ulong(p.nx)) return;
    reinterpret_cast<device uint *>(r.out)[gid] = soft32_add(as_type<uint>(r.mub2d[gid]), as_type<uint>(r.mup[gid]));
}
'''
    (ROOT / 'woof/metal/shaders' / (stem+'.metal')).write_text(
        '\n'.join(line.rstrip() for line in shader.splitlines())+'\n')
    return signatures


def main():
    registry=json.loads((ROOT/'woof/metal/kernel_signatures.json').read_text())
    by_name={s['name']:s for s in registry['kernels']}
    sigs=emit(True,by_name)+emit(False,by_name)
    (ROOT/'woof/metal/signature_fragments/wrf_exact_coriolis.json').write_text(
        json.dumps({'kernels':sigs},indent=2)+'\n')


if __name__ == '__main__': main()
