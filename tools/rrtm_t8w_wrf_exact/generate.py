"""Generate the exact classic RRTM interface-temperature kernel."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]

def main():
    family = ROOT / 'woof/metal/shaders/rrtm_lw_family.metal'
    source = family.read_text()
    start = source.index('kernel void rrtm_lw_t8w(')
    end = source.index('\nkernel void rrtm_lw_mm5atm(', start)
    body = source[start:end].strip()
    body = body.replace('rrtm_lw_t8w', 'wrf_exact_rrtm_lw_t8w')
    body = body.replace('RrtmLwT8W', 'WrfExactRrtmLwT8W')
    body = body.replace('const float ', 'const Exact32 ')
    for field in ('t', 'z_interface', 'fnm', 'fnp', 'tw'):
        const = '' if field == 'tw' else 'const '
        body = re.sub(r'\br\.' + field + r'\b',
                      f'reinterpret_cast<device {const}Exact32*>(r.{field})', body)
    shader = ('// Generated from the classic RRTM interface preparation.\n'
              '#include <metal_stdlib>\n#include "exact_binary32.metalh"\n'
              '#include "wrf_exact_rrtm_lw_t8w_bindings.metalh"\n'
              'using namespace metal;\n' + body + '\n')
    (ROOT / 'woof/metal/shaders/wrf_exact_rrtm_t8w.metal').write_text(shader)
    family_signatures = json.loads((ROOT / 'woof/metal/signature_fragments/rrtm_lw_family.json').read_text())
    entry = copy.deepcopy(next(x for x in family_signatures['kernels'] if x['name'] == 'rrtm_lw_t8w'))
    entry.update(name='wrf_exact_rrtm_lw_t8w', module='wrf_exact_rrtm_t8w',
                 source='woof/metal/shaders/wrf_exact_rrtm_t8w.metal',
                 arithmetic_policies=['wrf-exact'])
    (ROOT / 'woof/metal/signature_fragments/wrf_exact_rrtm_t8w.json').write_text(
        json.dumps({'kernels': [entry]}, indent=2) + '\n')

if __name__ == '__main__':
    main()
