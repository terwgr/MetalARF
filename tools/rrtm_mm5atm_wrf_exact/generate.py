"""Generate classic RRTM's strict MM5ATM column preparation."""
import copy
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]

def main():
    source = (ROOT / 'woof/metal/shaders/rrtm_lw_family.metal').read_text()
    start = source.index('inline float rrtm_lw_o3_layer(')
    end = source.index('kernel void rrtm_lw_setcoef(', start)
    fragment = source[start:end]
    t8w_start = fragment.index('kernel void rrtm_lw_pack_top_down(')
    mm5_start = fragment.index('kernel void rrtm_lw_mm5atm(')
    fragment = fragment[:t8w_start] + fragment[mm5_start:]
    fragment = fragment.replace('rrtm_lw_mm5atm', 'wrf_exact_rrtm_lw_mm5atm')
    fragment = fragment.replace('RrtmLwMm5Atm', 'WrfExactRrtmLwMm5Atm')
    # WOOF clips water before rho * q, but does not clip the product.
    for field in ('qr', 'qs'):
        old = f'pow(max(rho * max(r.{field}[src], 0.0f), 0.0f), 0.75f)'
        new = f'pow(rho * max(r.{field}[src], 0.0f), 0.75f)'
        if old not in fragment: raise ValueError('missing cloud power: ' + field)
        fragment = fragment.replace(old, new)
    # The shared caller computes model ozone and dry mass before zeroing
    # PZ's terminal level; preserve that original top for the no-buffer case.
    original_top = '(p.nlay == p.nz && lay == p.nz - 1 ? r.pw_pa[inlev] / 100.0f : r.pz[levbase + ulong(lay + 1)])'
    fragment = fragment.replace('r.pz[levbase + ulong(lay + 1)]', original_top)
    fragment = re.sub(r'\bfloat\b', 'Exact32', fragment)
    signatures = json.loads((ROOT / 'woof/metal/signature_fragments/rrtm_lw_family.json').read_text())
    entry = copy.deepcopy(next(x for x in signatures['kernels'] if x['name'] == 'rrtm_lw_mm5atm'))
    for field, dtype, access in entry['resources']:
        if dtype != 'float': raise ValueError(field)
        const = 'const ' if access == 'read' else ''
        fragment = re.sub(r'\br\.' + field + r'\b',
                          f'reinterpret_cast<device {const}Exact32*>(r.{field})', fragment)
    for field in ('g', 'r', 'co2vmr', 'n2ovmr', 'ch4vmr'):
        fragment = re.sub(r'\bp\.' + field + r'\b', f'Exact32(p.{field})', fragment)
    fragment = fragment.replace('* 0.01f', '/ 100.0f')
    fragment = fragment.replace('max(', 'exact_max(').replace('min(lay,', 'metal::min(lay,')
    fragment = fragment.replace('pow(', 'exact_gfk_pow(')
    fragment = fragment.replace('0.5f\n        * reinterpret_cast<device Exact32*>(r.pz)[levbase + nlay - 1ul]',
                                '0.5f * (Exact32(0.0f) + reinterpret_cast<device Exact32*>(r.pz)[levbase + nlay - 1ul])')
    header = '''// Generated from classic RRTM column preparation; do not edit.
#include <metal_stdlib>
#include "exact_gfk_math.metalh"
#include "wrf_exact_rrtm_lw_mm5atm_bindings.metalh"
using namespace metal;
static inline bool operator<(Exact32 a, Exact32 b) { return soft32_less(a.bits, b.bits); }
'''
    (ROOT / 'woof/metal/shaders/wrf_exact_rrtm_mm5atm.metal').write_text(header + fragment.rstrip() + '\n')
    entry.update(name='wrf_exact_rrtm_lw_mm5atm', module='wrf_exact_rrtm_mm5atm',
                 source='woof/metal/shaders/wrf_exact_rrtm_mm5atm.metal',
                 arithmetic_policies=['wrf-exact'])
    (ROOT / 'woof/metal/signature_fragments/wrf_exact_rrtm_mm5atm.json').write_text(
        json.dumps({'kernels':[entry]}, indent=2) + '\n')

if __name__ == '__main__': main()
