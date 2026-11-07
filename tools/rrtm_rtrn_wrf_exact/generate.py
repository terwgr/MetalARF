"""Generate the closed strict classic RRTM transfer kernels."""
import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = (ROOT / 'woof/metal/shaders/rrtm_lw_family.metal').read_text()
    entries = json.loads((ROOT / 'woof/metal/signature_fragments/rrtm_lw_family.json').read_text())['kernels']
    selected = []
    bodies = []
    for stage in ('down', 'up', 'fluxes'):
        name = 'rrtm_lw_rtrn_' + stage
        start = source.index('kernel void ' + name + '(')
        end = source.find('\nkernel void ', start + 1)
        body = source[start:end if end != -1 else None]
        entry = copy.deepcopy(next(item for item in entries if item['name'] == name))
        body = body.replace(name, 'wrf_exact_' + name).replace('RrtmLwRtrn', 'WrfExactRrtmLwRtrn')
        body = re.sub(r'\bfloat\b', 'Exact32', body)
        for resource, dtype, access in entry['resources']:
            if dtype == 'float':
                qualifier = 'const ' if access == 'read' else ''
                body = re.sub(r'\br\.' + resource + r'\b',
                              f'reinterpret_cast<device {qualifier}Exact32*>(r.{resource})', body)
        body = body.replace('rrtm_lw_planck(', 'exact_rrtm_planck(')
        body = body.replace('exp(-odcld)', 'exact_rrtm_exp(-odcld)')
        body = body.replace('p.bpade', 'Exact32(p.bpade)')
        body = body.replace('p.heatfac', 'Exact32(p.heatfac)')
        body = body.replace('p.fluxfac', 'Exact32(p.fluxfac)')
        if stage == 'up':
            body = body.replace('* semis * exact_rrtm_planck(', '* (semis * exact_rrtm_planck(')
            body = body.replace('r.tbound)[col], band);', 'r.tbound)[col], band));')
            body = body.replace('    for (int lay = 0; lay < p.nlay; ++lay) {',
                                '    reinterpret_cast<device Exact32*>(r.clear_up)[levels + ulong(ig)] = radclru;\n    for (int lay = 0; lay < p.nlay; ++lay) {')
            body = body.replace('        radclru = radclru + (bbu - radclru) * layer_abss;',
                                '        radclru = radclru + (bbu - radclru) * layer_abss;\n        reinterpret_cast<device Exact32*>(r.clear_up)[levels + ulong(lay + 1) * ulong(p.ngpt) + ulong(ig)] = radclru;')
            entry['resources'].append(['clear_up', 'float', 'read_write'])
            entry['shapes']['clear_up'] = [['ncol', 'nlay+1', 'ngpt']]
        elif stage == 'fluxes':
            body = body.replace('] * 0.5f;', '];')
            body = body.replace('if (lev < p.nmodel) {', 'if (lev < p.nlay) {')
            body = body.replace('up_flux *= Exact32(p.fluxfac);', 'up_flux = (up_flux * 0.5f) * Exact32(p.fluxfac);')
            body = body.replace('down_flux *= Exact32(p.fluxfac);', 'down_flux = (down_flux * 0.5f) * Exact32(p.fluxfac);')
            body = body.replace('const Exact32 fnet1 = (next_up - next_down) * Exact32(p.fluxfac);',
                                'const Exact32 fnet1 = (next_up * 0.5f) * Exact32(p.fluxfac) - (next_down * 0.5f) * Exact32(p.fluxfac);')
            marker = '    if (lev == 0) reinterpret_cast<device Exact32*>(r.glw)[col] = down_flux;'
            body = body.replace(marker, '''    Exact32 clear_up_flux = 0.0f, clear_down_flux = 0.0f;
    for (int ig = 0; ig < 140; ++ig) {
        clear_up_flux += reinterpret_cast<device const Exact32*>(r.clear_up)[base + ulong(ig)];
        clear_down_flux += reinterpret_cast<device const Exact32*>(r.clear_down)[base + ulong(ig)];
    }
    clear_up_flux = (clear_up_flux * 0.5f) * Exact32(p.fluxfac);
    clear_down_flux = (clear_down_flux * 0.5f) * Exact32(p.fluxfac);
    reinterpret_cast<device Exact32*>(r.clear_up_flux)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)] = clear_up_flux;
    reinterpret_cast<device Exact32*>(r.clear_down_flux)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)] = clear_down_flux;
    if (lev == p.nlay) {
        reinterpret_cast<device Exact32*>(r.htr)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)] = 0.0f;
        reinterpret_cast<device Exact32*>(r.clear_htr)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)] = 0.0f;
    }
''' + marker)
            marker = '        const Exact32 dp = '
            body = body.replace(marker, '''        Exact32 clear_next_up = 0.0f, clear_next_down = 0.0f;
        for (int ig = 0; ig < 140; ++ig) {
            clear_next_up += reinterpret_cast<device const Exact32*>(r.clear_up)[next_base + ulong(ig)];
            clear_next_down += reinterpret_cast<device const Exact32*>(r.clear_down)[next_base + ulong(ig)];
        }
        const Exact32 clear_fnet1 = (clear_next_up * 0.5f) * Exact32(p.fluxfac) - (clear_next_down * 0.5f) * Exact32(p.fluxfac);
''' + marker)
            marker = '        const Exact32 htr = Exact32(p.heatfac) * (fnet0 - fnet1) / dp / 86400.0f;'
            body = body.replace(marker, '''        const Exact32 htr_day = Exact32(p.heatfac) * (fnet0 - fnet1) / dp;
        reinterpret_cast<device Exact32*>(r.htr)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)] = htr_day;
        reinterpret_cast<device Exact32*>(r.clear_htr)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)] = Exact32(p.heatfac) * ((clear_up_flux - clear_down_flux) - clear_fnet1) / dp;
        const Exact32 htr = htr_day / 86400.0f;''')
            body = body.replace('        reinterpret_cast<device Exact32*>(r.rthratenlw)[',
                                '        if (lev < p.nmodel) reinterpret_cast<device Exact32*>(r.rthratenlw)[')
            for resource in ('clear_up', 'clear_down'):
                entry['resources'].append([resource, 'float', 'read'])
                entry['shapes'][resource] = [['ncol', 'nlay+1', '140']]
            for resource in ('clear_up_flux', 'clear_down_flux'):
                entry['resources'].append([resource, 'float', 'read_write'])
                entry['shapes'][resource] = [['ncol', 'nlay+1']]
            for resource in ('htr', 'clear_htr'):
                entry['resources'].append([resource, 'float', 'read_write'])
                entry['shapes'][resource] = [['ncol', 'nlay+1']]
        entry.update(name='wrf_exact_' + name, module='wrf_exact_rrtm_rtrn',
                     source='woof/metal/shaders/wrf_exact_rrtm_rtrn.metal', arithmetic_policies=['wrf-exact'])
        selected.append(entry)
        bodies.append(body.rstrip())
    header = '''// Generated from classic RRTM RTRN; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "wrf_exact_rrtm_lw_rtrn_down_bindings.metalh"
#include "wrf_exact_rrtm_lw_rtrn_up_bindings.metalh"
#include "wrf_exact_rrtm_lw_rtrn_fluxes_bindings.metalh"
using namespace metal;

static inline Exact32 exact_rrtm_exp(Exact32 x) { return Exact32::from_bits(soft_gfk_exp(x.bits)); }
static inline Exact32 exact_rrtm_planck(device const Exact32 *table, device const Exact32 *wave, Exact32 temperature, int band) {
    const int whole = int(trunc(as_type<float>(temperature.bits)));
    const int row = clamp(int(trunc(as_type<float>((temperature - 159.0f).bits))), 1, 181) - 1;
    const Exact32 fraction = temperature - Exact32(float(whole));
    const ulong lo = ulong(row) * 16ul + ulong(band);
    const ulong hi = ulong(min(row + 1, 180)) * 16ul + ulong(band);
    return wave[band] * (table[lo] + fraction * (table[hi] - table[lo]));
}
'''
    (ROOT / 'woof/metal/shaders/wrf_exact_rrtm_rtrn.metal').write_text(header + '\n\n'.join(bodies) + '\n')
    (ROOT / 'woof/metal/signature_fragments/wrf_exact_rrtm_rtrn.json').write_text(json.dumps({'kernels': selected}, indent=2) + '\n')


if __name__ == '__main__':
    main()
