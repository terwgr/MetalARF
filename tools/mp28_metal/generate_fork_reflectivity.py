#!/usr/bin/env python3
"""Translate the generic CUDA reflector used by the MP28 WRF 3.9 caller.

The full column comes from refl.cu; the dry expression comes from refl.py.
These entrypoints preserve that caller's arithmetic independently of MP8.
"""
import ast
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import generate as g
from translate import FN_RENAME, map_types, rewrite_division, tokenize, untok

STEM = 'thompson_aerosol_wrf39_refl_family'


def signature(name, params, cuda_symbol):
    resources, shapes, parameters = [], {}, []
    for p in params:
        if p['pointer']:
            resources.append([p['name'], 'float64_bits' if p['base'] == 'double' else 'float',
                              'read' if p['const'] else 'read_write'])
            shapes[p['name']] = [['any_1d']]
        elif p['base'] != 'double':
            parameters.append([p['name'], 'int32'])
    if any(p['base'] == 'double' and not p['pointer'] for p in params):
        resources.append(['radar_constants','float64_bits','read'])
        shapes['radar_constants'] = [[5]]
    return {'name': name, 'cuda_symbol': cuda_symbol, 'module': STEM,
            'source': 'woof/metal/shaders/'+STEM+'.metal',
            'cuda_source': 'woof/core/kernels/refl.cu', 'resources': resources,
            'shapes': shapes, 'parameters': parameters, 'threads_per_group': 32,
            'layout_version': 1, 'arithmetic_policies': g.POLICIES}


def main():
    FN_RENAME['powf'] = 'fork_powf'
    items = [re.sub(r'\breal\b', 'float', it) for it in g.items_of('refl', '461')]
    reached, _ = g.closure(items, ['refl10cm_thompson_column'])
    helpers, kernel = [], None
    for it in reached:
        kind = g.kind_of(it)
        if kind == 'kernel':
            if g.def_name(it) == 'refl10cm_thompson_column':
                kernel = it
        elif kind == 'struct':
            helpers.append(g.convert_struct(it))
        elif kind == 'table':
            helpers.append(g.convert_table(it))
        else:
            helpers.append(g.convert_function_item(it))
    for before, after in [('graupel_number_shadow','ng'),('t','temperature'),('p','pressure')]:
        kernel = re.sub(r'\b'+before+r'\b', after, kernel)
    name = 'thompson_aa_wrf39_refl10cm'
    full, params = g.convert_kernel(kernel, name, 'aa_fork_refl', [])
    doubles = [p for p in params if not p['pointer'] and p['base'] == 'double']
    for index, p in enumerate(doubles):
        full = full.replace('= p.'+p['name']+';',
                            '= Soft8::fromBits(r.radar_constants['+str(index)+']);')
    full = re.sub(r'(?<!device )const Soft8 \* (xxds|xdts|simpson) =',
                  r'device const Soft8 * \1 =', full)
    sigs = [signature(name, params, 'refl10cm_thompson_column')]
    # Preserve the separate scalar CUDA expression selected by melting=False.
    tree = ast.parse((ROOT/'woof/core/refl.py').read_text())
    dry_source = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == '_THOMPSON_DRY_SOURCE'
                              for t in node.targets))
    dry_source = re.sub(r'\bt\b', 'temperature', dry_source)
    dry_source = re.sub(r'\bp\b', 'pressure', dry_source)
    dry_body = untok(rewrite_division(map_types(tokenize(dry_source))))
    dry_name = 'thompson_aa_wrf39_refl10cm_dry'
    dry = ['kernel void '+dry_name+'(',
           '    constant ThompsonAaWrf39Refl10CmDryResources &r [[buffer(0)]],',
           '    constant ThompsonAaWrf39Refl10CmDryParams &p [[buffer(1)]],',
           '    uint gid [[thread_position_in_grid]]) {',
           '    if (gid >= uint(p.n)) return;', '    using namespace aa_fork_refl;']
    dry_params = []
    for field in ('qv','qr','nr','qs','qg','ng','temperature','pressure'):
        dry.append('    const float '+field+' = r.'+field+'[gid];')
        dry_params.append({'name':field,'pointer':True,'base':'float','const':True})
    dry.extend(['    float dbz;',dry_body,'    r.refl[gid] = dbz;','}'])
    dry_params.extend([{'name':'refl','pointer':True,'base':'float','const':False},
                       {'name':'n','pointer':False,'base':'int','const':True}])
    sigs.append(signature(dry_name,dry_params,'refl10cm_thompson_dry'))
    wrappers = """
static inline Soft8 pow(Soft8 x, Soft8 y) { return glibc_pow(x,y); }
static inline Soft8 exp(Soft8 x) { return glibc_exp(x); }
static inline Soft8 log(Soft8 x) { return glibc_log(x); }
static inline Soft8 log10(Soft8 x) { return aa_refl::thompson_aa_log10(x); }
static inline Soft8 hypot(Soft8 x, Soft8 y) { return aa_refl::thompson_aa_hypot(x,y); }
static inline Soft8 atan2(Soft8 y, Soft8 x) { return aa_refl::thompson_aa_atan2(y,x); }
static inline float fork_powf(float x, float y) { return as_type<float>(p3_reference_pow_bits(as_type<uint>(x),as_type<uint>(y))); }
"""
    header = '// Generated by tools/mp28_metal/generate_fork_reflectivity.py; do not edit.\n'
    (ROOT/'woof/metal/shaders/thompson_aerosol_wrf39_refl_fragment.metalh').write_text(
        header+'#pragma clang fp contract(off)\nnamespace aa_fork_refl {\n'+wrappers
        +'\n'.join(helpers)+'\n}\n')
    (ROOT/'woof/metal/shaders'/ (STEM+'.metal')).write_text(
        header+'#include <metal_stdlib>\n'
        '#include "thompson_aerosol_soft8.metalh"\n'
        '#include "p3_reference_pow.metalh"\n'
        '#include "thompson_aerosol_refl_fragment.metalh"\n'
        '#include "thompson_aerosol_wrf39_refl_fragment.metalh"\n'
        '#include "'+STEM+'_bindings.metalh"\n'+full+'\n'+'\n'.join(dry)+'\n')
    (ROOT/'woof/metal/signature_fragments'/ (STEM+'.json')).write_text(
        json.dumps({'kernels':sigs},indent=2)+'\n')


if __name__ == '__main__':
    main()
