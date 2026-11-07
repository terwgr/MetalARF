#!/usr/bin/env python3
"""Translate the pinned RRFS TEMPO serial column, excluding melting reflectivity."""
import ast
import importlib.util
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/mp28_metal'))
from parse import closure, def_name, split_top
spec = importlib.util.spec_from_file_location('tempo_generator', ROOT / 'tools/tempo_metal/generate.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)

SOURCE = ROOT / 'woof/hex/mpasphys/rrfs/tempo_v303.py'
OUTPUT = ROOT / 'woof/metal/shaders/hexrrfs_tempo_core.metalh'


def column():
    tree = ast.parse(SOURCE.read_text())
    return next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], 'id', '') == '_COLUMN')


def remove_melting_blocks(source):
    # Each use is conditional on the single source flag; refuse source drift.
    pattern = r'if\s*\(refl10cm_from_melting_flag\)\s*\{'
    matches = list(re.finditer(pattern, source))
    if len(matches) != 3:
        raise ValueError('melting reflectivity guards changed')
    for match in reversed(matches):
        start, i, depth = match.start(), match.end(), 1
        while depth:
            depth += (source[i] == '{') - (source[i] == '}')
            i += 1
        source = source[:start] + source[i:]
    return source


def preprocessed():
    source = remove_melting_blocks(column())
    # Expand per-field kinds before splitting the translation unit.
    source = re.sub(r'^using (TK_\w+) = (float|double);$', r'#define \1 \2', source, flags=re.M)
    proc = subprocess.run(['clang++', '-E', '-P', '-std=c++17', '-x', 'c++',
                           '-DTEMPO_NZ_MAX=TNZMAX', '-'],
                          input=source, text=True, capture_output=True, check=True)
    return shared.rewrite_blob_access(proc.stdout)


def build():
    items = split_top(preprocessed())
    items = [i for i in items if not i.startswith(('typedef', '#pragma', 'struct TempoBlob'))]
    reached, names = closure(items, ['tempo_main_column', 'tempo_subgrid_cloud_radii'])
    forbidden = {'complex_water_ray', 'reflectivity_from_melting_snow',
                 'reflectivity_from_melting_graupel', 'calc_gamma_p'}
    if names & forbidden:
        raise ValueError('unexpected vendor-math path: ' + str(names & forbidden))
    reached = [shared.g.typed_nulls(i, reached) for i in reached]
    pieces = []
    for item in reached:
        if def_name(item) == 'tempo_copy':
            pieces.append('template <typename D, typename S> static inline void tempo_copy(D dst, S src, int n) { for (int k=1; k<=n; ++k) dst[k]=src[k]; }')
        elif def_name(item) == 'tempo_fill':
            pieces.append('template <typename D, typename V> static inline void tempo_fill(D dst, V v, int n) { for (int k=1; k<=n; ++k) dst[k]=F(v); }')
        elif '{' not in item:
            pieces.append('static inline F tempo_cloud_number(F, F, F, F, F, F, F);')
        elif item.startswith('enum'):
            pieces.append(shared.convert(item))
        elif item.startswith('struct TempoParams'):
            pieces.append('struct TempoParams { device const Soft8 *L; };')
        else:
            pieces.append(shared.convert(item))
    text = '\n'.join(pieces)
    text = re.sub(r'\bfloat\b', 'F', text)
    text = text.replace('return a > b ? a : b ;', 'return a > b ? decltype(a+b)(a) : decltype(a+b)(b) ;', 1)
    text = text.replace('return a < b ? a : b ;', 'return a < b ? decltype(a+b)(a) : decltype(a+b)(b) ;', 1)
    text = re.sub(r'(?<![\w:])sqrt\s*\(', 'ntc::tc_sqrt(', text)
    for name in ('exp', 'log', 'log10', 'pow'):
        text = re.sub(r'\bmpas_' + name + r'f?\s*\(', 'ntc::tc_' + name + '(', text)
    text = re.sub(r'\bmpas_cosf\s*\(', 'tempo_cosf(', text)
    text = re.sub(r'\baa_lroundf?\s*\(', 'tempo_lround(', text)
    text = re.sub(r'(?<!device )\b(F|Soft8|bool|int) \* (\w+) = (?=[wW] \.)', r'device \1 * \2 = ', text)
    text = re.sub(r'(?<!device )\b(const )?(F|Soft8|bool|int) \* (\w+) = nullptr', r'device \1\2 * \3 = nullptr', text)
    return text


def main():
    source = column()
    body = source.split('#define TEMPO_FIELDS(TX)', 1)[1].split('struct TempoBlob', 1)[0]
    tree = ast.parse(SOURCE.read_text())
    expected = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and getattr(node.targets[0], 'id', '') == 'TABLE_LAYOUT')
    layout = [(match[1], tuple(map(int, match.groups()[1:]))) for match in re.finditer(
        r'TX\((\w+),(\d+),(\d+),(\d+),(\d+),(\d+)\)', body.replace(' ', ''))]
    if layout != expected:
        raise ValueError('TEMPO column table offsets disagree with the native blob layout')
    offset = 0
    offsets = []
    for name, shape in layout:
        offsets.append(f'constant constexpr long TOFF_{name} = {offset};')
        offset += math.prod(shape)
    OUTPUT.write_text('// Generated by tools/hexrrfs_tempo/generate.py; do not edit.\n'
                      '// Non-melting reflectivity profile only.\n#pragma clang fp contract(off)\n'
                       + '\n'.join(offsets) + '\n'
                      + 'static inline F tempo_cosf(F x) { return f32_word(soft_glibc_solar_cosf(x.bits)); }\n'
                      + 'static inline int tempo_lround(F x) { return int(round(x)); }\n'
                      + 'static inline int tempo_lround(Soft8 x) { return int(round(x)); }\n'
                      + build().rstrip() + '\n')


if __name__ == '__main__':
    main()
