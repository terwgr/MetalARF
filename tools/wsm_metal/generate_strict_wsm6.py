"""Generate the strict regional WSM6 column from the retained CUDA source."""
import argparse
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "woof/metal/shaders"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    'wsm_generate', ROOT / 'tools/wsm_metal/generate.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
parser = argparse.ArgumentParser()
parser.add_argument('--tier', type=int, choices=(64, 80), default=64)
tier = parser.parse_args().tier
generator.SOFT['wsm6'] = 'all'
enums, fragments, kernels = generator.translate_tier(
    'wsm6', 'WSM6_KMAX', tier, (('wsm6_column', 'wsm6_column'),))
# Struct-valued quantities must also use the word-backed scalar.
fragments = [
    re.sub(r'\bfloat\b', 'Sf', fragment)
    if fragment.lstrip().startswith('struct Wsm6') else fragment
    for fragment in fragments
]
kernel, params = generator.convert_kernel(
    kernels['wsm6_column'], 'wsm6_column', 'wsm6')
resources = [param for param in params if param['pointer']]
scalars = [param for param in params if not param['pointer']]
resource_declarations = [
    f" device {'const ' if param['const'] else ''}float *{param['name']} [[id({index})]];\n"
    for index, param in enumerate(resources)
]
scalar_declarations = [
    f"{'float' if param['base'] == 'float' else 'int'} {param['name']};\n"
    for param in scalars
]
structures = (
    'struct Wsm6ColumnResources {\n' + ''.join(resource_declarations) +
    '};\nstruct Wsm6ColumnParams {\n' + ''.join(scalar_declarations) + '};\n'
)
support = r'''
#include "hexv842_mpas_thompson_column_math.metalh"
static inline Sf __fdiv_rn(Sf a, Sf b) { return aa_div_impl(a,b); }
static inline Sf gfk_exp(Sf a) { return sf_make(native_tc_math::thompson_aa_expf(f32_word(sf_bits(a))).bits); }
static inline Sf gfk_log(Sf a) { return sf_make(soft_gfk_log(sf_bits(a))); }
static inline Sf gfk_pow(Sf a, Sf b) { return sf_make(soft_gfk_pow_bits(sf_bits(a),sf_bits(b))); }
'''
preamble = (
    '#include <metal_stdlib>\n'
    '#include "wsm_support.metalh"\n'
    '#pragma clang fp contract(off)\n'
)
kernel = kernel.replace(') {\n', f') {{\n using namespace t{tier};\n', 1)
text = (preamble + support + structures + f'\nnamespace t{tier} {{\n'
        + '\n'.join(enums + fragments) + '\n}\n' + kernel)
# External device arrays are native storage: force the source product to Sf.
native_product = 't [ k ] = theta [ id ] * pii [ id ] ;'
assert text.count(native_product) == 1
text = text.replace(native_product, 't [ k ] = Sf(theta [ id ]) * Sf(pii [ id ]) ;')
entrypoint = 'strict_regional_wsm6' + ('_deep' if tier == 80 else '')
text = text.replace('kernel void wsm6_column(', f'kernel void {entrypoint}(')
(OUT / f'{entrypoint}.metal').write_text(text.rstrip() + '\n')
print('emitted', len(text), 'bytes, functions', len(fragments))
