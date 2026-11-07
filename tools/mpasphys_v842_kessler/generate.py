"""Generate native MPAS Kessler Metal kernels from the selected CUDA source."""
import argparse
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OWNER = 'woof/hex/mpasphys/kessler_v842.py'

# Resource and scalar ordering matches the native wrapper boundary.
ENTRIES = {
    'kessler_v842': (
        'column', 't qv qc qr rho pii z dz8w rainnc rainncv work',
        'dt xlv cp ep2 svp1 svp2 svp3 svpt0 rhowater', 'nz nc ns',
    ),
    'kessler_from_mpas': (
        'from', 'theta_m qv qc qr rho_zz zz exner zgrid t qvp qcp qrp rho pii z dz',
        'rvord', 'nz nc ns',
    ),
    'kessler_to_mpas': (
        'to', 'theta_m qv qc qr rho_zz zz exner exner_b rtheta_b rtheta_p pressure_p '
        'pressure_b zgrid surface_pressure tend_sfc_pressure rt_diabatic_tend '
        'dtheta_dt_mp t qvp qcp qrp', 'rvord rd p0 rcv gravity dt_dyn', 'mp_top nz nc ns',
    ),
    'kessler_precip_to_mpas': (
        'precip', 'rho qvp dz rain_local rainnc rainncv precipw i_rainnc',
        'bucket', 'accumulate nz nc ns correct_wb3',
    ),
    'kessler_relhum': (
        'relhum', 't pii qvp pressure_b pressure_p relhum', '', 'n nc ns',
    ),
}

HEADER = '''#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
using namespace ntc;
#pragma clang fp contract(off)
#define A(a,b) tc_add((a),(b))
#define S(a,b) tc_sub((a),(b))
#define M(a,b) tc_mul((a),(b))
#define D(a,b) tc_div((a),(b))
#define MAX(a,b) tc_max((a),(b))
#define MIN(a,b) tc_min((a),(b))
'''


def function_body(source, symbol):
    declaration = re.search(r'extern "C" __global__ void ' + symbol + r'\([^)]*\)\s*\{', source)
    if declaration is None:
        raise ValueError(f'missing native Kessler function {symbol}')
    begin = end = declaration.end()
    depth = 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end - 1], source[:declaration.start()]


def generate(root, output):
    tree = ast.parse((root / OWNER).read_text())
    names = ('CUDA_SOURCE', 'CUDA_RELHUM_SOURCE', 'CORRECT_MODE_SOURCE')
    selected = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names}
    shaders = output / 'woof/metal/shaders'
    fragments = output / 'woof/metal/signature_fragments'
    shaders.mkdir(parents=True, exist_ok=True)
    fragments.mkdir(parents=True, exist_ok=True)

    for symbol, (phase, array_names, scalar_names, integer_names) in ENTRIES.items():
        source = selected['CUDA_RELHUM_SOURCE' if phase == 'relhum' else 'CUDA_SOURCE']
        arrays, scalars, integers = array_names.split(), scalar_names.split(), integer_names.split()
        body, prefix = function_body(source, symbol)
        if phase == 'precip':
            original, corrected, count = selected['CORRECT_MODE_SOURCE']['WB3'][0]
            assert count == 1 and body.count(original) == count
            original_value = original.split('=', 1)[1].removesuffix(';')
            corrected_value = corrected.split('=', 1)[1].removesuffix(';')
            body = body.replace(original, f'float rho_a=correct_wb3 ? {corrected_value} : {original_value};')
        body = body.replace('blockIdx.x*blockDim.x+threadIdx.x', 'int(gid)')
        body = body.replace('float *prodk=work', 'device F *prodk=work')
        body = body.replace('float *rhok=work', 'device F *rhok=work')
        body = re.sub(r'\bfloat\b', 'F', body)
        body = body.replace('if(c>=nc) return;', 'if(c>=ns) return;')
        body = body.replace('if(p>=nz*nc) return;', 'if(p>=nz*nc || p%nc>=ns) return;')
        body = body.replace('if(p>=n) return;', 'if(p>=n || p%nc>=ns) return;')
        for old, new in (('mpas_powf', 'tc_pow'), ('mpas_expf', 'tc_exp'), ('mpas_sqrtf', 'tc_sqrt'),
                         ('__fadd_rn', 'tc_add'), ('__fmul_rn', 'tc_mul'),
                         ('__fsub_rn', 'tc_sub'), ('__fdiv_rn', 'tc_div')):
            body = body.replace(old, new)
        body = body.replace('(1004.0f/287.0f)', 'F(1004.0f/287.0f)')
        helpers = ''
        if phase == 'relhum':
            helpers = re.sub(r'\bfloat\b', 'F', prefix.replace('__device__', 'static inline'))

        entry = 'hexv842_mpas_kessler_' + phase + '_f32'
        title = ''.join(word.title() for word in entry.split('_'))
        parameters = [[name, 'float32'] for name in scalars] + [[name, 'int32'] for name in integers]
        resources = [[name, 'int32' if name == 'i_rainnc' else 'float', 'read_write']
                     for name in arrays]
        declarations = []
        for name in arrays:
            typ = 'int' if name == 'i_rainnc' else 'F'
            declarations.append(f'device {typ}* {name}=(device {typ}*)r.{name};\n')
        declarations.extend(f'F {name}=F(params.{name});\n' for name in scalars)
        declarations.extend(f'int {name}=params.{name};\n' for name in integers)
        declaration = (f'kernel void {entry}(constant {title}Resources &r [[buffer(0)]],'
                       f'constant {title}Params &params [[buffer(1)]],uint gid [[thread_position_in_grid]]) {{\n')
        (shaders / f'{entry}.metal').write_text(
            HEADER + helpers + f'#include "{entry}_bindings.metalh"\n'
            + declaration + ''.join(declarations) + body + '\n}\n')
        spec = dict(name=entry, module=entry, source=f'woof/metal/shaders/{entry}.metal',
                    cuda_source=OWNER, cuda_symbol=symbol, resources=resources,
                    shapes={name: [['any_1d']] for name in arrays}, parameters=parameters,
                    threads_per_group=32, layout_version=1, arithmetic_policies=['default'])
        (fragments / f'{entry}.json').write_text(json.dumps({'kernels': [spec]}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    generate(ROOT, args.output_root)
