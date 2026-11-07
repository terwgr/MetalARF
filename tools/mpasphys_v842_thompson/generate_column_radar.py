"""Translate the native Thompson radar driver and its column workspace."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/mp28_metal')]
from parse import split_top, def_name
from translate import convert_table
from generate import convert_kernel, POLICIES
from woof.hex.mpasphys import thompson_v842 as native

name = 'hexv842_mpas_thompson_column_radar_f32'
source = native.cuda_source(device_math_source='// Metal maths', max_levels=256, numerics='strict')
source = source.replace('#define THOMPSON_COLUMN_KMAX 256', '#define THOMPSON_COLUMN_KMAX TCKMAX', 1)
pre = subprocess.run(['clang++', '-E', '-P', '-std=c++17', '-x', 'c++', '-'],
                     input=source, capture_output=True, text=True, check=True).stdout
items = split_top(pre)
struct = next(item for item in items if def_name(item) == 'ThompsonColumn')
for method in re.findall(r'__device__[^\n]*? (\w+)\(', struct):
    if method in ('calc_refl10cm', 'rayleigh_soak_wetgraupel'):
        continue
    match = re.search(r'__device__[^\n]*? ' + method + r'\(', struct)
    end = native._body_end(struct, struct.index('{', match.start()))
    struct = struct[:match.start()] + struct[end:]
spec = spec_from_file_location('column_rewriter', ROOT / 'tools/thompson_column_metal/generate.py')
rewriter = module_from_spec(spec)
spec.loader.exec_module(rewriter)
struct = rewriter.eliminate_gotos(struct)
constants, body = rewriter.convert_column_struct(struct)
body = re.sub(r'^bool (\w+) \[ TCKMAX \] ;$', r'TCB ( \1 , TCKMAX ) ;', body, flags=re.M)
body = body.replace('device Soft8 * tc_wsd ;', 'device Soft8 * tc_wsd ;\ndevice bool * tc_wsb ;')
body = re.sub(r'\bfloat\b', 'F', body).replace('as_type<F>', 'as_type<float>')
body = re.sub(r'(constant F \* \w+ = )(tc_\w+) ;', r'\1(constant F*)\2 ;', body)
item = next(item for item in items if def_name(item) == 'mpas_thompson_radar')
code, params = convert_kernel(item, name, 'ntc', [])
code = code.replace('&p [[buffer(1)]]', '&params [[buffer(1)]]').replace(' = p.', ' = params.')
code = re.sub(r'\bfloat\b', 'F', code)
code = re.sub(r'(device (?:const )?F \*\w+ = )(r\.\w+);', r'\1(device F *)\2;', code)
code = code.replace('int col = int ( gid ) ;', 'int slot = int(gid); if(slot>=params.count)return; int col=params.first+slot;')
local = re.findall(r'\w+ \[ TCKMAX \]', code)
code = re.sub(r'^F (?:\w+ \[ TCKMAX \] ,? ?)+;$',
              lambda m: ' '.join('TCKF ( %s , TCKMAX ) ;' % n for n in re.findall(r'(\w+) \[ TCKMAX \]', m.group())), code, flags=re.M)
nf, nd, nb = body.count('TCF ('), body.count('TCD ('), body.count('TCB (')
nf += len(local)
assert (nf, nd, nb) == tuple(native.workspace_layout()['mpas_thompson_radar'][k] for k in ('float', 'double', 'bool'))
code = code.replace('ThompsonColumn m ;', f'ThompsonColumn m; m.nz=nz; m.tc_wsf=(device F*)r.ws_f+ulong(slot)*{nf}*(nz+1); m.tc_wsd=(device Soft8*)r.ws_d+ulong(slot)*{nd}*(nz+1); m.tc_wsb=(device bool*)r.ws_b+ulong(slot)*{nb}*(nz+1);')
header = '''#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "thompson_column_helpers.metalh"
using namespace ntc;
using tc44::cd;
using tc44::refl_rayleigh_soak_wetgraupel;
#define TCKMAX nz
#define TCSPACE device
#define TCF(n,sz) device F * n=tc_wsf;tc_wsf+=(sz)
#define TCD(n,sz) device Soft8 * n=tc_wsd;tc_wsd+=(sz)
#define TCB(n,sz) device bool * n=tc_wsb;tc_wsb+=(sz)
#define TCKF(n,sz) device F * n=m.tc_wsf;m.tc_wsf+=(sz)
static inline bool tc_any(device bool* a,int lo,int hi){for(int k=lo;k<=hi;k++)if(a[k-lo])return true;return false;}
'''
tables = '\n'.join(convert_table(item) for item in items if '__constant__' in item)
(ROOT / f'woof/metal/shaders/{name}.metal').write_text(header + tables + constants + '\n' + body + f'\n#include "{name}_bindings.metalh"\n' + code)
resources = [[p['name'], 'float', 'read' if p['const'] else 'read_write'] for p in params if p['pointer']]
resources += [['ws_f','float','read_write'], ['ws_d','uint64','read_write'], ['ws_b','uint32','read_write']]
parameters = [[p['name'], 'int32'] for p in params if not p['pointer']] + [['first','int32'], ['count','int32']]
fragment = {'kernels': [dict(name=name, module=name, source=f'woof/metal/shaders/{name}.metal',
    cuda_source='woof/hex/mpasphys/thompson_v842.py', cuda_symbol='mpas_thompson_radar', resources=resources,
    shapes={r[0]: [['any_1d']] for r in resources}, parameters=parameters, threads_per_group=32,
    layout_version=1, arithmetic_policies=POLICIES)]}
(ROOT / f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(fragment, indent=2)+'\n')
print('radar workspace', nf, nd, nb)
