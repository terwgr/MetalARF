"""Generate strict WSM6 preparation from its source-ordered caller."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from woof.core.constants import CUDA_DEFINES

source = (Path(__file__).parent / 'strict_wsm6_prep.cu').read_text()
match = re.search(r'extern "C" __global__ void strict_wsm6_prep\([^)]*\)\s*\{', source)
start = end = match.end()
depth = 1
while depth:
    depth += (source[end] == '{') - (source[end] == '}')
    end += 1
body = source[start:end - 1].replace('blockIdx.x*blockDim.x+threadIdx.x', 'int(gid)')
body = re.sub(r'\bfloat\b', 'F', body)
arrays = 'thb thp alt pressure phb php t rho pii dz zi saved'.split()
ints = 'nz nc thb3d phb3d'.split()
readonly = set(arrays[:6])
header = '''#include <metal_stdlib>
using namespace metal;
#include "soft_word32.metalh"
#pragma clang fp contract(off)
'''
for name in ('P0', 'RCP', 'G'):
    header += f'#define {name} F({float(CUDA_DEFINES[name])!r}f)\n'
header += 'struct Resources {\n' + ''.join(
    f'device {"const " if name in readonly else ""}F* {name} [[id({i})]];\n'
    for i, name in enumerate(arrays)) + '};\n'
header += 'struct Params {\n' + ''.join(f'int {name};\n' for name in ints) + '};\n'
header += ('kernel void strict_regional_wsm6_prep(constant Resources &r [[buffer(0)]], '
           'constant Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n')
header += ''.join(
    f'device {"const " if name in readonly else ""}F* {name}=r.{name};\n'
    for name in arrays)
header += ''.join(f'int {name}=p.{name};\n' for name in ints)
(ROOT / 'woof/metal/shaders/strict_regional_wsm6_prep.metal').write_text(
    header + body + '\n}\n')
