"""Generate the retained RRFS dycore kernel bodies and their Metal ABI."""
import argparse
import importlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]

def generate(area):
    from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
    module = 'woof.hex.cuda_' + area + '_rrfs'
    source = importlib.import_module(module)._CUDA_SOURCE.replace(CUDA_FTZ_HELPERS, '')
    if area == 'horizontal':
        from woof.hex.cuda_horizontal import _CUDA_SOURCE as shared
        begin = shared.index('extern "C" __global__ void mixing_scaling_f32')
        end = shared.index('\n}', begin) + 2
        source += shared[begin:end].replace('powf(mean_density,', 'rrfs_native_powf(mean_density,')
    matches = list(re.finditer(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\((.*?)\)\s*\{', source, re.S))
    pieces, end, specs, arguments = [], 0, [], {}
    for match in matches:
        symbol = match[1]
        finish, depth = match.end(), 1
        while depth:
            depth += (source[finish] == '{') - (source[finish] == '}')
            finish += 1
        pieces.append(source[end:match.start()])
        end = finish
        name = 'hexv841_rrfs_' + area + '_' + symbol
        title = ''.join(word.title() for word in name.split('_'))
        resources, params, declarations, args = [], [], [], []
        for declaration in match[2].split(','):
            const, kind, pointer, arg = re.fullmatch(r'\s*(const\s+)?(int|float)\s*(\*)?\s*(\w+)\s*', declaration).groups()
            dtype = 'int32' if kind == 'int' else 'float32'
            args.append((arg, bool(pointer), dtype))
            if pointer:
                resources.append([arg, dtype, 'read' if const else 'read_write'])
                declarations.append(f'device {"const " if const else ""}{kind} *{arg} = r.{arg};')
            else:
                params.append([arg, dtype])
                declarations.append(f'{"const " if const else ""}{kind} {arg} = p.{arg};')
        params += [[n, 'int32'] for n in ('dispatch_x', 'dispatch_levels', 'dispatch_species')]
        pieces.append(f'''#include "{name}_bindings.metalh"
kernel void {name}(constant {title}Resources &r [[buffer(0)]],
    constant {title}Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    ''' + '\n    '.join(declarations) + source[match.end():finish])
        spec = dict(name=name, module='hexv841_rrfs_'+area,
                    source='woof/metal/shaders/hexv841_rrfs_'+area+'.metal',
                    cuda_source='woof/hex/cuda_horizontal.py' if symbol == 'mixing_scaling_f32' else 'woof/hex/cuda_'+area+'_rrfs.py', cuda_symbol=symbol,
                    resources=resources, shapes={n:[['any_1d']] for n,_,_ in resources},
                    parameters=params, threads_per_group=128, layout_version=1, arithmetic_policies=['default'])
        specs.append(spec)
        arguments[symbol] = args
    pieces.append(source[end:])
    text = ''.join(pieces)
    text = re.sub(r'\bhalf\b', 'half_value', text)
    text = re.sub(r'\bvertex\b', 'vertex_id', text)
    text = text.replace('(long long)', '').replace('long long', 'int')
    text = re.sub(r'block(?:Dim\.x\s*\*\s*blockIdx\.x|Idx\.x\s*\*\s*blockDim\.x)\s*\+\s*threadIdx\.x', 'int(gid % p.dispatch_x)', text)
    text = re.sub(r'gridDim\.x\s*\*\s*blockDim\.x', 'p.dispatch_x', text)
    text = text.replace('blockIdx.y', 'int((gid / p.dispatch_x) % p.dispatch_levels)').replace('blockIdx.z', 'int(gid / (p.dispatch_x * p.dispatch_levels))')
    text = text.replace('__device__ __forceinline__', 'static inline').replace('__device__', '')
    # Address-space annotations on the retained helper function parameters.
    text = re.sub(r'(?<!device )\b(const\s+)?(int|float)\s*\*(\w+)(?=\s*[,\)])', lambda m: 'device '+(m[1] or '')+m[2]+' *'+m[3], text)
    for old,new in [('fminf','fct_min'),('fmaxf','fct_max'),('mpas_min','fct_min'),('mpas_max','fct_max'),('fabsf','mpas_abs'),('copysignf','mpas_copysign'),('sqrtf','mpas_sqrt')]:
        text = re.sub(r'\b'+old+r'\(', new+'(', text)
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    if any(word in text for word in ('blockIdx.', 'threadIdx.', 'gridDim.', 'blockDim.')):
        raise ValueError('unlowered launch geometry: '+area)
    if area == 'horizontal':
        text = '#include \"hex_rrfs_native_powf.metalh\"\n' + text
    text = '#include <metal_stdlib>\nusing namespace metal;\n#include "hex_fct_ieee32.metalh"\n#pragma clang fp contract(off)\n' + text.strip()+'\n'
    (ROOT/specs[0]['source']).write_text(text)
    for spec in specs:
        (ROOT/'woof/metal/signature_fragments'/f"{spec['name']}.json").write_text(json.dumps({'kernels':[spec]}, indent=2)+'\n')
    return arguments

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('areas', nargs='+')
    args=parser.parse_args()
    mappings={area:generate(area) for area in args.areas}
    path=ROOT/'woof/hex/metal_rrfs_arguments.py'
    current = {} if not path.exists() else __import__('runpy').run_path(str(path))['ARGUMENTS']
    current.update(mappings)
    path.write_text('"""Generated RRFS kernel argument order; see tools/hex_rrfs_dycore/generate.py."""\nARGUMENTS = '+repr(current)+'\n')
