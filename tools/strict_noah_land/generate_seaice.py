"""Translate the retained strict Noah sea-ice source graph, without admission."""
import json,re,subprocess,sys
from pathlib import Path
from tools.mp28_metal.parse import closure,def_name,split_top
from woof.core.kernels import module_source
from woof.wrf_exact import ENABLED
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/thompson_real_column_parity'))
from host_backend import parse_kernels
assert ENABLED
source='#define __launch_bounds__(...)\n'+module_source('noah')
text=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input=source,text=True,capture_output=True,check=True).stdout
items,_=closure(split_top(text),['noah_seaice_column'])
external={'gfk_checkint','gfk_d2f_rn','gfk_exp','gfk_exp2_core','gfk_log','gfk_pow','gfk_powf_log2','gfk_zeroinfnan','real','noah_f2d'}
def translate(item):
    name=def_name(item)
    if name in external or name.startswith('GFK_'):return ''
    item=re.sub(r'\b(__device__|__global__|__forceinline__|__noinline__|__restrict__)\b','',item)
    item=item.replace('extern "C"','')
    if name=='noah_seaice_column':
        item=item.replace('noah_seaice_column(', 'noah_seaice_column_body(')
        item=item.replace('int col = blockIdx.x * blockDim.x + threadIdx.x;', '')
        pos=item.index('{');head,body=item[:pos],item[pos:];i=head.rfind(')');item=head[:i]+', int col'+head[i:]+body
    pos=item.index('{');head,body=item[:pos],item[pos:]
    space='device' if name=='noah_seaice_column' else 'thread'
    head=re.sub(r'\b(const\s+)?(real|float|int)\s*\*',lambda m:space+' '+('const ' if m[1] else '')+m[2]+'*',head)
    head=re.sub(r'\b(real|float)\s*&',r'thread \1&',head)
    item=head+body
    item=re.sub(r'noah_f2d\(([^()]*(?:\[[^\]]*\])?)\) > 0\.0',r'soft32_greater((\1).bits,0u)',item)
    item=item.replace('atomicExch(fatal, 1)','atomic_exchange_explicit(reinterpret_cast<device atomic_int *>(fatal),1,memory_order_relaxed)')
    item=re.sub(r'\b(?:real|float)\b','Exact32',item);item=re.sub(r'\bsize_t\b','ulong',item)
    item=item.replace('__uint_as_float(', 'Exact32::from_bits(')
    for before,after in {'__fadd_rn':'exact_add','__fsub_rn':'exact_sub','__fmul_rn':'exact_mul','__fdiv_rn':'exact_div','fabsf':'exact_abs','fminf':'exact_min','fmaxf':'exact_max','gfk_pow':'exact_gfk_pow','gfk_exp':'exact_gfk_exp','gfk_log':'exact_gfk_log'}.items():item=re.sub(r'\b'+before+r'\b',after,item)
    if name=='nsice_hrtice':
        item=item.replace('static void nsice_hrtice', '// Retain this boundary for source-word agreement; the inline divergence is unresolved.\nstatic __attribute__((noinline)) void nsice_hrtice')
    return item
fragment=next(x for x in json.loads((ROOT/'woof/metal/signature_fragments/noah_family.json').read_text())['kernels'] if x['name']=='noah_seaice_column')
resources={name:(kind,access) for name,kind,access in fragment['resources']}
params=parse_kernels(source)['noah_seaice_column']
args=[]
for arg in params:
    if not arg.pointer:args.append('p.'+arg.name);continue
    name=arg.name.removesuffix('_a');kind,access=resources[name]
    if kind=='float':args.append('reinterpret_cast<device '+('const ' if access=='read' else '')+'Exact32*>(r.'+name+')')
    else:args.append('r.'+name)
args.append('int(gid)')
prefix='''#include <metal_stdlib>
#include "exact_gfk_math.metalh"
#include "wrf_exact_noah_seaice_bindings.metalh"
using namespace metal;
static inline bool operator<(Exact32 a, Exact32 b) { return soft32_less(a.bits,b.bits); }
static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_less_equal(a.bits,b.bits); }
static inline bool operator==(Exact32 a, Exact32 b) { return soft32_equal(a.bits,b.bits); }
static inline bool operator!=(Exact32 a, Exact32 b) { return !soft32_equal(a.bits,b.bits); }
'''
wrapper='kernel void wrf_exact_noah_seaice(constant WrfExactNoahSeaiceResources &r [[buffer(0)]], constant WrfExactNoahSeaiceParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n  if (gid >= uint(p.ny)*uint(p.nx)) return;\n  noah_seaice_column_body('+', '.join(args)+');\n}\n'
out=prefix+'\n\n'.join(filter(None,(translate(x) for x in items)))+'\n'+wrapper
(ROOT/'woof/metal/shaders/wrf_exact_noah_seaice.metal').write_text(
    '\n'.join(line.rstrip() for line in out.splitlines()).rstrip()+'\n')
fragment['name']='wrf_exact_noah_seaice';fragment['module']='wrf_exact_noah_seaice';fragment['source']='woof/metal/shaders/wrf_exact_noah_seaice.metal';fragment['arithmetic_policies']=['wrf-exact']
(ROOT/'woof/metal/signature_fragments/wrf_exact_noah_seaice.json').write_text(json.dumps({'kernels':[fragment]},indent=2)+'\n')
print(f'{len(out)} characters generated from strict sea-ice source; no admission')
