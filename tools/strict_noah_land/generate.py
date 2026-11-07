"""Source-closure translation for the strict Noah land column.

The generated graph is not admitted until its ABI and source comparisons pass.
"""
from pathlib import Path
import json, re, subprocess
from tools.mp28_metal.parse import closure, def_name, split_top
from woof.core.kernels import module_source
from woof.wrf_exact import ENABLED

ROOT=Path(__file__).resolve().parents[2]
assert ENABLED, 'generate with GPUWM_WRF_EXACT=1'
source='#define __launch_bounds__(...)\n'+module_source('noah')
preprocessed=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input=source,text=True,capture_output=True,check=True).stdout
items,_=closure(split_top(preprocessed),['noah_column'])
external={'gfk_checkint','gfk_d2f_rn','gfk_exp','gfk_exp2_core','gfk_log','gfk_pow','gfk_powf_log2','gfk_zeroinfnan','real'}

def translate(item):
    name=def_name(item)
    if name in external or name.startswith('GFK_') or name=='noah_column': return ''
    if name=='noah_f2d':
        return 'static inline Soft8 noah_f2d(Exact32 x) { return Soft8::fromBits(soft64_from_f32_bits(x.bits)); }'
    item=re.sub(r'\b(__device__|__forceinline__|__noinline__|__restrict__)\b','',item)
    if name=='noah_column_body':
        item=item.replace('int col = blockIdx.x * blockDim.x + threadIdx.x;', '')
        pos=item.index('{'); head,body=item[:pos],item[pos:]
        i=head.rfind(')'); head=head[:i]+', int col'+head[i:]; item=head+body
    pos=item.index('{'); head,body=item[:pos],item[pos:]
    # CUDA column/global carriers are device pointers; lower-level helpers
    # consume private temporary arrays and references.
    space='device' if name in ('noah_column_body','noah_glacial_column') else 'thread'
    head=re.sub(r'\b(const\s+)?(real|float|int)\s*\*',lambda m:space+' '+('const ' if m[1] else '')+m[2]+'*',head)
    head=re.sub(r'\b(real|float)\s*&',r'thread \1&',head)
    item=head+body
    item=re.sub(r'\(double\)\s*([A-Za-z_]\w*|(?:\d+(?:\.\d*)?(?:e[+-]?\d+)?f))',r'widen(\1)',item,flags=re.I)
    item=item.replace('(real)reslin','narrow(reslin)')
    item=item.replace('reslin = 0.0;', 'reslin = Soft8(0);')
    item=item.replace('(int)vv[0]', 'as_int(vv[0])').replace('(int)genp[14]', 'as_int(genp[14])')
    item=item.replace('real* glacial_bep', 'device real* glacial_bep')
    item=item.replace('const real* sv =', 'device const real* sv =')
    item=item.replace('const real* vv =', 'device const real* vv =')
    item=re.sub(r'\bdouble\b','Soft8',item)
    item=re.sub(r'\b(?:real|float)\b','Exact32',item)
    item=re.sub(r'\bsize_t\b','ulong',item)
    item=item.replace('__float_as_uint(x)','x.bits')
    item=item.replace('__uint_as_float(', 'Exact32::from_bits(')
    item=re.sub(r'noah_f2d\(([^()]*(?:\[[^\]]*\])?)\) > 0\.0',r'noah_f2d(\1) > Soft8(0)',item)
    for before,after in {'__fadd_rn':'exact_add','__fsub_rn':'exact_sub','__fmul_rn':'exact_mul','__fdiv_rn':'exact_div','fabsf':'exact_abs','fminf':'exact_min','fmaxf':'exact_max','gfk_pow':'exact_gfk_pow','gfk_exp':'exact_gfk_exp','gfk_log':'exact_gfk_log'}.items():
        item=re.sub(r'\b'+before+r'\b',after,item)
    return item

prefix='''#include <metal_stdlib>
#include "exact_gfk_math.metalh"
#include "thompson_aerosol_soft8.metalh"
#include "wrf_exact_noah_column_bindings.metalh"
#include "wrf_exact_noah_column_urban_bindings.metalh"
using namespace metal;
static inline bool operator<(Exact32 a, Exact32 b) { return soft32_less(a.bits,b.bits); }
static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_less_equal(a.bits,b.bits); }
static inline bool operator==(Exact32 a, Exact32 b) { return soft32_equal(a.bits,b.bits); }
static inline bool operator!=(Exact32 a, Exact32 b) { return !soft32_equal(a.bits,b.bits); }
static inline Soft8 widen(Exact32 x) { return Soft8::fromBits(soft64_from_f32_bits(x.bits)); }
static inline Exact32 narrow(Soft8 x) { return Exact32::from_bits(soft64_to_f32_bits(x.bits)); }
static inline int as_int(Exact32 x) { return soft64_to_i32_rz(soft64_from_f32_bits(x.bits)).value; }
'''
graph='\n\n'.join(filter(None,(translate(x) for x in items)))
metal=(ROOT/'woof/metal/shaders/noah_family.metal').read_text()
wrapper=metal[metal.index('kernel void noah_column('):metal.index('kernel void noah_column_urban(')]
wrapper=wrapper.replace('kernel void noah_column(', 'kernel void wrf_exact_noah_column(')
wrapper=wrapper.replace('NoahColumnResources','WrfExactNoahColumnResources').replace('NoahColumnParams','WrfExactNoahColumnParams')
fragment=json.loads((ROOT/'woof/metal/signature_fragments/noah_family.json').read_text())['kernels'][0]
for name,kind,access in fragment['resources']:
    if kind=='float':
        space='device const Exact32*' if access=='read' else 'device Exact32*'
        wrapper=re.sub(r'\br\.'+name+r'\b',f'reinterpret_cast<{space}>(r.{name})',wrapper)
wrapper=wrapper.replace('int(gid), 0, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);',
                        '0, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, int(gid));')
urban_wrapper=metal[metal.index('kernel void noah_column_urban('):metal.index('kernel void noah_check_soil_categories(')]
urban_wrapper=urban_wrapper.replace('kernel void noah_column_urban(', 'kernel void wrf_exact_noah_column_urban(')
urban_wrapper=urban_wrapper.replace('NoahColumnUrbanResources','WrfExactNoahColumnUrbanResources').replace('NoahColumnUrbanParams','WrfExactNoahColumnUrbanParams')
urban_wrapper=urban_wrapper.replace('int(gid), p.urban_opt,', 'p.urban_opt,')
urban_wrapper=urban_wrapper.replace('r.rural_q2k, r.rural_zlvl);', 'r.rural_q2k, r.rural_zlvl, int(gid));')
urban_fragment=json.loads((ROOT/'woof/metal/signature_fragments/noah_urban.json').read_text())['kernels'][0]
for name,kind,access in urban_fragment['resources']:
    if kind=='float':
        space='device const Exact32*' if access=='read' else 'device Exact32*'
        urban_wrapper=re.sub(r'\br\.'+name+r'\b',f'reinterpret_cast<{space}>(r.{name})',urban_wrapper)
out=prefix+graph+'\n'+wrapper+'\n'+urban_wrapper
path=ROOT/'woof/metal/shaders/wrf_exact_noah_land.metal'
path.write_text('\n'.join(line.rstrip() for line in out.splitlines()).rstrip()+'\n')
fragment['name']='wrf_exact_noah_column'
fragment['module']='wrf_exact_noah_land'
fragment['source']='woof/metal/shaders/wrf_exact_noah_land.metal'
fragment['arithmetic_policies']=['wrf-exact']
urban_fragment['name']='wrf_exact_noah_column_urban'
urban_fragment['module']='wrf_exact_noah_land'
urban_fragment['source']='woof/metal/shaders/wrf_exact_noah_land.metal'
urban_fragment['arithmetic_policies']=['wrf-exact']
(ROOT/'woof/metal/signature_fragments/wrf_exact_noah_land.json').write_text(json.dumps({'kernels':[fragment, urban_fragment]},indent=2)+'\n')
print(f'{len(out)} characters generated from the strict source graph; no admission')
