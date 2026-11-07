"""Translate native output and cold thermodynamics without changing operation order."""
from pathlib import Path
import json,re
from woof.hex.mpasphys.diagnostics_v842 import CUDA_SOURCE
ROOT=Path(__file__).resolve().parents[2]
entries=[]
shader='''#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_scalar.metalh"
#include "hexv842_mpas_diagnostics_f32_bindings.metalh"
#pragma clang fp contract(off)
static inline float native_diagnostic_pow(float a,float b) {
    return as_type<float>(thompson_exner_powf_bits(as_type<uint>(a),as_type<uint>(b)));
}
'''
for name in ('mpas_output_diagnostics_v842','mpas_cold_thermodynamics_v842'):
    match=re.search(r'extern "C" __global__ void '+name+r'\((.*?)\)\s*\{(.*?)\n\}',CUDA_SOURCE,re.S)
    assert match,name
    resources=[];parameters=[]
    for arg in match[1].split(','):
        words=arg.strip().split();field=words[-1].lstrip('*')
        if '*' in arg:resources.append([field,'float','read' if words[0]=='const' else 'read_write'])
        else:parameters.append([field,'int32' if 'int' in words else 'float32'])
    entry='hexv842_'+name+'_f32'
    title=''.join(part.title() for part in entry.split('_'))
    body=match[2].replace('blockIdx.x * blockDim.x + threadIdx.x','int(gid)')
    for field,_,_ in resources:body=re.sub(r'\b'+field+r'\[',f'r.{field}[',body)
    for field,_ in parameters:body=re.sub(r'\b'+field+r'\b','p.'+field,body)
    for old,new in {'__fadd_rn':'tc_add','__fsub_rn':'tc_sub','__fmul_rn':'tc_mul','__fdiv_rn':'tc_div','thompson_aa_powf':'native_diagnostic_pow'}.items():body=body.replace(old,new)
    shader+=f'\nkernel void {entry}(constant {title}Resources &r [[buffer(0)]],constant {title}Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {{'+body+'\n}\n'
    entries.append(dict(name=entry,module='hexv842_mpas_diagnostics_f32',source='woof/metal/shaders/hexv842_mpas_diagnostics_f32.metal',cuda_source='woof/hex/mpasphys/diagnostics_v842.py',cuda_symbol=name,resources=resources,shapes={field:[['any_1d']] for field,_,_ in resources},parameters=parameters,threads_per_group=128,layout_version=1,arithmetic_policies=['default']))
(ROOT/'woof/metal/shaders/hexv842_mpas_diagnostics_f32.metal').write_text(shader)
(ROOT/'woof/metal/signature_fragments/hexv842_mpas_diagnostics_f32.json').write_text(json.dumps({'kernels':entries},indent=2)+'\n')
