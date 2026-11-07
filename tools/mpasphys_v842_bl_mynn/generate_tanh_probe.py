"""Build the MYNN tanh primitive entry used by source comparisons."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
name='hexv842_mpas_bl_mynn_tanh_f32';typename='Hexv842MpasBlMynnTanhF32'
source=f'''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_bl_mynn_tanh.metalh"
#include "{name}_bindings.metalh"
kernel void {name}(constant {typename}Resources &r [[buffer(0)]],constant {typename}Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]) {{if(i<uint(p.count))r.output[i]=as_type<float>(mynn_tanh_bits(as_type<uint>(r.input[i])));}}
'''
resources=[['input','float','read'],['output','float','read_write']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/data/mpasphys-maths/core-math/tanhf.c','cuda_symbol':'cr_tanhf','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':[['count','int32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/shaders/{name}.metal').write_text(source)
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
