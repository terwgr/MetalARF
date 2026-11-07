"""Generate MYNN's contiguous copy of native interface profile prefixes."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
name='hexv842_mpas_bl_mynn_driver_interface_f32'
text=f'''#include <metal_stdlib>
using namespace metal;
#include "{name}_bindings.metalh"
kernel void {name}(constant Hexv842MpasBlMynnDriverInterfaceF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDriverInterfaceF32Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]) {{
    uint nl=uint(p.nl), nc=uint(p.nc);
    if(i>=nc*nl)return;
    uint col=i/nl,k=i%nl;
    r.output[i]=r.input[col*(nl+1)+k];
}}
'''
resources=[['input','float','read'],['output','float','read_write']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_bl_mynn_run','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':[['nc','int32'],['nl','int32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/shaders/{name}.metal').write_text(text);(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
