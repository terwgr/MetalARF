"""Generate the native ordered Noah module-location update."""
from pathlib import Path
import json,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.noah_v842 import _CORE_CUDA
source=_CORE_CUDA.split('extern "C" __device__ void noah_finish_module_locations(',1)[1].split('\n}',1)[0]
body=source[source.index('{')+1:].replace('locations[','r.locations[').replace('calls[','r.routes[').replace('state[','r.state[')
stem='hexv842_mpas_noah_locations_i32';title='Hexv842MpasNoahLocationsI32'
shader=f'''#include <metal_stdlib>
using namespace metal;
#include "{stem}_bindings.metalh"
kernel void {stem}(constant {title}Resources &r [[buffer(0)]],constant {title}Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {{
if(c)return; int n=p.n;
'''+body+'\n}\n'
(root/f'woof/metal/shaders/{stem}.metal').write_text(shader)
resources=[['locations','int32','read'],['routes','int32','read'],['state','int32','read_write']]
spec={'kernels':[{'name':stem,'module':stem,'source':f'woof/metal/shaders/{stem}.metal','cuda_source':'woof/hex/mpasphys/noah_v842.py','cuda_symbol':'finish_module_locations','resources':resources,'shapes':{r[0]:[['any_1d']] for r in resources},'parameters':[['n','int32']],'threads_per_group':1,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{stem}.json').write_text(json.dumps(spec,indent=2)+'\n')
