"""Generate native MPAS GF solve-cell output coupling."""
from pathlib import Path
import json
from woof.hex.mpasphys import gf_mpas_v842 as native
root=Path(__file__).resolve().parents[2];name='hexv842_mpas_gf_scatter_f32'
level=native._SCATTER_LEVEL_NAMES;surface=native._SCATTER_SURFACE_NAMES;integer=native.INTEGER_OUTPUTS
resources=[['levels','float','read'],['surface','float','read'],['integer','int32','read']]
resources += [[n,'float','read_write'] for n in level+surface]+[[n,'int32','read_write'] for n in integer]
struct='Hexv842MpasGfScatterF32'
code=['#include <metal_stdlib>','using namespace metal;',f'#include "{name}_bindings.metalh"',f'kernel void {name}(constant {struct}Resources &r [[buffer(0)]],constant {struct}Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]){{',' if(gid>=uint(p.n*p.nz))return; int c=int(gid)%p.n,k=int(gid)/p.n;']
for i,n in enumerate(level):code.append(f' r.{n}[gid]=r.levels[(c*6+{i})*p.nz+k];')
code.append(' if(k==0){')
for i,n in enumerate(surface):code.append(f'  r.{n}[c]=r.surface[c*6+{i}];')
for i,n in enumerate(integer):code.append(f'  r.{n}[c]=r.integer[c*3+{i}];')
code+=[' }','}']
(root/f'woof/metal/shaders/{name}.metal').write_text('\n'.join(code)+'\n')
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/gf_mpas_v842.py','cuda_symbol':'gf_mpas_scatter_outputs','resources':resources,'shapes':{r[0]:[['any_1d']] for r in resources},'parameters':[['n','int32'],['nz','int32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
