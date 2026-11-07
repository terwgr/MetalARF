"""Generate the native MPAS GF input gather."""
from pathlib import Path
import json
from woof.hex.mpasphys import gf_mpas_v842 as native
from tools.mp28_metal.generate import POLICIES
root=Path(__file__).resolve().parents[2]
name='hexv842_mpas_gf_pack_f32'
profiles=native._PACK_LEVEL_SOURCES
vectors=('ht',)+native._PACK_VECTOR_SOURCES[1:]
resources=[[n,'float','read'] for n in profiles+vectors]+[['levels','float','read_write'],['surface','float','read_write']]
struct='Hexv842MpasGfPackF32'
lines=['#include <metal_stdlib>','using namespace metal;','#include "soft_binary32.metalh"','#include "soft_gfk_pow.metalh"',f'#include "{name}_bindings.metalh"',f'kernel void {name}(constant {struct}Resources &r [[buffer(0)]], constant {struct}Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{',' if(gid>=uint(p.n*p.nz))return;',' int c=int(gid)%p.n,k=int(gid)/p.n, dst=(c*15)*p.nz+k;']
for i,n in enumerate(native.LEVEL_INPUTS):
    if n=='rthraten':lines.append(f' r.levels[dst+{i}*p.nz]=as_type<float>(soft32_add(as_type<uint>(r.rthratenlw[gid]),as_type<uint>(r.rthratensw[gid])));')
    else:
        source={'rthften':'rthdynten','rqvften':'rqvdynten'}.get(n,n)
        lines.append(f' r.levels[dst+{i}*p.nz]=r.{source}[gid];')
lines+=[' if(k==0){']
for i,n in enumerate(vectors[:5]):lines.append(f'  r.surface[8*c+{i}]=r.{n}[c];')
lines+=['  uint pw=soft_gfk_pow_bits(as_type<uint>(r.meshDensity[c]),0x3e800000u);',
        '  r.surface[8*c+5]=as_type<float>(soft32_div(as_type<uint>(p.len_disp),pw));',
        '  r.surface[8*c+6]=r.areaCell[c];r.surface[8*c+7]=p.dt;', ' }','}']
(root/f'woof/metal/shaders/{name}.metal').write_text('\n'.join(lines)+'\n')
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/gf_mpas_v842.py','cuda_symbol':'gf_mpas_pack_inputs','resources':resources,'shapes':{r[0]:[['any_1d']] for r in resources},'parameters':[['n','int32'],['nz','int32'],['len_disp','float32'],['dt','float32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':POLICIES}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
