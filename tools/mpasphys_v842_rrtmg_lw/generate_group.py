"""Generate source-defined KISS pressure checking and heat packing."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
specs=[('pressure',[['a','float','read'],['status','int32','read_write']],'''if(gid>=uint(p.nc))return;
 ulong at=ulong(gid)*12*p.nl;
 r.status[gid]=soft32_less(as_type<uint>(r.a[at]),as_type<uint>(r.a[at+1]))?1:0;''','mpas_mcica_make'),
('heat',[['htr','float','read'],['heating','float','read_write']],'''if(ulong(gid)>=ulong(p.nc)*p.nl)return;
 uint col=gid/p.nl,k=gid%p.nl;
 r.heating[gid]=r.htr[ulong(col)*(p.nl+1)+k];''','solve_device')]
for suffix,resources,body,symbol in specs:
    name='hexv842_mpas_rrtmg_lw_'+suffix+'_f32';typ='Hexv842MpasRrtmgLw'+suffix.title()+'F32'
    code='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
'''+f'#include "{name}_bindings.metalh"\n'
    code+=f'kernel void {name}(constant {typ}Resources &r [[buffer(0)]],constant {typ}Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {{\n{body}\n}}\n'
    (root/f'woof/metal/shaders/{name}.metal').write_text(code)
    spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/rrtmg_lw_v842.py','cuda_symbol':symbol,'resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':[['nc','int32'],['nl','int32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
    (root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
