"""Generate the native Thompson relative humidity diagnostic."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
name='hexv842_mpas_thompson_column_relhum_f32'
saturation=(root/'woof/metal/shaders/hexv842_mpas_thompson_column_saturation_f32.metal').read_text()
body=saturation.split('#include "hexv842_mpas_thompson_column_saturation_f32_bindings.metalh"',1)[0]
body+=f'''#include "{name}_bindings.metalh"
kernel void {name}(constant Hexv842MpasThompsonColumnRelhumF32Resources &r [[buffer(0)]], constant Hexv842MpasThompsonColumnRelhumF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {{
 if(i>=uint(p.n))return;
 float t=tc_mul(r.th[i],r.pii[i]);
 float tempc=tc_sub(t,273.16f);
 float qvs=rslf(r.pressure[i],t);
 if(soft32_less_equal(as_type<uint>(tempc),0u))qvs=rsif(r.pressure[i],t);
 float relhum=tc_div(r.qv[i],qvs);
 r.relhum[i]=tc_mul(relhum,100.0f);
}}
'''
(root/f'woof/metal/shaders/{name}.metal').write_text(body)
resources=[[n,'float','read_write' if n=='relhum' else 'read'] for n in ('qv','th','pii','pressure','relhum')]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/thompson_v842.py','cuda_symbol':'mpas_thompson_relhum','resources':resources,'shapes':{r[0]:[['any_1d']] for r in resources},'parameters':[['n','int32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
