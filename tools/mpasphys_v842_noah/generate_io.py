"""Generate native Noah level-first MPAS pool packing and publication."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.noah_v842 import _MPAS_IO_CUDA, _MPAS_INPUT_FIELDS, _MPAS_OUTPUT_FIELDS
inputs=list(_MPAS_INPUT_FIELDS)
outputs=list(_MPAS_OUTPUT_FIELDS)
for direction,names in [('from',inputs),('to',outputs)]:
 symbol='noah_'+direction+'_mpas'
 src=_MPAS_IO_CUDA.split('void '+symbol+'(',1)[1]
 body=src[src.index('{')+1:src.index('\n}')]
 body=body[body.index('float* x='):] if direction=='from' else body[body.index('const float* x='):]
 body=body.replace('float* x=xs+local*79;', 'device float* x=r.floats+c*79;').replace('int* ix=ixs+local*7;', 'device int* ix=r.integers+c*7;')
 body=body.replace('const float* x=xs+local*79;', 'device const float* x=r.floats+c*79;')
 if direction=='from':body += '\nr.locations[2*c]=int(c)+1;r.locations[2*c+1]=1;'
 body=re.sub(r'(\w+)\[c\*(?:nv|npres|4)\+(\d+)\]',r'\1[\2*n+c]',body)
 body=re.sub(r'dzs\[start\*4\+(\d+)\]',r'dzs[\1*n]',body)
 for name in names:
  body=re.sub(r'\b'+name+r'\[', 'r.'+name+'[', body)
 if direction=='from':
  for key in ('dt','xice_threshold','micro','convection','isurban','isice','step','usemonalb'):
   body=re.sub(r'\b'+key+r'\b','p.'+key,body)
 stem='hexv842_mpas_noah_'+direction+'_f32'; title=''.join(x.title() for x in stem.split('_'))
 resources=[['floats','float','read_write' if direction=='from' else 'read']]
 if direction=='from':resources += [['integers','int32','read_write'],['locations','int32','read_write']]
 else:resources += [['status','int32','read']]
 for name in names:
  dtype='int32' if _MPAS_INPUT_FIELDS[name]=='integer_surface' else 'float'
  resources.append([name,dtype,'read' if direction=='from' else 'read_write'])
 params=[['n','int32']]
 if direction=='from':params += [['dt','float32'],['xice_threshold','float32']]+[[k,'int32'] for k in ('micro','convection','isurban','isice','step','usemonalb')]
 header='''#include <metal_stdlib>\nusing namespace metal;\n#include "hex_ieee32.metalh"\n#include "cumulus_support.metalh"\n#pragma clang fp contract(off)\n#define AD(a,b) mpas_add(a,b)\n#define SU(a,b) mpas_sub(a,b)\n#define DV(a,b) cu_div(a,b)\n'''+f'#include "{stem}_bindings.metalh"\n'
 guard='if(c>=uint(n)'+(' || r.status[c]' if direction=='to' else '')+')return;\n'
 shader=header+f'kernel void {stem}(constant {title}Resources &r [[buffer(0)]], constant {title}Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {{\nint n=p.n;\n'+guard+body+'\n}\n'
 (root/f'woof/metal/shaders/{stem}.metal').write_text(shader)
 spec={'kernels':[{'name':stem,'module':stem,'source':f'woof/metal/shaders/{stem}.metal','cuda_source':'woof/hex/mpasphys/noah_v842.py','cuda_symbol':symbol,'resources':resources,'shapes':{r[0]:[['any_1d']] for r in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
 (root/f'woof/metal/signature_fragments/{stem}.json').write_text(json.dumps(spec,indent=2)+'\n')
