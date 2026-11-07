"""Translate source-selected strict nesting coupling variants."""
import copy,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.mp28_metal.parse import closure,def_name,split_top
from tools.thompson_real_column_parity.host_backend import parse_kernels
from tools.lbc_wrf_exact.generate import translate as common_translate
KERNELS=('couple_nest_field','couple_nest_field_window','couple_nest_field_frame')
def retained(enabled):
 from woof.core.kernels import module_source
 src=f'#define GPUWM_WRF_EXACT 1\n#define GPUWM_WRF_EXACT_C_BIGSTEP {int(enabled)}\n'+module_source('lbc_state')
 pre=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input=src,text=True,capture_output=True,check=True).stdout
 return closure(split_top(pre),KERNELS)[0]
def translate(item):
 if def_name(item) in KERNELS:
  item=re.sub(r'extern\s+"C"\s+__global__\s*','',item);pos=item.index('{');head,body=item[:pos].rstrip(),item[pos:];head=head[:-1]+', int tid)\n';body,n=re.subn(r'int tid = blockIdx.x\*blockDim.x \+ threadIdx.x;','',body)
  if n!=1:raise ValueError('unexpected index')
  item=head+body
 return common_translate(item)
def main():
 registry={x['name']:x for x in json.loads((ROOT/'woof/metal/kernel_signatures.json').read_text())['kernels']};sigs=[]
 for enabled in (0,1):
  stem='wrf_exact_nest_couple_'+('on' if enabled else 'off');items=retained(enabled);params=parse_kernels('\n'.join(items));defs=[translate(d) for d in items if def_name(d)!='real'];wrappers=[]
  for name in KERNELS:
   sig=copy.deepcopy(registry[name]);sig.update(name=stem+'_'+name,module=stem,source='woof/metal/shaders/'+stem+'.metal',arithmetic_policies=['wrf-exact']);sigs.append(sig);title=''.join(x.title() for x in sig['name'].split('_'));args=[]
   for p in params[name]:
    if p.pointer:args.append('reinterpret_cast<device '+('const ' if 'const' in p.text.split() else '')+'Exact32*>(r.'+p.name+')')
    else:args.append('Exact32(q.'+p.name+')' if p.base in ('float','real') else 'q.'+p.name)
   count={'couple_nest_field':'q.nz*q.ny*q.nx','couple_nest_field_window':'q.nz*q.nj*q.ni','couple_nest_field_frame':'q.nz*q.points'}[name]
   wrappers.append(f'''kernel void {sig['name']}(constant {title}Resources &r [[buffer(0)]],constant {title}Params &q [[buffer(1)]],uint gid [[thread_position_in_grid]]) {{
 if(gid>=uint({count})) return;
 {stem}::{name}({', '.join(args)},int(gid));
}}''')
  shader='''// Generated from the selected CUDA LBC state source; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "%s_bindings.metalh"
using namespace metal;
namespace %s {
'''%(stem,stem)+'\n'.join(defs)+'\n}\n'+'\n'.join(wrappers)+'\n';(ROOT/'woof/metal/shaders'/f'{stem}.metal').write_text('\n'.join(x.rstrip() for x in shader.splitlines())+'\n')
 (ROOT/'woof/metal/signature_fragments/wrf_exact_nest_couple.json').write_text(json.dumps({'kernels':sigs},indent=2)+'\n')
if __name__=='__main__':main()
