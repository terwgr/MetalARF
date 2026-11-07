"""Translate source-selected strict LBC frame relaxation to Metal."""
import copy,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.mp28_metal.parse import closure,def_name,split_top
from tools.thompson_real_column_parity.host_backend import parse_kernels
def retained(enabled):
 from woof.core.kernels import module_source
 src=f'#define GPUWM_WRF_EXACT 1\n#define GPUWM_WRF_EXACT_C_BIGSTEP {int(enabled)}\n'+module_source('lbc_state')
 pre=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input=src,text=True,capture_output=True,check=True).stdout
 return closure(split_top(pre),('state_specified_relaxation',))[0]
def translate(item):
 name=def_name(item)
 if name=='state_specified_relaxation':
  item=re.sub(r'extern\s+"C"\s+__global__\s*','',item)
  pos=item.index('{');head,body=item[:pos].rstrip(),item[pos:];head=head[:-1]+', int tid)\n'
  body,n=re.subn(r'int tid = blockIdx.x\*blockDim.x \+ threadIdx.x;','',body)
  if n!=1:raise ValueError('unexpected source indexing')
  item=head+body
 item=re.sub(r'\b(__device__|__forceinline__|__restrict__)\b','',item)
 item=re.sub(r'\bint\s*\*', 'thread int*',item)
 item=re.sub(r'(?<!const )\breal\s*\*\s*value\b', 'thread Exact32* value',item)
 item=re.sub(r'\bconst real\s*\*', 'device const Exact32*',item)
 item=re.sub(r'\breal\s*\*', 'device Exact32*',item)
 item=re.sub(r'\breal\b','Exact32',item)
 item=re.sub(r'\bsize_t\b','ulong',item)
 for a,b in {'__fadd_rn':'exact_add','__fsub_rn':'exact_sub','__fmul_rn':'exact_mul','__fdiv_rn':'exact_div'}.items():item=re.sub(r'\b'+a+r'\b',b,item)
 if re.search(r'\b(float|double|blockIdx|threadIdx)\b',item):raise ValueError('untranslated type/index '+name)
 return item
def main():
 registry={x['name']:x for x in json.loads((ROOT/'woof/metal/kernel_signatures.json').read_text())['kernels']};sigs=[]
 for enabled in (0,1):
  stem='wrf_exact_lbc_'+('on' if enabled else 'off');items=retained(enabled);params=parse_kernels('\n'.join(items))['state_specified_relaxation'];defs=[translate(d) for d in items if def_name(d)!='real']
  sig=copy.deepcopy(registry['state_specified_relaxation']);sig.update(name=stem+'_state_specified_relaxation',module=stem,source='woof/metal/shaders/'+stem+'.metal',arithmetic_policies=['wrf-exact']);sig['parameters'].append(['frame_count','int32']);sigs.append(sig)
  args=[]
  for p in params:
   if p.pointer:args.append('reinterpret_cast<device '+('const ' if 'const' in p.text.split() else '')+'Exact32*>(r.'+p.name+')')
   else:args.append('Exact32(p.'+p.name+')' if p.base in ('float','real') else 'p.'+p.name)
  title=''.join(x.title() for x in sig['name'].split('_'))
  shader='''// Generated from the selected CUDA LBC state source; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "%s_bindings.metalh"
using namespace metal;
namespace %s {
'''%(stem,stem)+'\n'.join(defs)+f'''\n}}
kernel void {sig['name']}(constant {title}Resources &r [[buffer(0)]],constant {title}Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {{
 if(ulong(gid)>=ulong(p.nz)*ulong(p.frame_count)) return;
 {stem}::state_specified_relaxation({', '.join(args)},int(gid));
}}\n'''
  (ROOT/'woof/metal/shaders'/f'{stem}.metal').write_text('\n'.join(x.rstrip() for x in shader.splitlines())+'\n')
 (ROOT/'woof/metal/signature_fragments/wrf_exact_lbc.json').write_text(json.dumps({'kernels':sigs},indent=2)+'\n')
if __name__=='__main__':main()
