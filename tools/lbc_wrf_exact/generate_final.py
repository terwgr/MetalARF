"""Translate strict boundary finalization for both BIGSTEP branches."""
import copy,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.mp28_metal.parse import closure,def_name,split_top
from tools.thompson_real_column_parity.host_backend import parse_kernels
from tools.lbc_wrf_exact.generate import translate as common_translate
def retained(enabled):
 from woof.core.kernels import module_source
 src=f'#define GPUWM_WRF_EXACT 1\n#define GPUWM_WRF_EXACT_C_BIGSTEP {int(enabled)}\n'+module_source('lbc_state')
 pre=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input=src,text=True,capture_output=True,check=True).stdout
 return closure(split_top(pre),('finalize_state_field',))[0]
def translate(item):
 if def_name(item)=='finalize_state_field':
  item=re.sub(r'extern\s+"C"\s+__global__\s*','',item);pos=item.index('{');head,body=item[:pos].rstrip(),item[pos:];head=head[:-1]+', int tid)\n';body,n=re.subn(r'int tid = blockIdx.x\*blockDim.x \+ threadIdx.x;','',body)
  if n!=1:raise ValueError('unexpected finalization index')
  item=head+body
 item=common_translate(item);item=re.sub(r'\bisnan\(', 'exact_isnan(',item);item=re.sub(r'\bfmaxf\(', 'exact_max(',item);return item
def main():
 registry={x['name']:x for x in json.loads((ROOT/'woof/metal/kernel_signatures.json').read_text())['kernels']};sigs=[]
 for enabled in (0,1):
  stem='wrf_exact_lbc_final_'+('on' if enabled else 'off');items=retained(enabled);params=parse_kernels('\n'.join(items))['finalize_state_field'];defs=[translate(d) for d in items if def_name(d)!='real']
  sig=copy.deepcopy(registry['finalize_state_field']);sig.update(name=stem,module='lbc_state',source='woof/metal/shaders/'+stem+'.metal',arithmetic_policies=['wrf-exact']);sig['resources'].append(['muts_wrf','float','read']);sig['shapes']['muts_wrf']=[['mny','mnx']];sig['parameters'].append(['has_muts','int32']);sigs.append(sig);args=[]
  for p in params:
   if p.pointer:
    arg='reinterpret_cast<device '+('const ' if 'const' in p.text.split() else '')+'Exact32*>(r.'+p.name+')';args.append('(q.has_muts ? '+arg+' : nullptr)' if p.name=='muts_wrf' else arg)
   else:args.append('Exact32(q.'+p.name+')' if p.base in ('float','real') else 'q.'+p.name)
  title=''.join(x.title() for x in stem.split('_'));shader='''// Generated from the selected CUDA LBC state source; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "%s_bindings.metalh"
using namespace metal;
namespace %s {
static inline bool exact_isnan(Exact32 value){return soft32_is_nan(value.bits);}
'''%(stem,stem+'_impl')+'\n'.join(defs)+f'''\n}}
kernel void {stem}(constant {title}Resources &r [[buffer(0)]],constant {title}Params &q [[buffer(1)]],uint gid [[thread_position_in_grid]]) {{
 if(ulong(gid)>=ulong(q.nz)*ulong(q.ny)*ulong(q.nx)) return;
 {stem}_impl::finalize_state_field({', '.join(args)},int(gid));
}}\n'''
  (ROOT/'woof/metal/shaders'/f'{stem}.metal').write_text('\n'.join(x.rstrip() for x in shader.splitlines())+'\n')
 (ROOT/'woof/metal/signature_fragments/wrf_exact_lbc_final.json').write_text(json.dumps({'kernels':sigs},indent=2)+'\n')
if __name__=='__main__':main()
