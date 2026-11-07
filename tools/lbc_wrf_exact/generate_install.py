"""Translate strict specified MU installation from the selected CUDA source."""
import copy,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.mp28_metal.parse import closure,def_name,split_top
from tools.thompson_real_column_parity.host_backend import parse_kernels
def retained():
 from woof.core.kernels import module_source
 pre=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input='#define GPUWM_WRF_EXACT 1\n'+module_source('lbc_state'),text=True,capture_output=True,check=True).stdout
 return closure(split_top(pre),('install_mu_boundary',))[0]
def translate(item):
 name=def_name(item)
 if name=='install_mu_boundary':
  item=re.sub(r'extern\s+"C"\s+__global__\s*','',item);pos=item.index('{');head,body=item[:pos].rstrip(),item[pos:];head=head[:-1]+', int p)\n';body,n=re.subn(r'int p = blockIdx.x\*blockDim.x \+ threadIdx.x;','',body)
  if n!=1:raise ValueError('unexpected index');item=head+body
  item=head+body
 item=re.sub(r'\b(__device__|__forceinline__|__restrict__)\b','',item);item=re.sub(r'\bint\s*\*','thread int*',item);item=re.sub(r'(?<!const )\breal\s*\*\s*value\b','thread Exact32* value',item);item=re.sub(r'\bconst real\s*\*','device const Exact32*',item);item=re.sub(r'\breal\s*\*','device Exact32*',item);item=re.sub(r'\breal\b','Exact32',item);item=re.sub(r'\bsize_t\b','ulong',item)
 for a,b in {'__fadd_rn':'exact_add','__fsub_rn':'exact_sub','__fmul_rn':'exact_mul','__fdiv_rn':'exact_div'}.items():item=re.sub(r'\b'+a+r'\b',b,item)
 if re.search(r'\b(float|double|blockIdx|threadIdx)\b',item):raise ValueError('untranslated source '+name)
 return item
def main():
 reg=json.loads((ROOT/'woof/metal/kernel_signatures.json').read_text())['kernels'];sig=copy.deepcopy(next(x for x in reg if x['name']=='install_mu_boundary'));stem='wrf_exact_install_mu_boundary';sig.update(name=stem,module='lbc_state',source='woof/metal/shaders/'+stem+'.metal',arithmetic_policies=['wrf-exact']);items=retained();params=parse_kernels('\n'.join(items))['install_mu_boundary'];defs=[translate(x) for x in items if def_name(x)!='real'];args=[]
 for p in params:
  if p.pointer:args.append('reinterpret_cast<device '+('const ' if 'const' in p.text.split() else '')+'Exact32*>(r.'+p.name+')')
  else:args.append('Exact32(q.'+p.name+')' if p.base in ('float','real') else 'q.'+p.name)
 shader='''// Generated from the selected CUDA LBC state source; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_install_mu_boundary_bindings.metalh"
using namespace metal;
namespace wrf_exact_lbc_install {
'''+'\n'.join(defs)+'''\n}
kernel void wrf_exact_install_mu_boundary(constant WrfExactInstallMuBoundaryResources &r [[buffer(0)]],constant WrfExactInstallMuBoundaryParams &q [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if(gid>=uint(q.frame_count)) return;
 wrf_exact_lbc_install::install_mu_boundary('''+', '.join(args)+''',int(gid));
}
'''
 (ROOT/'woof/metal/shaders'/f'{stem}.metal').write_text('\n'.join(x.rstrip() for x in shader.splitlines())+'\n');(ROOT/'woof/metal/signature_fragments'/f'{stem}.json').write_text(json.dumps({'kernels':[sig]},indent=2)+'\n')
if __name__=='__main__':main()
