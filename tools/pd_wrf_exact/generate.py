"""Translate the selected CUDA PD advection branches to integer arithmetic."""
import copy,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.mp28_metal.parse import closure,def_name,split_top
from tools.thompson_real_column_parity.host_backend import parse_kernels
KERNELS=('pd_fluxes','pd_renorm_apply','pd_vertical_sl')
def retained(enabled):
 from woof.core.kernels import module_source
 source=f'#define GPUWM_WRF_EXACT 1\n#define GPUWM_WRF_EXACT_C_ADVECTION {int(enabled)}\n'+module_source('pd_vertical_sl')
 pre=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input=source,text=True,capture_output=True,check=True).stdout
 return closure(split_top(pre),KERNELS)[0]
def translate(item):
 name=def_name(item)
 if name in KERNELS:
  item=re.sub(r'extern\s+"C"\s+__global__\s*','',item)
  pos=item.index('{');head,body=item[:pos].rstrip(),item[pos:];head=head[:-1]+', int i, int j, int k)\n'
  body,n=re.subn(r'int i = blockIdx.x \* blockDim.x \+ threadIdx.x;\s*int j = blockIdx.y;\s*int k = blockIdx.z;','',body)
  if n!=1:raise ValueError('unexpected index body')
  body=re.sub(r'unsigned int active = __ballot_sync\(0xffffffffu,\s*i < nx && j < ny && k < nz\);','',body)
  body=re.sub(r'int lane = threadIdx.x & 31;\s*real sc_l = __shfl_up_sync\(active, sc, 1\);\s*real sc_r = __shfl_down_sync\(active, sc, 1\);\s*bool have_l = lane > 0 && \(active & \(1u << \(lane - 1\)\)\);\s*bool have_r = lane < 31 && \(active & \(1u << \(lane \+ 1\)\)\);','real sc_l = sc, sc_r = sc; bool have_l = false, have_r = false;',body)
  item=head+body
 item=item.replace('int shift = (int)copysignf(floorf(fabsf(cr)),cr);','int shift = pd_shift(cr);')
 item=re.sub(r'__double2float_rn\(__ddiv_rn\(\(double\)\(vel\*sum\),\s*fmax\(1.0e-15,\(double\)fabsf\(cr\)\)\)\)','pd_double_ratio(vel*sum,cr)',item)
 item=re.sub(r'\b(__device__|__forceinline__|__restrict__)\b','',item)
 item=re.sub(r'\bconst real\s*\*','device const Exact32*',item)
 item=re.sub(r'\breal\s*\*','device Exact32*',item)
 item=re.sub(r'\breal\b','Exact32',item)
 item=re.sub(r'\bsize_t\b','ulong',item)
 for before,after in {'__fdiv_rn':'exact_div','fabsf':'exact_abs','fminf':'exact_min','fmaxf':'exact_max','copysignf':'exact_copysign'}.items():item=re.sub(r'\b'+before+r'\b',after,item)
 if re.search(r'\b(__ballot_sync|__shfl\w*|double|floorf|threadIdx)\b',item):raise ValueError('untranslated source '+name)
 return item
def main():
 registry={x['name']:x for x in json.loads((ROOT/'woof/metal/kernel_signatures.json').read_text())['kernels']};sigs=[]
 for flag in (0,1):
  stem='wrf_exact_pd_'+('on' if flag else 'off');items=retained(flag);params=parse_kernels('\n'.join(items));defs=[translate(x) for x in items if def_name(x)!='real'];wrappers=[]
  for name in KERNELS:
   sig=copy.deepcopy(registry['pd_fluxes' if name=='pd_vertical_sl' else name]);sig.update(name=stem+'_'+name,module=stem,source='woof/metal/shaders/'+stem+'.metal',cuda_source='woof/core/kernels/'+('pd_vertical_sl.cu' if name=='pd_vertical_sl' else 'pd_advection.cu'),cuda_symbol=name,arithmetic_policies=['wrf-exact'])
   if name=='pd_renorm_apply' and flag:
    sig['resources'] += [['mub','float','read'],['mup_old','float','read']];sig['shapes'].update(mub=[['ny','nx']],mup_old=[['ny','nx']])
   sigs.append(sig);title=''.join(x.title() for x in sig['name'].split('_'));args=[]
   for p in params[name]:
    field='tend' if p.name=='tend_out' else p.name
    if p.pointer:args.append('reinterpret_cast<device '+('const ' if 'const' in p.text.split() else '')+'Exact32*>(r.'+field+')')
    else:args.append('Exact32(p.'+field+')' if p.base in ('float','real') else 'p.'+field)
   ex='p.nx' if name=='pd_renorm_apply' else '(p.nx+1)';ey='p.ny' if name=='pd_renorm_apply' else '(p.ny+1)';ez='p.nz' if name=='pd_renorm_apply' else '(p.nz+1)'
   wrappers.append(f'''kernel void {sig['name']}(constant {title}Resources &r [[buffer(0)]],constant {title}Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {{
 if (ulong(gid)>=ulong({ex})*ulong({ey})*ulong({ez})) return;
 int i=int(gid%uint({ex})),j=int((gid/uint({ex}))%uint({ey})),k=int(gid/(uint({ex})*uint({ey})));
 {stem}::{name}({', '.join(args)},i,j,k);
}}''')
  shader='''// Generated from the selected CUDA PD source; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "soft_binary64.metalh"
#include "%s_bindings.metalh"
using namespace metal;
namespace %s {
static inline bool operator<(Exact32 a,Exact32 b){return soft32_less(a.bits,b.bits);}
static inline Exact32 exact_copysign(Exact32 a,Exact32 b){return Exact32::from_bits((a.bits&0x7fffffffu)|(b.bits&0x80000000u));}
static inline int pd_shift(Exact32 cr){Soft64I32Result v=soft64_to_i32_rz(soft64_from_f32_bits(cr.bits));return v.valid?v.value:as_type<int>(0x80000000u);}
static inline Exact32 pd_double_ratio(Exact32 numerator,Exact32 cr){
 const ulong a=soft64_from_f32_bits(numerator.bits), b=soft64_from_f32_bits(soft32_abs_bits(cr.bits)), eps=0x3cd203af9ee75616ul;
 return Exact32::from_bits(soft64_to_f32_bits(soft64_div(a,soft64_greater(b,eps)?b:eps)));
}
'''%(stem,stem)+'\n'.join(defs)+'\n}\n'+'\n'.join(wrappers)+'\n'
  (ROOT/'woof/metal/shaders'/f'{stem}.metal').write_text('\n'.join(line.rstrip() for line in shader.splitlines())+'\n')
 (ROOT/'woof/metal/signature_fragments/wrf_exact_pd.json').write_text(json.dumps({'kernels':sigs},indent=2)+'\n')
if __name__=='__main__':main()
