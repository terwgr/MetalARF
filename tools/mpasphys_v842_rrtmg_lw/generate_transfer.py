"""Generate native longwave optical-depth and diffusivity stages."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "soft_gfk_exp.metalh"
#pragma clang fp contract(off)
'''
specs=[('taut',[[n,'float','read'] for n in ('gas_tau','taua')]+[['ngb','int32','read'],['taut','float','read_write']],[[n,'int32'] for n in ('ncol','nl')],'''ulong total=ulong(p.ncol)*p.nl*140;
if(ulong(tid)>=total)return;
uint ig=tid%140; ulong collay=ulong(tid)/140;
r.taut[tid]=mpas_add(r.gas_tau[tid],r.taua[collay*16+(r.ngb[ig]-1)]);''','rlw_taut'),
('secdiff',[[n,'float','read'] for n in ('pwvcm','a0','a1','a2')]+[['secdiff','float','read_write']],[['ncol','int32']],'''if(tid>=uint(p.ncol))return;
float pwvcm=r.pwvcm[tid];
for(int ibnd=1;ibnd<=16;++ibnd){
 float sd;
 if(ibnd==1||ibnd==4||ibnd>=10)sd=1.66f;
 else {
  float exponent=mpas_mul(r.a2[ibnd-1],pwvcm);
  sd=mpas_add(r.a0[ibnd-1],mpas_mul(r.a1[ibnd-1],as_type<float>(soft_gfk_exp(as_type<uint>(exponent)))));
  if(sd>1.80f)sd=1.80f;
  if(sd<1.50f)sd=1.50f;
 }
 r.secdiff[ulong(tid)*16+(ibnd-1)]=sd;
}''','rlw_rtrn_secdiff')]
for suffix,resources,params,body,symbol in specs:
    name='hexv842_mpas_rrtmg_lw_'+suffix+'_f32';typ='Hexv842MpasRrtmgLw'+suffix.title()+'F32'
    code=header+f'#include "{name}_bindings.metalh"\n'
    code+=f'kernel void {name}(constant {typ}Resources &r [[buffer(0)]],constant {typ}Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {{\n{body}\n}}\n'
    (root/f'woof/metal/shaders/{name}.metal').write_text(code)
    spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/core/kernels/rrtmg_lw_chain.cu','cuda_symbol':symbol,'resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
    (root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
