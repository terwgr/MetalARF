from pathlib import Path
import json
root=Path.cwd();name='hexv842_mpas_rrtmg_lw_pack_f32'
resources=[['a','float','read'],['lev','float','read'],['surface','float','read']]+[[n,'float','read_write'] for n in ('pavel','tavel','plev','tlev','tsfc','semiss','wkl','wx','reic','relq','resn')]
params=[[n,'int32'] for n in ('ncol','nl')]
shader='''#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_rrtmg_lw_pack_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_pack_f32(constant Hexv842MpasRrtmgLwPackF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwPackF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 if(ulong(tid)>=ulong(p.ncol)*p.nl)return;
 uint c=tid/p.nl,k=tid%p.nl;ulong a=ulong(c)*12*p.nl,lev=ulong(c)*2*(p.nl+1),base=ulong(c)*p.nl;
 r.pavel[tid]=r.a[a+k];r.tavel[tid]=r.a[a+p.nl+k];
 r.plev[ulong(c)*(p.nl+1)+k]=r.lev[lev+k];r.tlev[ulong(c)*(p.nl+1)+k]=r.lev[lev+p.nl+1+k];
 if(k==0){r.plev[ulong(c)*(p.nl+1)+p.nl]=r.lev[lev+p.nl];r.tlev[ulong(c)*(p.nl+1)+p.nl]=r.lev[lev+2*(p.nl+1)-1];r.tsfc[c]=r.surface[ulong(c)*6];for(int b=0;b<16;++b)r.semiss[ulong(c)*16+b]=r.surface[ulong(c)*6+1];}
 ulong gas=ulong(c)*38*p.nl+k;
 r.wkl[gas]=r.a[a+3*p.nl+k];r.wkl[gas+p.nl]=379.e-6f;r.wkl[gas+2*p.nl]=r.a[a+4*p.nl+k];r.wkl[gas+3*p.nl]=319.e-9f;r.wkl[gas+5*p.nl]=1774.e-9f;r.wkl[gas+6*p.nl]=.209488f;
 ulong x=ulong(c)*4*p.nl+k;
 r.wx[x]=.093e-9f;r.wx[x+p.nl]=.251e-9f;r.wx[x+2*p.nl]=.538e-9f;r.wx[x+3*p.nl]=.169e-9f;
 r.reic[base+k]=r.a[a+10*p.nl+k];r.relq[base+k]=r.a[a+9*p.nl+k];r.resn[base+k]=r.a[a+11*p.nl+k];
}
'''
(root/f'woof/metal/shaders/{name}.metal').write_text(shader)
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/rrtmg_lw_v842.py','cuda_symbol':'solve_device','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
