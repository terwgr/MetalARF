"""Generate native MYNN EDMF plume condensation."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source
m=re.search(r'__device__ __noinline__ void f_condensation_edmf\s*\(.*?\)\s*\{',source,re.S);assert m
j=m.end();depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
s=source[m.end():j-1].replace('MynnFrame mynn_frame(arena);\n','').replace('f_qsat_blend(t, t0c, tice, p, {})','f_qsat_blend(t, t0c, tice, p, 0)')
s=s.replace('((mynn_abs(mpas_sub(qc, qcold))) < (diff))','soft32_less(as_type<uint>(mynn_abs(mpas_sub(qc,qcold))),as_type<uint>(diff))').replace('((zagl) < (100.f))','soft32_less(as_type<uint>(zagl),as_type<uint>(100.f))')
name='hexv842_mpas_bl_mynn_edmf_condensation_f32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_condensation2_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_condensation2_f32_bindings.metalh',name+'_bindings.metalh').replace('#include "hexv842_mpas_bl_mynn_condensation_exp_f32_bindings.metalh"','')
header+='\nstatic inline float mynn_abs(float a){return as_type<float>(as_type<uint>(a)&0x7fffffffu); }\n'
text=header+f'''\nkernel void {name}(constant Hexv842MpasBlMynnEdmfCondensationF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnEdmfCondensationF32Params &cfg [[buffer(1)]],uint tid [[thread_position_in_grid]]){{
if(tid>=uint(cfg.count))return;float qt=r.qt[tid],thl=r.thl[tid],p=r.p[tid],zagl=r.zagl[tid],qc=r.qc[tid],thv=r.thv[tid];float p1000mb=cfg.p1000mb,rcp=cfg.rcp,xlvcp=cfg.xlvcp,t0c=cfg.t0c,rvovrd=cfg.rvovrd;constexpr float tice=240.f;
'''+s+'\nr.qc[tid]=qc;r.thv[tid]=thv;\n}\n'
resources=[[n,'float','read'] for n in ('qt','thl','p','zagl')]+[[n,'float','read_write'] for n in ('thv','qc')]
params=[['count','int32']]+[[n,'float32'] for n in ('p1000mb','rcp','xlvcp','t0c','rvovrd')]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_condensation_edmf','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/shaders/{name}.metal').write_text(text);(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
