"""Generate native MYNN driver surface fluxes for stability selector 1."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source
m=re.search(r'__device__ __noinline__ void f_bl_mynn_run\s*\(.*?\)\s*\{',source,re.S);assert m;s=source[m.end():]
a=s.index('// bl_mynn.F90:796');b=s.index('// bl_mynn.F90:823',a);body=s[a:b].replace('mynn_ipow<3>','mynn_cube')
name='hexv842_mpas_bl_mynn_driver_surface_f32'
old=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_stability_f32.metal').read_text()
header=old.split('\nkernel void ')[0].replace('hexv842_mpas_bl_mynn_stability_f32_bindings.metalh',name+'_bindings.metalh').replace('#include "hexv842_mpas_bl_mynn_stability_validate_f32_bindings.metalh"','')
header+='\nstatic inline float mynn_cube(float a){return mynn_mul(mynn_mul(a,a),a);}\n'
profiles=('qv1','rho1','ex1','dz1');columns=('ps1','ts1','ust1','hfx1','qfx1')
outputs=('flqv','flqc','th_sfc','flq','flt','fltv','rmol1','zet','pmz','phh')
macros=''.join(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]\n' for n in profiles)
undef=''.join(f'#undef {n}\n' for n in profiles)
text=header+f'''\nkernel void {name}(constant Hexv842MpasBlMynnDriverSurfaceF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDriverSurfaceF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]){{
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1;float cp=cfg.cp,p1000mb=cfg.p1000mb,rcp=cfg.rcp,xlvcp=cfg.xlvcp,p608=cfg.p608,karman=cfg.karman,gtr=mynn_div(cfg.gravity,300.f);
'''+''.join(f'float {n}=r.{n}[col];\n' for n in columns)+''.join(f'float {n};\n' for n in outputs)+'''float sqcg,cpm,exnerg,phi_m;\n'''+macros+body+'\nphi_m=f_phim(zet);pmz=mpas_sub(phi_m,zet);phh=f_phih(zet);\n'+''.join(f'r.{n}[col]={n};\n' for n in outputs)+undef+'}\n'
resources=[[n,'float','read'] for n in profiles+columns]+[[n,'float','read_write'] for n in outputs]
params=[['nc','int32'],['nl','int32']]+[[n,'float32'] for n in ('cp','p1000mb','rcp','xlvcp','p608','karman','gravity')]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_bl_mynn_run','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/shaders/{name}.metal').write_text(text);(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
