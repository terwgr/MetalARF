"""Compare strict NUCOND core caller with host-compiled selected CUDA source."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from woof.wrf_exact import ENABLED,effective_options
if not ENABLED:raise SystemExit('GPUWM_WRF_EXACT=1 is required')
from woof.core.kernels import module_source,module_options
from woof.core.nssl2_nucond import launch_nucond
from woof.core.nssl2_driver_support import NSSL2DriverWorkspace
from woof.metal.nssl2_nucond import NSSL2MetalNucond
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
src=module_source('nssl2_nucond');a=src.index('namespace {',src.index('WRF v4.6.1 NSSL option-18 NUCOND'))
shim=ROOT/'tools/thompson_real_column_parity/cuda_host_shim.h';glibc=ROOT/'woof/core/kernels/glibc_flt32.cuh';header=(ROOT/'woof/core/kernels/nssl2_wrf_math.cuh').read_text()
macros='\n'.join(line for line in header.splitlines() if line.startswith(('#define NSSL2_WRF_CWMASN ','#define NSSL2_WRF_CWMASX ','#define NSSL2_WRF_XVRMN ','#define NSSL2_WRF_XVRMX0 ')))
names=('full_theta','air_density','pressure_pa','exner','w_interface','qv','qc','qr','qi','qs','qndrop','qnr','qni','qns','qnn','qg','qh','qng','qnh','qvolg','qvolh')
params=','.join('float *'+name for name in names)
args=','.join(names[5:15]);frozen=','.join(names[15:])
wrap='''\nextern "C" void run('''+params+''',float *ss,int has,float dt,int conc,int n,int nz,int horizontal,int ccn){
 for(int i=0;i<n;++i){blockIdx.x=i;nssl2_nucond_supersaturation(full_theta,pressure_pa,exner,qv,ss,n);}
 for(int i=0;i<n;++i){blockIdx.x=i;nssl2_nucond_default(full_theta,air_density,pressure_pa,exner,w_interface,ss,'''+args+','+frozen+''',has,dt,conc,n,nz,horizontal,ccn);}}
'''
with tempfile.TemporaryDirectory(prefix='woof-nssl-nucond-') as tmp:
 tmp=Path(tmp);body=src[a:]
 for old,new in (
  ('if (gathered) {','if (gathered) { ++coverage[0];'),
  ('const float pqs = __fdiv_rn(380.0f, pres);','++coverage[1]; const float pqs = __fdiv_rn(380.0f, pres);'),
  ('if (renuc_block) {','if (renuc_block) { ++coverage[2];'),
  ('} else if (qc > 0.0f) {','} else if (qc > 0.0f) { ++coverage[3];'),
  ('if (has_frozen) {','if (has_frozen) { ++coverage[4];'),
  ('if (tmpg < CNOHMN) {','if (tmpg < CNOHMN) { ++coverage[5];'),
  ('int nc = 2 * (int)lroundf','++coverage[6]; int nc = 2 * (int)lroundf'),
 ):
  assert old in body;body=body.replace(old,new,1)
 (tmp/'source.cpp').write_text(f'#include "{shim}"\n#include "{glibc}"\n'+macros+'\nextern "C" { int coverage[7]={0}; }\n'+body+wrap)
 subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(tmp/'source.cpp'),'-o',str(tmp/'source.dylib')],check=True)
 lib=ctypes.CDLL(str(tmp/'source.dylib')); f=lib.run; coverage=(ctypes.c_int*7).in_dll(lib,'coverage');fp=ctypes.POINTER(ctypes.c_float);f.argtypes=[fp]*22+[ctypes.c_int,ctypes.c_float]+[ctypes.c_int]*5;ptr=lambda x:x.ctypes.data_as(fp)
 shape=(4,2,24);nz,ny,nx=shape;n=int(np.prod(shape));rng=np.random.default_rng(3313)
 data={'full_theta':rng.uniform(245,315,shape).astype(np.float32),'air_density':rng.uniform(.55,1.3,shape).astype(np.float32),'pressure_pa':rng.uniform(50000,101000,shape).astype(np.float32),'exner':rng.uniform(.8,1.0,shape).astype(np.float32),'w_interface':rng.uniform(-.5,2.,(nz+1,ny,nx)).astype(np.float32)}
 for key in ('qv','qc','qr','qi','qs','qg','qh'):data[key]=np.power(10.,rng.uniform(-14,-2,shape)).astype(np.float32)
 data['qv'][:]=rng.uniform(0,.03,shape)
 for key in ('qndrop','qnr','qni','qns','qnn','qng','qnh'):data[key]=np.power(10.,rng.uniform(-10,9,shape)).astype(np.float32)
 for key in ('qvolg','qvolh'):data[key]=np.power(10.,rng.uniform(-15,-3,shape)).astype(np.float32)
 for key in ('qc','qr','qi','qs','qndrop','qnr','qni','qns','qnn','qg','qh','qng','qnh','qvolg','qvolh'):data[key].flat[:3]=[0.,-0.,np.nextafter(np.float32(0),np.float32(1))]
 cases=[]
 with MetalContext() as ctx:
  ctx.batch_kernel_calls=False; submit=ctx._submit_kernel_batch
  def sync(*a,**kw):
   v=submit(*a,**kw);v.wait();return v
  ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
  for conc,has in ((False,False),(True,False),(True,True)):
   for ccn in (False,True):
    for dt in (.5,10.):
     reference={key:data[key].copy() for key in names};ss=np.empty(shape,np.float32)
     f(*[ptr(reference[x]) for x in names],ptr(ss),int(has),np.float32(dt),int(conc),n,nz,ny*nx,int(ccn))
     dev={key:xp.asarray(data[key]) for key in names};scratch=xp.zeros(shape)
     launch_nucond(*[dev[x] for x in names[:15]],dt,supersaturation_scratch=scratch,concentration_space=conc,predicted_ccn=ccn,validate_values=False,frozen=tuple(dev[x] for x in names[15:]) if has else None)
     compared=list(names[:4])+list(names[5:15])+(list(names[15:]) if has else [])
     actual={key:dev[key].to_numpy() for key in compared}
     diffs={key:int(np.count_nonzero(reference[key].view(np.uint32)!=actual[key].view(np.uint32))) for key in compared}
     for key in compared:
      mismatches=np.flatnonzero(reference[key].view(np.uint32).ravel()!=actual[key].view(np.uint32).ravel())
      for j in mismatches[:3]:
       print('DIFF',conc,has,ccn,dt,key,int(j),hex(int(reference[key].view(np.uint32).flat[j])),hex(int(actual[key].view(np.uint32).flat[j])),{name:float(data[name].flat[j]) for name in names if name!='w_interface'})
     diffs['supersaturation']=int(np.count_nonzero(ss.view(np.uint32)!=scratch.to_numpy().view(np.uint32)))
     cases.append({'concentration':conc,'frozen':has,'predicted_ccn':ccn,'dt':dt,'words':n*len(diffs),'different':diffs})
  workspace_cases=[]
  order=('qv','qc','qr','qi','qs','qg','qh','qndrop','qnr','qni','qns','qng','qnh','qnn','qvolg','qvolh')
  for ccn in (False,True):
   reference={key:data[key].copy() for key in names};ss=np.empty(shape,np.float32)
   f(*[ptr(reference[x]) for x in names],ptr(ss),1,np.float32(10.),1,n,nz,ny*nx,int(ccn))
   state=xp.asarray(np.stack([data[x] for x in order]));scratch=xp.zeros(shape)
   workspace=NSSL2DriverWorkspace(state=state,category_surface_export=xp.zeros((5,ny,nx)),shape=shape)
   fields=SimpleNamespace(theta=xp.asarray(data['full_theta']),rho=xp.asarray(data['air_density']),pressure=xp.asarray(data['pressure_pa']),pii=xp.asarray(data['exner']),w=xp.asarray(data['w_interface']))
   NSSL2MetalNucond(scratch,10.,predicted_ccn=ccn)(workspace,fields)
   actual=state.to_numpy();diffs={key:int(np.count_nonzero(reference[key].view(np.uint32)!=actual[i].view(np.uint32))) for i,key in enumerate(order)}
   diffs['full_theta']=int(np.count_nonzero(reference['full_theta'].view(np.uint32)!=fields.theta.to_numpy().view(np.uint32)))
   diffs['supersaturation']=int(np.count_nonzero(ss.view(np.uint32)!=scratch.to_numpy().view(np.uint32)))
   workspace_cases.append({'predicted_ccn':ccn,'words':n*len(diffs),'different':diffs})
 print(json.dumps({'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'effective_options':effective_options(module_options('nssl2_nucond')),'cases':cases,'workspace_cases':workspace_cases,'source_coverage':dict(zip(('gather','qvexcess','renucleation','evaporation','frozen','graupel_intercept','adaptive_lround'),coverage))},indent=2))
 if any(v for case in cases+workspace_cases for v in case['different'].values()):raise SystemExit(1)
