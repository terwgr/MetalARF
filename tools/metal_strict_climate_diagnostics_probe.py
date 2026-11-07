"""Compare the strict Metal climate caller with the selected CUDA source."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from woof.wrf_exact import ENABLED, effective_options
from woof.core.kernels import module_source, module_options
from woof.core.climate_diagnostics import HISTORY, PERSISTENT, update
from woof.io.restart import SERIALIZED_SCRATCH_SLOTS, write_restart, restore_restart
from woof.io.wrf_output_schema import REGISTRY_VAR_META
from woof.config import RunConfig
from woof.core.grid import make_vertical_coord, make_base_state
from woof.core.state import init_theta_perturbation
from woof.metal.climate_diagnostics import clock_refusal
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

if not ENABLED: raise SystemExit('GPUWM_WRF_EXACT=1 is required')
inputs=('t2','q2','tsk','swdown','swddni','u10','v10','raincv','rainncv')
names=(*inputs,*PERSISTENT)
source=module_source('climate_diagnostics')
shim=(ROOT/'tools/thompson_real_column_parity/cuda_host_shim.h').read_text()
args=','.join(('const float *' if name in inputs else 'float *')+name for name in names)
wrapper='\nextern "C" void run('+args+',int n,float dt,float xtime,float secs,int di,int wi,int restart){blockDim.x=1;threadIdx.x=0;for(int i=0;i<n;++i){blockIdx.x=i;clwrf_output_calc('+','.join(names)+',n,dt,xtime,secs,di,wi,restart);}}\n'
with tempfile.TemporaryDirectory(prefix='strict-climate-') as temp:
 temp=Path(temp);(temp/'source.cpp').write_text(shim+'\n'+source+wrapper)
 subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(temp/'source.cpp'),'-o',str(temp/'source.dylib')],check=True)
 f=ctypes.CDLL(str(temp/'source.dylib')).run; fp=ctypes.POINTER(ctypes.c_float)
 f.argtypes=[fp]*len(names)+[ctypes.c_int]+[ctypes.c_float]*3+[ctypes.c_int]*3
 ptr=lambda a:a.ctypes.data_as(fp)
 cfg=RunConfig(nx=8,ny=7,nz=5,dx=1000.,dy=1000.,ztop=5000.,dt=float(os.environ.get('WOOF_CLIMATE_PROBE_DT','15')),run_seconds=180.,diff_opt=0,output_diagnostics=1,diag_int=1,wind_int=2)
 shape=(cfg.ny,cfg.nx); count=cfg.nx*cfg.ny
 assert set(PERSISTENT)<=SERIALIZED_SCRATCH_SLOTS
 for symbol,units,description in HISTORY.values(): assert REGISTRY_VAR_META[symbol] == (description,units)
 coord=make_vertical_coord(cfg.nz,hybrid_opt=cfg.hybrid_opt,etac=cfg.etac)
 base=make_base_state(coord,lambda z:np.full_like(np.asarray(z,dtype=np.float64),300.),p_surf=cfg.p_surf,ztop=cfg.ztop)
 with MetalContext() as context:
  context.batch_kernel_calls=False; submit=context._submit_kernel_batch
  def sync(*a,**kw):
   result=submit(*a,**kw);result.wait();return result
  context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
  make_state=lambda: init_theta_perturbation(cfg,coord,base,lambda x,z:np.zeros((cfg.nz,cfg.ny,cfg.nx)),array_module=xp)
  state=make_state();refs={name:np.zeros(shape,np.float32) for name in PERSISTENT}
  rng=np.random.default_rng(637);words=diff=0
  for step in range(10):
   host={name:rng.uniform(-3,3,shape).astype(np.float32) for name in inputs}
   for name in ('t2','tsk'):host[name]+=np.float32(280)
   for name in ('swdown','swddni'):host[name]+=np.float32(150)
   for name in inputs:
    host[name].flat[:4]=[0.,-0.,np.nextafter(np.float32(0),np.float32(1)), -np.nextafter(np.float32(0),np.float32(1))]
   dev={name:xp.asarray(a) for name,a in host.items()}
   state.physics=SimpleNamespace(fields={name:dev[name] for name in inputs[:7]},cu_raincv=dev['raincv'],microphysics=None)
   # Supply both synthetic precipitation producers through the actual caller.
   caller_cfg=SimpleNamespace(**vars(cfg));caller_cfg.mp_physics=8
   state.physics.microphysics=SimpleNamespace(rainncv=dev['rainncv'])
   state.elapsed_seconds=float(step*cfg.dt);restart=(step==4)
   f(*map(ptr,[*(host[name] for name in inputs),*(refs[name] for name in PERSISTENT)]),count,np.float32(cfg.dt),np.float32(state.elapsed_seconds/60),np.float32(state.elapsed_seconds),cfg.diag_int,cfg.wind_int,int(restart))
   update(state,caller_cfg,is_restart=restart)
   for name in PERSISTENT:
    got=state.existing_scratch(name).to_numpy();delta=got.view(np.uint32)!=refs[name].view(np.uint32)
    c=int(np.count_nonzero(delta));diff+=c;words+=count
    if c: print('DIFF',step,name,c,[(hex(int(x)),hex(int(y))) for x,y in zip(refs[name].view(np.uint32)[delta][:3],got.view(np.uint32)[delta][:3])])
   if step==5:
    state.physics=None;path=temp/'climate.npz';write_restart(path,state,cfg)
    resumed=make_state();restore_restart(path,resumed,cfg)
    for name in PERSISTENT:
     assert np.array_equal(state.existing_scratch(name).to_numpy().view(np.uint32),resumed.existing_scratch(name).to_numpy().view(np.uint32))
    state=resumed
  invalid_clocks=((61.,1,2),(0.,1,2),(float('inf'),1,2),(1e-40,1,2),(15.,0,2),(15.,2**31,2))
  for dt,di,wi in invalid_clocks:
   assert clock_refusal(dt,di,wi)
   for clock,restart in ((0.,False),(15.,False),(0.,True)):
    bad=SimpleNamespace(**vars(caller_cfg));bad.dt=dt;bad.diag_int=di;bad.wind_int=wi
    state.elapsed_seconds=clock
    before=[state.existing_scratch(name).to_numpy().view(np.uint32).copy() for name in PERSISTENT]
    try: update(state,bad,is_restart=restart)
    except ValueError: pass
    else: raise AssertionError('invalid climate clock accepted')
    for name,old in zip(PERSISTENT,before):
     assert np.array_equal(old,state.existing_scratch(name).to_numpy().view(np.uint32))
  print(json.dumps({'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'effective_options':effective_options(module_options('climate_diagnostics')),'words':words,'different':diff,'steps':10,'dt':cfg.dt,'output_planes':len(HISTORY),'persistent_planes':len(PERSISTENT),'checkpoint':'DomainState file; synthetic producer inputs','invalid_clock_cases':len(invalid_clocks)*3}))
  if diff: raise SystemExit(1)
