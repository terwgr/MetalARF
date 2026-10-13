#!/usr/bin/env python3
"""Replay Smagorinsky coefficients from an actual dump_backend_step1 capture.

Host mode uses CUDA source via the CPU shim (host libm, contraction disabled).
Metal mode uses three native repetitions against the saved host outputs.
"""
from pathlib import Path
import argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--capture-root',type=Path,required=True)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--backend',choices=['host','metal'],required=True)
p.add_argument('--reference-root',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
options=a
if options.backend=='host':
 from pathlib import Path
 from types import SimpleNamespace
 import ctypes,json,sys,numpy as np
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent/'thompson_real_column_parity'))
 import host_backend
 host_backend.install()
 # The generic host launcher used tx/ty as loop variables, shadowing this
 # module's pointer arguments of the same names. Rename launcher loops only.
 launcher=host_backend._launcher
 def safe_launcher(*args):
  s=launcher(*args)
  for v,dim in [('tx','x'),('ty','y'),('tz','z')]:
   s=s.replace('for (unsigned '+v+' = 0; '+v+' < b'+dim+'; ++'+v+')', 'for (unsigned host_lane_'+v+' = 0; host_lane_'+v+' < b'+dim+'; ++host_lane_'+v+')')
   s=s.replace('threadIdx.'+dim+' = '+v+';', 'threadIdx.'+dim+' = host_lane_'+v+';')
  return s
 host_backend._launcher=safe_launcher
 from woof.core import dycore
 from woof.experiment import load_experiment
 from woof.metal.generated import kernel_specs
 root=options.capture_root;events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()];cfg=load_experiment(options.config).domains[0].run
 kernel=next(e for e in events if e['point']=='acoustic_uv');fields=kernel_specs.ADVANCE_UV_PARAM_FIELDS;P=type('P',(ctypes.Structure,),{'_fields_':[(n,getattr(ctypes,t.split('.')[-1])) for n,t in fields]});p=P.from_buffer_copy(bytes.fromhex(kernel['parameters']['parameter_bytes_hex']))
 def load(e,n):return np.load(root/e['arrays'][n]['blob']).copy()
 class State(SimpleNamespace):
  def scratch(self,shape,slot):return self._scratch.setdefault(slot,np.zeros(shape,np.float32))
 def state(e):return State(**{n[6:]:load(e,n) for n in e['arrays'] if n.startswith('state.')},_scratch={},cf1=p.cf1,cf2=p.cf2,cf3=p.cf3,has_msf=True)
 def pairs(pt):
  stack=[]
  for e in events:
   if e['point']!=pt:continue
   if e['side']=='before':stack.append(e)
   else:yield stack.pop(),e
 rows=[]; saved={}
 def compare(b,a,n,host,actual):
  d=actual.astype(float)-host.astype(float);rows.append({'point':b['point'],'field':n,'rmse':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.abs(d).max()),'reference_max':float(np.abs(host).max()),'index':list(map(int,np.unravel_index(np.abs(d).argmax(),d.shape)))})
 for b,a in pairs('smag_coefficients'):
  s=state(b);km=load(b,'input.arg2');kh=load(b,'input.arg3');deform=dycore.launch_wrf_smag2d_km(s,cfg,km,kh,time_t=True)
  saved.update({'km':km,'kh':kh,**{'deformation'+str(i):v for i,v in enumerate(deform)}});compare(b,a,'km',km,load(a,'input.arg2'));compare(b,a,'kh',kh,load(a,'input.arg3'))
  for i,v in enumerate(deform):compare(b,a,'deformation'+str(i),v,load(a,'input.return'+str(i)))
 np.savez((options.output/'outputs.npz'),**saved)
 (options.output/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
else:
 from pathlib import Path
 from types import SimpleNamespace
 import ctypes,json,sys,numpy as np
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
 from woof.metal.runtime import MetalContext
 from woof.metal.array import MetalArrayModule
 from woof.core import dycore
 from woof.experiment import load_experiment
 from woof.metal.generated import kernel_specs
 root=options.capture_root;events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()];cfg=load_experiment(options.config).domains[0].run
 kernel=next(e for e in events if e['point']=='acoustic_uv');fields=kernel_specs.ADVANCE_UV_PARAM_FIELDS;P=type('P',(ctypes.Structure,),{'_fields_':[(n,getattr(ctypes,t.split('.')[-1])) for n,t in fields]});p=P.from_buffer_copy(bytes.fromhex(kernel['parameters']['parameter_bytes_hex']))
 b=next(e for e in events if e['point']=='smag_coefficients' and e['side']=='before');a=next(e for e in events if e['point']=='smag_coefficients' and e['side']=='after')
 def load(e,n):return np.load(root/e['arrays'][n]['blob']).copy()
 raw={n[6:]:load(b,n) for n in b['arrays'] if n.startswith('state.')};ref=dict(np.load((options.reference_root/'outputs.npz')));runs=[]
 class State(SimpleNamespace):
  def scratch(self,shape,slot):return self._scratch.setdefault(slot,self._array_module.zeros(shape,np.float32))
 with MetalContext() as context:
  xp=MetalArrayModule(context)
  for repeat in range(3):
   s=State(**{n:xp.asarray(v) for n,v in raw.items()},_scratch={},_array_module=xp,_metal_setup_state=True,cf1=p.cf1,cf2=p.cf2,cf3=p.cf3,has_msf=True)
   km=xp.asarray(load(b,'input.arg2'));kh=xp.asarray(load(b,'input.arg3'));deform=dycore.launch_wrf_smag2d_km(s,cfg,km,kh,time_t=True)
   runs.append({'km':km.to_numpy(),'kh':kh.to_numpy(),**{'deformation'+str(i):v.to_numpy() for i,v in enumerate(deform)}})
 rows=[]
 for n,v in runs[0].items():
  old=load(a,{'km':'input.arg2','kh':'input.arg3'}.get(n,'input.return'+n[-1]));d=v.astype(float)-ref[n];od=old.astype(float)-ref[n]
  rows.append({'field':n,'before_rmse':float(np.sqrt(np.mean(od*od))),'after_rmse':float(np.sqrt(np.mean(d*d))),'before_max_abs':float(np.abs(od).max()),'after_max_abs':float(np.abs(d).max()),'old_nonzero':int(np.count_nonzero(old)),'new_nonzero':int(np.count_nonzero(v)),'repeat_identical':all(np.array_equal(v,x[n]) for x in runs[1:]),'boundary_zero':bool(np.all(v[...,0,:]==0) and np.all(v[...,-1,:]==0) and np.all(v[...,:,0]==0) and np.all(v[...,:,-1]==0)) if n in ['km','kh'] else None})
 np.savez((options.output/'outputs.npz'),**runs[0]);(options.output/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
