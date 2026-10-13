#!/usr/bin/env python3
"""Capture the first production surface/PBL calls and replay identical inputs.

Runs the public CLI with unchanged prepared inputs and stops before step two.
NumPy mirrors diagnose arithmetic differences; they are not exact CUDA oracles.
"""
from pathlib import Path
import argparse,inspect,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--capture-root',type=Path,required=True)
p.add_argument('--points',default='',help='Optional i,j pairs separated by semicolons')
p.add_argument('--at-seconds',type=float,default=0.,help='First due surface call at or after this model time')
a,cli=p.parse_known_args();a.capture_root.mkdir(parents=True,exist_ok=True)
from woof.core import physics,sfclay,noah,ysu
from tests.physics_reference.npref import np_sfclay,np_noah_column,np_ysu_column
records=[];elapsed=0.
class Complete(BaseException):pass
def host(x):return x.to_numpy() if hasattr(x,'to_numpy') else np.asarray(x)
def selected(shape):
 ny,nx=shape
 if a.points:return [tuple(map(int,s.split(','))) for s in a.points.split(';')]
 return [(i,j) for j in sorted(set([0,ny//4,ny//2,3*ny//4,ny-1])) for i in sorted(set([0,nx//4,nx//2,3*nx//4,nx-1]))]
def metrics(op,got,refs):
 rr=[]
 for n,ref in refs.items():
  if n not in got:continue
  d=got[n].astype(float)-np.asarray(ref,float)
  rr.append({'field':n,'rmse':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.abs(d).max()),'reference_max_abs':float(np.abs(ref).max()),'peak_index':list(map(int,np.unravel_index(np.abs(d).argmax(),d.shape)))})
 records.append({'operator':op,'elapsed_seconds':elapsed,'metrics':rr})
 (a.capture_root/'replay.json').write_text(json.dumps(records,indent=2)+'\n')
 np.savez(a.capture_root/(op+'-outputs.npz'),**{'metal_'+n:v for n,v in got.items()},**{'reference_'+n:np.asarray(v) for n,v in refs.items()})
 print(op,json.dumps(rr),flush=True)
def replace(fn,new):
 for module in list(sys.modules.values()):
  if module is not None and getattr(module,'__name__','').startswith('woof.'):
   for n,v in list(vars(module).items()):
    if v is fn:setattr(module,n,new)
orig_sf=sfclay.launch_sfclay
sig_sf=inspect.signature(orig_sf)
def capture_sf(*args,**kw):
 if elapsed<a.at_seconds:return orig_sf(*args,**kw)
 b=sig_sf.bind(*args,**kw);b.apply_defaults();d=b.arguments;r=d['result']
 inputs={n:host(d[n]).copy() for n in ['u','v','t','qv','p','dz8w','psfc','tsk','pblh','mavail','xland','lakemask']}
 inputs.update({n:host(getattr(r,n)).copy() for n in ['znt','ust','ustm','mol','hfx','qfx','qsfc','zol']})
 np.savez(a.capture_root/'sfclay-inputs.npz',**inputs)
 orig_sf(*args,**kw)
 refs=np_sfclay(**inputs,**{n:d[n] for n in ['option','dx','isfflx','isftcflx','iz0tlnd']})
 metrics('sfclay',{n:host(getattr(r,n)) for n in sfclay.SFCLAY_OUTPUTS},refs)
replace(orig_sf,capture_sf)
orig_no=noah.launch_noah;sig_no=inspect.signature(orig_no)
def capture_no(*args,**kw):
 if elapsed<a.at_seconds:return orig_no(*args,**kw)
 b=sig_no.bind(*args,**kw);b.apply_defaults();d=b.arguments;f=args[0];params=args[1];dt=args[2];dzs=args[3]
 fields={n:host(f[n]).copy() for n in list(noah._F2D)+list(noah._F3D)+['ivgtyp','isltyp','ebal']};points=selected(fields['tsk'].shape)
 np.savez(a.capture_root/'noah-inputs.npz',**fields,points=np.asarray(points),veg=params.veg,soil=params.soil,gen=params.gen,dzs=dzs)
 orig_no(*args,**kw)
 outputs={n:host(f[n]) for n in fields};refs={};got={}
 for i,j in points:
  col={n:(v[:,j,i] if v.ndim==3 else v[j,i]) for n,v in fields.items()}
  opts={n:kw[n] for n in ['isurban','isice','xice_threshold','frpcpn','usemonalb','rdlai2d','opt_thcnd'] if n in kw}
  ref=np_noah_column(col,params,dt,dzs,**opts)
  for n in fields:
   if n not in ref:continue
   refs.setdefault(n,[]).append(ref[n]);got.setdefault(n,[]).append(outputs[n][:,j,i] if outputs[n].ndim==3 else outputs[n][j,i])
 metrics('noah',{n:np.asarray(v) for n,v in got.items()},{n:np.asarray(v) for n,v in refs.items()})
replace(orig_no,capture_no)
orig_ys=ysu.launch_ysu;sig_ys=inspect.signature(orig_ys)
def capture_ys(*args,**kw):
 if elapsed<a.at_seconds:return orig_ys(*args,**kw)
 b=sig_ys.bind(*args,**kw);b.apply_defaults();d=b.arguments;names=['u','v','theta','qv','qc','qi','p','p_interface','exner','dz','rthraten','psfc','znt','ust','hfx','qfx','wspd','br','psim','psih','xland','u10','v10']
 inputs={n:host(d[n]).copy() for n in names};points=selected(inputs['psfc'].shape)
 np.savez(a.capture_root/'ysu-inputs.npz',**inputs,points=np.asarray(points))
 result=orig_ys(*args,**kw);outputs={n:host(v) for n,v in result.items()};refs={};got={}
 for i,j in points:
  col={n:(v[:,j,i] if v.ndim==3 else float(v[j,i])) for n,v in inputs.items()}
  ref=np_ysu_column(**col,dt=d['dt'],ysu_topdown_pblmix=d['ysu_topdown_pblmix'])
  for n,v in outputs.items():
   if n not in ref:continue
   refs.setdefault(n,[]).append(ref[n]);got.setdefault(n,[]).append(v[:,j,i] if v.ndim==3 else v[j,i])
 metrics('ysu',{n:np.asarray(v) for n,v in got.items()},{n:np.asarray(v) for n,v in refs.items()})
 (a.capture_root/'capture.json').write_text(json.dumps({'status':'COMPLETE','elapsed_seconds':elapsed,'operators':['sfclay','noah','ysu'],'points':points},indent=2)+'\n')
 raise Complete()
replace(orig_ys,capture_ys)
original=physics.PhysicsDriver._run_sfclay
def timing(self,*args,**kw):
 global elapsed
 elapsed=float(self.state.elapsed_seconds)
 return original(self,*args,**kw)
physics.PhysicsDriver._run_sfclay=timing
from woof.cli import main
try:main(cli)
except Complete:print('Captured first production surface calls; stopped before step end.');raise SystemExit(0)
if (a.capture_root/'capture.json').exists():raise SystemExit(0)
raise RuntimeError('CLI ended before a complete surface capture.')
