#!/usr/bin/env python3
"""Capture identical first-call Thompson component inputs through the public CLI.

Readbacks preserve production FP32 temperature, entry masks and scratch buffers.
Capture representative columns plus the strongest initial hydrometeors, then
stop after the first microphysics call without completing the first step.
"""
from pathlib import Path
import argparse,sys,json,inspect,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--capture-root',type=Path,required=True)
a,cli=p.parse_known_args()
from woof.core import microphysics,thompson
out=a.capture_root;out.mkdir(parents=True,exist_ok=True);owner={};stage={}
class Complete(BaseException):pass
def host(v):return v.to_numpy() if hasattr(v,'to_numpy') else np.asarray(v)
def columns(v):
 h=host(v);pts=owner['points'];return np.asarray([h[:,j,i] for i,j in pts])
def snapshot():
 s=owner['state'];d={n:columns(getattr(s,n)) for n in ['qv','qc','qr','qi','qs','qg','ni','nr']}
 for n,slot in [('T','mp_thompson_temperature'),('ng','mp_thompson_graupel_number_shadow'),('th','mp_th')]:
  if slot in s._scratch:d[n]=columns(s._scratch[slot])
 for n in ['rainnc','rainncv','snownc','graupelnc']:
  v=s._scratch.get('mp_'+n)
  if v is not None:
   h=host(v);d[n]=np.asarray([h[j,i] for i,j in owner['points']])
 return d
def replace(fn,new):
 for m in list(sys.modules.values()):
  if m is not None and getattr(m,'__name__','').startswith('woof.'):
   for n,v in list(vars(m).items()):
    if v is fn:setattr(m,n,new)
for name,label in [('launch_frozen_vapor_network_from_owner','cold'),('launch_warm_frozen_source_network_from_owner','sources'),('launch_cloud_saturation_adjust','condensation'),('launch_rain_evaporation','rain_evaporation'),('launch_rain_sedimentation','sedimentation'),('launch_final_phase_cleanup','cleanup'),('launch_classic_graupel_number_finalize','finalize')]:
 original=getattr(thompson,name)
 def factory(original=original,label=label):
  def call(*args,**kw):
   parameters=inspect.signature(original).bind(*args,**kw);parameters.apply_defaults();inputs={};scalars={};objects=[]
   for n,x in parameters.arguments.items():
    if hasattr(x,'to_numpy'):
     h=host(x);pts=owner['points']
     if h.ndim==3:inputs[n]=np.asarray([h[:,j,i] for i,j in pts]).T[:,None,:].copy()
     elif h.ndim==2:inputs[n]=np.asarray([[h[j,i] for i,j in pts]])
     else:inputs[n]=h.copy()
    elif x is None or isinstance(x,(bool,int,float,str)):scalars[n]=x
    else:objects.append(n)
   np.savez(out/(label+'-inputs.npz'),**inputs)
   (out/(label+'-parameters.json')).write_text(json.dumps({'function':original.__name__,'scalars':scalars,'objects':objects},indent=2)+'\n')
   v=original(*args,**kw)
   outputs={}
   for n in inputs:
    h=host(parameters.arguments[n]);pts=owner['points']
    if h.ndim==3:outputs[n]=np.asarray([h[:,j,i] for i,j in pts]).T[:,None,:].copy()
    elif h.ndim==2:outputs[n]=np.asarray([[h[j,i] for i,j in pts]])
    else:outputs[n]=h.copy()
   np.savez(out/(label+'-actual-outputs.npz'),**outputs)
   if owner:stage[label]=snapshot()
   return v
  return call
 replace(original,factory())
original=microphysics._apply_thompson
def capture(s,cfg,dt,**kw):
 ny,nx=s.p.shape[1:];pts={(i,j) for j in [0,ny//4,ny//2,3*ny//4,ny-1] for i in [0,nx//4,nx//2,3*nx//4,nx-1]}
 # Also include strongest common initial hydrometeors in both warm/cold regimes.
 for n in ['qc','qr','qs','qg']:
  z=host(getattr(s,n));inds=np.argpartition(z.max(axis=0).ravel(),-8)[-8:]
  pts.update((int(v%nx),int(v//nx)) for v in inds)
 pts=sorted(pts);owner.update(state=s,points=pts)
 thb=host(s.thb);thb=thb[:,None,None] if thb.ndim==1 else thb
 phb=host(s.phb);phb=phb[:,None,None] if phb.ndim==1 else phb
 th=thb+host(s.thp);geop=phb+host(s.php)
 raw={n:columns(getattr(s,n)) for n in ['p','w','qv','qc','qr','qi','qs','qg','ni','nr']}
 raw['th']=np.asarray([th[:,j,i] for i,j in pts]);raw['geop']=np.asarray([geop[:,j,i] for i,j in pts]);np.savez(out/'inputs.npz',**raw)
 result=original(s,cfg,dt,**kw);stage['final']=snapshot()
 np.savez(out/'metal-stages.npz',**{label+'__'+n:v for label,d in stage.items() for n,v in d.items()})
 (out/'capture.json').write_text(json.dumps({'status':'COMPLETE','dt':dt,'elapsed_seconds':float(s.elapsed_seconds),'points':pts,'count':len(pts),'stages':list(stage),'mp_tend_lim':cfg.mp_tend_lim,'no_mp_heating':cfg.no_mp_heating,'thompson_version':cfg.thompson_version},indent=2)+'\n')
 raise Complete()
replace(original,capture)
from woof.cli import main
try:main(cli)
except Complete:raise SystemExit(0)
if (out/'capture.json').exists():raise SystemExit(0)
raise RuntimeError('Incomplete microphysics capture')
