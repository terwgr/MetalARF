#!/usr/bin/env python3
"""Replay actual step-one advection and Coriolis inputs against NumPy mirrors."""
from pathlib import Path
from types import SimpleNamespace
import argparse,inspect,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture-root',type=Path,required=True)
a,cli=p.parse_known_args();a.capture_root.mkdir(parents=True,exist_ok=True)
from woof import prepared_single_domain_forecast
from woof.core import dycore,advection
from tests.physics_reference import npref
owner={'stage':0,'records':[],'scalar_counts':{}}
class Complete(BaseException):pass
def host(x):return x.to_numpy() if hasattr(x,'to_numpy') else np.asarray(x)
def replace(fn,new):
 for module in list(sys.modules.values()):
  if module is not None and getattr(module,'__name__','').startswith('woof.'):
   for n,v in list(vars(module).items()):
    if v is fn:setattr(module,n,new)
def save(name,data,got,ref):
 d=got.astype(float)-ref;core=d[...,5:-5,5:-5];idx=np.unravel_index(np.abs(d).argmax(),d.shape)
 row={'operator':name,'stage':owner['stage'],'rmse':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.abs(d).max()),'reference_max_abs':float(np.abs(ref).max()),'core_rmse':float(np.sqrt(np.mean(core*core))),'core_max_abs':float(np.abs(core).max()),'peak_index_kji':list(map(int,idx))}
 owner['records'].append(row);number=len(owner['records'])
 np.savez(a.capture_root/f'{number:02d}-{name}-stage{owner["stage"]}.npz',**data,metal=got,reference=ref)
 (a.capture_root/'replay.json').write_text(json.dumps(owner['records'],indent=2)+'\n');print(json.dumps(row),flush=True)
for name in ['scalar','u','v','w']:
 original=getattr(advection,'launch_flux_div_'+name);signature=inspect.signature(original);reference=getattr(npref,'np_flux_div_'+name)
 def wrap(original=original,signature=signature,reference=reference,name=name):
  def call(*args,**kw):
   stage=owner['stage']
   if name=='scalar':
    count=owner['scalar_counts'].get(stage,0);owner['scalar_counts'][stage]=count+1
    if count>=2:return original(*args,**kw)
   b=signature.bind(*args,**kw);b.apply_defaults();z=b.arguments;coord=z['coord'];fieldname=next(iter(signature.parameters));label=name
   data={n:host(z[n]).copy() for n in [fieldname,'ru','rv','rw']};data['before']=host(z['tend']).copy()
   weights={n:host(getattr(coord,n)).copy() for n in ['fnm','fnp','rdnw','rdn']};data.update(weights)
   msf=None if z['msf'] is None or z['has_msf'] is False else host(z['msf']).copy()
   if msf is not None:data['msf']=msf
   ref=reference(data[fieldname],data['ru'],data['rv'],data['rw'],SimpleNamespace(**weights),z['dx'],z['dy'],open_x=z['open_x'],open_y=z['open_y'],spec=z['spec'],vorder=z['vorder'],msf=msf)
   original(*args,**kw);save(label,data,host(z['tend']),data['before'].astype(float)+ref)
  return call
 replace(original,wrap())
original_cor=dycore.launch_coriolis_curvature;sig_cor=inspect.signature(original_cor)
def coriolis(*args,**kw):
 b=sig_cor.bind(*args,**kw);b.apply_defaults();z=b.arguments
 names=['ru','rv','u','v','w','mut','msft','msfu','msfv','f','e','c1f','c2f','fnm','fnp'];data={n:host(z[n]).copy() for n in names}
 for n in ['sina','cosa']:
  if z[n] is not None:data[n]=host(z[n]).copy()
 before={n:host(z[n]).copy() for n in ['ru_t','rv_t','rw_t']}
 refs=npref.np_coriolis_curvature(*[data[n] for n in names[:11]],SimpleNamespace(**{n:data[n] for n in ['c1f','c2f','fnm','fnp']}),z['dx'],z['dy'],sina=data.get('sina'),cosa=data.get('cosa'),boundary_x=z['boundary_x'],boundary_y=z['boundary_y'])
 original_cor(*args,**kw)
 for n,ref in zip(before,refs):save('coriolis_'+n,data,host(z[n]),before[n].astype(float)+ref)
replace(original_cor,coriolis)
original_slow=dycore._add_slow_tendencies
def slow(*args,**kw):
 owner['stage']+=1;return original_slow(*args,**kw)
replace(original_slow,slow)
original_step=dycore.step
def first_step(state,cfg,*args,**kw):
 result=original_step(state,cfg,*args,**kw)
 (a.capture_root/'capture.json').write_text(json.dumps({'status':'COMPLETE','steps':1,'stages':owner['stage'],'model_seconds':float(state.elapsed_seconds),'comparison':'Post operator vs incoming tendency plus identical-input float64 increment; no selectors or state inputs changed.'},indent=2)+'\n')
 raise Complete()
replace(original_step,first_step)
from woof.cli import main
try:main(cli)
except Complete:print('Captured one production step; stopped before a second step.');raise SystemExit(0)
if (a.capture_root/'capture.json').exists():
 print('Captured one production step; stopped before a second step.');raise SystemExit(0)
raise RuntimeError('CLI ended before a complete first step.')
