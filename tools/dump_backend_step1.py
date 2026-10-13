#!/usr/bin/env python3
"""Dump identical operator boundaries in one production step on Metal or CUDA.

No inputs/selectors are modified. Diagnostic readbacks synchronize kernels.
Metal batches are submitted as ordered singletons solely to expose each acoustic
operator. Stops before a second step; it never runs the 12-hour forecast.
Use --list-points without GPU access to inspect exact hooks and common names.
"""
from __future__ import annotations
import argparse
from functools import wraps
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

# Bootstrap checkout imports when invoked as python tools/dump_backend_step1.py.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
HOOKS={
 'update_diagnostics':'eos_diagnostics',
 'prepare_moist_cq':'moist_acoustic_coefficients',
 'prepare_acoustic_coefficients':'implicit_acoustic_coefficients',
 'couple_ysu_tendencies':'pbl_mass_map_and_face_coupling',
 'stage_fluxes':'coupled_flux_and_omega',
 'launch_flux_div_scalar':'scalar_advection',
 'launch_flux_div_u':'u_advection',
 'launch_flux_div_v':'v_advection',
 'launch_flux_div_w':'w_advection',
 '_launch_slow_pgf':'horizontal_pressure_gradient',
 '_launch_slow_buoyancy':'buoyancy',
 '_launch_slow_geopotential':'geopotential_advection',
 'add_coriolis_curvature':'coriolis_curvature',
 'launch_wrf_smag2d_km':'smag_coefficients',
 'launch_wrf_smag2d_hd':'smag_stress_or_scalar_flux',
 'launch_diff6_to_edge':'sixth_order_edge_diffusion',
 'launch_diff6':'sixth_order_diffusion',
 'prepare_fixed_tendencies':'held_diffusion_tendencies_complete',
 'add_fixed_dry_tendencies':'held_dry_diffusion_added',
 'add_h_diabatic_tendency':'previous_microphysics_heating_added',
 'apply_w_damping':'vertical_courant_damping',
 'apply_state_lateral_boundaries':'specified_and_relaxation_tendencies',
 'refresh_saved_wind_fluxes':'saved_momentum_flux_refresh',
 'advance_scalars_stage':'moisture_stage_update',
 'launch_pd_fluxes':'positive_definite_fluxes',
 'launch_pd_renorm_apply':'positive_definite_budget_and_update',
 'apply_state_boundary_values':'boundary_values',
 'apply_specified_w_zero_gradient':'specified_w_ghosts',
 'set_w_surface':'terrain_surface_w',
 'apply_microphysics':'microphysics_and_zero_out',
}
PHYSICS_HOOKS={'compute':'physics_tendencies_complete','_run_radiation':'radiation',
 '_run_sfclay':'surface_layer','_run_noah':'noah_surface_flux',
 '_run_ysu':'ysu_pbl'}
KERNEL_POINTS={
 'small_step_init_uv':'acoustic_seed_uv', 'small_step_init_column':'acoustic_seed_column',
 'small_step_finish_uv':'acoustic_fold_uv', 'small_step_finish_column':'acoustic_fold_column',
 'apply_emdiv':'external_divergence_filter',
 'advance_uv':'acoustic_uv', 'advance_mu_th':'acoustic_mass_theta',
 'advance_mu_th_msf':'acoustic_mass_theta',
 'advance_w_phi':'acoustic_w_phi','advance_w_phi_msf':'acoustic_w_phi',
 'upper_wind_limiter':'upper_wind_limiter',
 'acoustic_boundary_frame':'acoustic_boundary_frame',
 'advance_specified_phi_w':'acoustic_boundary_frame',
 'advance_nested_phi_w':'acoustic_boundary_frame',
 'zero_sumflux':'sumflux_zero','accumulate_sumflux':'sumflux_accumulate',
 'finish_sumflux':'sumflux_mean',
 'wrf_smag_deform':'smag_deformation',
 'wrf_smag2d_km':'smag_mixing_coefficient',
 'wrf_smag_km_bc':'smag_coefficient_boundary',
 'wrf_smag_flux_s':'smag_scalar_metric_flux',
 'wrf_smag_hd_s':'smag_scalar_metric_divergence',
 'diff6':'diff6_flux_divergence',
}
STATE_FIELDS='u v w thp php mup qv qc qr qi qs qg ni nr thb phb pb p al alt mub2d msft msfu msfv f e sina cosa c1h c2h c1f c2f fnm fnp dnw dn rdnw rdn ht u0 v0 w0 thp0 php0 mup0 qv0 ru_t rv_t rw_t rth_t rph_t rmu_t h_diabatic u_pp v_pp w_pp ph_pp th_pp mu_pp p_pp p_pp_old al_pp ww_pp'.split()

class DumpComplete(BaseException):pass

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--list-points',action='store_true')
 p.add_argument('--backend',choices=['metal','cuda'])
 p.add_argument('--experiment',type=Path)
 p.add_argument('--prepared-root',type=Path)
 p.add_argument('--wps-namelist',type=Path)
 p.add_argument('--outdir',type=Path)
 a=p.parse_args(argv)
 if a.list_points:
  print(json.dumps({'function_boundaries':HOOKS,'physics_boundaries':PHYSICS_HOOKS,
   'acoustic_kernel_boundaries':KERNEL_POINTS,'class_boundaries':{'PhysicsTendencies.add_to_slow':'held_physics_tendencies_added'},'RK_stages':'all three: 1,2,3; acoustic substeps 1,2,4 for time_step_sound4',
   'dumps':'before and after; immutable content-addressed .npy blobs; chronological manifest includes inherited full state, scratch, physics and inputs'},indent=2));return 0
 for k in ['backend','experiment','prepared_root','wps_namelist','outdir']:
  if getattr(a,k) is None:p.error('--'+k.replace('_','-')+' required')
 # Refuse a nonempty destination to protect preceding evidence.
 if a.outdir.exists() and any(a.outdir.iterdir()):p.error('outdir must be empty or absent')
 a.outdir.mkdir(parents=True,exist_ok=True);blobs=a.outdir/'arrays';blobs.mkdir()
 from woof import prepared_single_domain_forecast  # load production aliases
 from woof.core import dycore,advection,moist,physics,acoustic
 from woof.cli import main as cli_main
 owner={'state':None,'cfg':None,'active':False,'stage':0,'substep':0,'events':[],'depth':0}
 def host(x):
  if isinstance(x,np.ndarray):return x
  if hasattr(x,'to_numpy'):return x.to_numpy()
  if hasattr(x,'get') and hasattr(x,'shape'):return x.get()
  return None
 def collect(extra):
  s=owner['state'];d={}
  for n in STATE_FIELDS:
   x=getattr(s,n,None)
   if x is not None:d['state.'+n]=x
  for n,x in getattr(s,'_scratch',{}).items():
   if isinstance(n,str) and n.startswith(('rk_','smag_','diff6_','acoustic_')):d['scratch.'+n]=x
  driver=getattr(s,'physics',None)
  if driver:
   for n,x in getattr(driver,'fields',{}).items():d['physics.'+n]=x
   for name in ['rthratenlw','rthratensw','pbl_tendencies','radiation_tendencies']:
    obj=getattr(driver,name,None)
    if hasattr(obj,'shape'):d['physics.'+name]=obj
    elif obj is not None:
     for n,x in vars(obj).items() if hasattr(obj,'__dict__') else []:
      if hasattr(x,'shape'):d['physics.'+name+'.'+n]=x
  if extra:
   for n,x in extra.items():
    if hasattr(x,'shape'):d['input.'+n]=x
  return d
 def dump(point,side,extra=None,parameters=None):
  if not owner['active']:return
  arrays={}
  for n,x in collect(extra).items():
   h=host(x)
   if h is None or h.dtype.kind not in 'fiub':continue
   h=np.ascontiguousarray(h)
   digest=hashlib.sha256(h.dtype.str.encode()+repr(h.shape).encode()+h.tobytes()).hexdigest()
   path=blobs/(digest+'.npy')
   if not path.exists():np.save(path,h,allow_pickle=False)
   arrays[n]={'blob':'arrays/'+path.name,'shape':list(h.shape),'dtype':h.dtype.str}
  event={'index':len(owner['events']),'point':point,'side':side,'rk_stage':owner['stage'],
   'acoustic_substep':owner['substep'],'elapsed_seconds':float(owner['state'].elapsed_seconds),
   'dt':float(owner['cfg'].dt),'arrays':arrays,'parameters':parameters}
  owner['events'].append(event)
  with (a.outdir/'events.jsonl').open('a') as f:f.write(json.dumps(event,default=str)+'\n')
 def replace_aliases(original,replacement):
  # Caller modules often import functions by value. Patch every loaded alias.
  for module in list(sys.modules.values()):
   if module is None or not getattr(module,'__name__','').startswith('woof.'):continue
   for name,value in list(vars(module).items()):
    if value is original:setattr(module,name,replacement)
 def wrap_function(original,point,stage=False):
  @wraps(original)
  def wrapped(*args,**kwargs):
   if not owner['active']:return original(*args,**kwargs)
   if stage:owner['stage']+=1;owner['substep']=0
   extra={}
   def add_extra(key,value):
    if hasattr(value,'shape'):extra[key]=value
    elif isinstance(value,dict):
     for n,x in value.items():add_extra(key+'.'+str(n),x)
    elif isinstance(value,(tuple,list)):
     for n,x in enumerate(value):add_extra(key+'.'+str(n),x)
    elif isinstance(value,physics.PhysicsTendencies):
     for n,x in vars(value).items():add_extra(key+'.'+n,x)
   for i,x in enumerate(args):add_extra(f'arg{i}',x)
   for k,v in kwargs.items():add_extra(k,v)
   parameters={k:v for k,v in kwargs.items() if isinstance(v,(bool,int,float,str,np.generic))}
   dump(point,'before',extra,parameters)
   result=original(*args,**kwargs)
   # Tuples of stage mass fluxes and coefficient arrays are actual outputs.
   if isinstance(result,tuple):extra.update({f'return{i}':x for i,x in enumerate(result) if hasattr(x,'shape')})
   elif result is not None and hasattr(result,'__dict__'):
    extra.update({f'return_{n}':x for n,x in vars(result).items() if hasattr(x,'shape')})
   dump(point,'after',extra,parameters);return result
  wrapped._backend_dump_wrapped=True
  return wrapped
 originals=set()
 for name,point in HOOKS.items():
  for module in [dycore,advection,moist,acoustic,physics]:
   fn=getattr(module,name,None)
   if callable(fn) and fn not in originals and not getattr(fn,'_backend_dump_wrapped',False):
    originals.add(fn);replace_aliases(fn,wrap_function(fn,point))
 for name,point in PHYSICS_HOOKS.items():
  fn=getattr(physics.PhysicsDriver,name);setattr(physics.PhysicsDriver,name,wrap_function(fn,point))
 physics.PhysicsTendencies.add_to_slow=wrap_function(physics.PhysicsTendencies.add_to_slow,'held_physics_tendencies_added')
 fn=dycore._add_slow_tendencies
 replace_aliases(fn,wrap_function(fn,'slow_tendencies_complete',stage=True))
 # Closure factories invoke these raw kernels directly, bypassing function
 # wrappers. CUDA launch proxies retain grids, args and kernels unchanged.
 original_get=acoustic.get_kernel
 def traced_get(module,name,*args,**kwargs):
  kernel=original_get(module,name,*args,**kwargs)
  if name not in KERNEL_POINTS:return kernel
  def launch(*la,**lk):
   if owner['active'] and name=='advance_uv':owner['substep']+=1
   values=la[2] if len(la)>2 else ()
   extra={f'kernel_arg{i}':x for i,x in enumerate(values) if hasattr(x,'shape')}
   scalars={f'arg{i}':x.item() if isinstance(x,np.generic) else x for i,x in enumerate(values) if isinstance(x,(bool,int,float,np.generic))}
   dump(KERNEL_POINTS[name],'before',extra,scalars)
   result=kernel(*la,**lk);dump(KERNEL_POINTS[name],'after',extra,scalars);return result
  return launch
 replace_aliases(original_get,traced_get)
 if a.backend=='metal':
  from woof.metal.runtime import MetalContext
  original_batch=MetalContext._submit_kernel_batch
  def singleton_batch(context,calls,*,label):
   if not owner['active']:return original_batch(context,calls,label=label)
   result=None
   for call in calls:
    point=KERNEL_POINTS.get(call.entrypoint)
    if call.entrypoint=='advance_uv':owner['substep']+=1
    params={'entrypoint':call.entrypoint,'parameter_bytes_hex':call.parameter_bytes.hex(),
     'grid_size':call.grid_size,'exact_policy':call.exact_policy}
    if point:dump(point,'before',parameters=params)
    result=original_batch(context,(call,),label=label)
    if point:dump(point,'after',parameters=params)
   return result
  MetalContext._submit_kernel_batch=singleton_batch
  # Direct prepared calls are another public execution switch. Force the
  # batch route, then split it as above so all actual calls are intercepted.
  original_step=dycore.step
 else:original_step=dycore.step
 @wraps(original_step)
 def first_step(state,cfg,*args,**kwargs):
  if owner['active']:raise RuntimeError('nested step capture unsupported')
  if int(cfg.grid_id)!=1:raise RuntimeError('dump script requires one root domain')
  owner.update(state=state,cfg=cfg,active=True,stage=0,substep=0)
  if a.backend=='metal':state._array_module.context.batch_kernel_calls=True
  dump('initial_restored_state','before')
  original_step(state,cfg,*args,**kwargs)
  dump('completed_first_step','after')
  receipt={'status':'CAPTURE_COMPLETE','backend':a.backend,'outer_steps':1,
   'rk_stages':owner['stage'],'events':len(owner['events']),'model_seconds':float(state.elapsed_seconds),
   'source_inputs_unmodified':True,'execution_change':'synchronous readbacks and ordered singleton Metal batches; no arithmetic/selector override',
   'interpretation':'Compare common state/scratch arrays at common points, not positional raw ABI names. Stage0 includes initial physics and held diffusion. All3RK stages and all7acoustic substeps captured. A missing point is not a zero; compare manifests and stop at first large operator delta.'}
  (a.outdir/'capture.json').write_text(json.dumps(receipt,indent=2)+'\n');owner['active']=False
  raise DumpComplete()
 replace_aliases(original_step,first_step)
 cli_args=['go',str(a.experiment),'--prepared-root',str(a.prepared_root),'--wps-namelist',str(a.wps_namelist),
  '--backend',a.backend,'--outdir',str(a.outdir/'cli-initialization'),'--run-stamp','off','--products','none']
 try:code=cli_main(cli_args)
 except DumpComplete:print('Captured first step; no second step or full forecast.');return 0
 # Some runner versions record the deliberate stop and return a CLI error.
 # Only the completed receipt proves all three stages were captured.
 receipt_path=a.outdir/'capture.json'
 if receipt_path.exists():
  receipt=json.loads(receipt_path.read_text())
  if receipt.get('status')=='CAPTURE_COMPLETE' and receipt.get('outer_steps')==1 and receipt.get('rk_stages')==3:
   print('Captured first step; runner recorded the deliberate diagnostic stop.');return 0
 raise RuntimeError(f'Production CLI returned {code} before a complete step capture')

if __name__=='__main__':raise SystemExit(main())
