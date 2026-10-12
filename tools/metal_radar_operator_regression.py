import pathlib,sys,json
from types import SimpleNamespace
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from woof.metal.runtime import MetalContext
original=MetalContext._submit_kernel_batch
def sync(self,*a,**kw):
 r=original(self,*a,**kw)
 if r is not None:r.wait()
 return r
MetalContext._submit_kernel_batch=sync
from woof.metal.array import MetalArrayModule,MetalArray
from woof.metal.suites import configuration
from woof.core.grid import make_vertical_coord,make_base_state
from woof.core.state import init_at_rest
from woof.core.physics import initialize_physics
from woof.da.obsop import simulated_reflectivity
module=MetalArrayModule();results={}
try:
 for mp in [8,9,16,18]:
  cfg=configuration((mp,0,0,0,0,'periodic'))
  coord=make_vertical_coord(cfg.nz);base=make_base_state(coord,lambda z:300.+.003*np.asarray(z),cfg.p_surf,cfg.ztop)
  state=init_at_rest(cfg,coord,base,array_module=module);state.qv[...]=np.float32(.004)
  initialize_physics(state,cfg,glw=300.)
  from woof.core.diagnostics import update_diagnostics
  update_diagnostics(state,cfg.hypsometric_opt)
  shape=state.p.shape
  for name,value in [('qc',2e-4),('qr',3e-4),('qi',2e-5),('qs',1e-4),('qg',5e-5),('qh',1e-5),('nc',1e8),('nr',1e4),('ni',1e4),('ns',1e3),('ng',1e3),('nh',1e2),('qndrop',1e8),('qnr',1e4),('qni',1e4),('qns',1e3),('qng',1e3),('qnh',1e2),('qvolg',5e-5/400),('qvolh',1e-5/900)]:
   a=getattr(state,name,None)
   if a is not None:a[...]=np.float32(value)
  # A sub-threshold graupel residue must survive the Thompson H(x) route.
  graupel=state.qg.to_numpy();graupel[:,0,0]=np.float32(1e-13);state.qg[...]=graupel
  fields={n:a for n,a in vars(state).items() if isinstance(a,MetalArray)}
  before={n:a.to_numpy().copy() for n,a in fields.items()}
  first=simulated_reflectivity(state,cfg).to_numpy().copy()
  assert np.isfinite(first).all() and first.max()>0,(mp,first.min(),first.max())
  for _ in range(2):np.testing.assert_array_equal(simulated_reflectivity(state,cfg).to_numpy().view(np.uint32),first.view(np.uint32))
  for n,a in fields.items():np.testing.assert_array_equal(a.to_numpy().view(np.uint32),before[n].view(np.uint32),err_msg=n)
  if mp!=18:
   p=before['p'];thb=before['thb'];thb=thb[:,None,None] if thb.ndim==1 else thb
   t=np.asarray((thb+before['thp'])*np.power(p/np.float32(100000.),np.float32(287./1004.5)),np.float32)
   supplied=simulated_reflectivity(state,cfg,temperature=module.asarray(t),pressure=state.p).to_numpy()
   np.testing.assert_allclose(supplied,first,rtol=0,atol=.002)
  else:
   try:simulated_reflectivity(state,cfg,temperature=state.qv,pressure=state.p)
   except ValueError:pass
   else:raise AssertionError('NSSL accepted an unsupported supplied thermodynamic pair')
  for n,a in fields.items():np.testing.assert_array_equal(a.to_numpy().view(np.uint32),before[n].view(np.uint32),err_msg=n)
  for n in ['qc','qr','qi','qs','qg','qh']:
   a=getattr(state,n,None)
   if a is not None:a[...]=np.float32(0.)
  clear=simulated_reflectivity(state,cfg).to_numpy()
  floor={8:-35.,9:-99.,16:-35.,18:0.}[mp]
  np.testing.assert_array_equal(clear,np.float32(floor))
  results[mp]={'clear_air_floor_dbz':floor,'wet_dbz_range':[float(first.min()),float(first.max())],'repeat_bitwise':True,'forecast_unchanged':True}
 print(json.dumps({'schemes':results,'synchronous':True}))
finally:module.context.close()
