"""Exercise the coupled PWP and ice physics checkpoint on synthetic columns."""
import json
import sys
import tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from woof.metal.array import MetalArrayModule
from woof.metal.suites import configuration
from woof.core.grid import make_vertical_coord,make_base_state
from woof.core.state import init_at_rest
from woof.core.physics import initialize_physics
from woof.core.dycore import run_steps
from woof.io.restart import write_restart,restore_restart
xp=MetalArrayModule()
xp.context.batch_kernel_calls=False
submit=xp.context._submit_kernel_batch
def sync(*a,**kw):
 r=submit(*a,**kw);r.wait();return r
xp.context._submit_kernel_batch=sync
cfg=replace(configuration((0,1,2,0,0,'periodic')),sf_ocean_physics=2,oml_hml0=20.,ocean_levels=3,
            ocean_mld_climatology='configured_uniform_mld',run_seconds=20.)
coord=make_vertical_coord(cfg.nz);base=make_base_state(coord,lambda z:262.+.004*np.asarray(z),cfg.p_surf,cfg.ztop)
water=np.zeros((cfg.ny,cfg.nx),bool);water[:,:5]=True
xice=np.where(water,np.linspace(0,1,cfg.nx)[None,:]*np.ones((cfg.ny,1)),0.).astype(np.float32)
def build():
 state=init_at_rest(cfg,coord,base,array_module=xp);state.qv[...]=np.float32(.001)
 tsk=np.where(water,271.2,255.).astype(np.float32)
 initialize_physics(state,cfg,glw=250.,swdown=300.,tsk=tsk,sst=np.where(water,271.3,0.).astype(np.float32),
                    xice=xice,landmask=(~water).astype(np.float32),xland=np.where(water,2.,1.).astype(np.float32))
 return state
def words(a):return a.to_numpy().view(np.uint32).copy()
try:
 with tempfile.TemporaryDirectory() as td:
  continuous=build();split=build();restored=build();run_steps(continuous,cfg,6);run_steps(split,cfg,3)
  path=Path(td)/'checkpoint.npz';write_restart(path,split,cfg);restore_restart(path,restored,cfg)
  run_steps(restored,cfg,3)
  fields=continuous.physics.fields;other=restored.physics.fields
  keys=[k for k in fields if k.startswith('ice_') or k.startswith('om_') or k in ('tsk','sst','_pwp_tend','_pwp_status')]
  mismatch={k:int(np.count_nonzero(words(fields[k])!=words(other[k]))) for k in keys}
  growth=fields['ice_growth'].to_numpy();owed=fields['ice_ocean_owed'].to_numpy();coupled=fields['ice_ocean_coupled'].to_numpy()
  result={'different_words':sum(mismatch.values()),'fields':mismatch,'coupled_columns':int(np.count_nonzero(coupled)),
          'growth_columns':int(np.count_nonzero(growth)),'owed_columns':int(np.count_nonzero(owed)),
          'finite':all(np.isfinite(fields[k].to_numpy()).all() for k in keys)}
  print(json.dumps(result,indent=2));assert result['finite'] and result['different_words']==0 and result['coupled_columns']>0
finally:xp.context.close()
