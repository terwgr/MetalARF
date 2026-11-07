#!/usr/bin/env python3
"""Exercise the real UH updater, independent windows, reset and restart."""
from pathlib import Path
import tempfile
import numpy as np
from woof.config import RunConfig
from woof.core.grid import make_vertical_coord, make_base_state
from woof.core.state import init_theta_perturbation
from woof.core.uh_diag import update_up_heli_max,reset_up_heli_max,TRACKER_WINDOW_SLOTS
from woof.io.restart import write_restart,restore_restart
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

cfg=RunConfig(nx=7,ny=6,nz=12,dx=1000.,dy=1000.,ztop=9000.,dt=1.,run_seconds=2.,nwp_diagnostics=1,open_x=True,open_y=True)
coord=make_vertical_coord(cfg.nz)
base=make_base_state(coord,lambda z:np.full_like(np.asarray(z,dtype=np.float64),300.),p_surf=cfg.p_surf,ztop=cfg.ztop)
with tempfile.TemporaryDirectory() as directory, MetalContext() as context:
    context.batch_kernel_calls=False; submit=context._submit_kernel_batch
    def sync(*a,**kw):
        result=submit(*a,**kw);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    def fresh():
        s=init_theta_perturbation(cfg,coord,base,lambda x,z:np.zeros((cfg.nz,cfg.ny,cfg.nx)),array_module=xp)
        u=np.broadcast_to(-.04*np.arange(cfg.ny,dtype=np.float32)[None,:,None],(cfg.nz,cfg.ny,cfg.nx+1)).copy()
        v=np.broadcast_to(.07*np.arange(cfg.nx,dtype=np.float32)[None,None,:],(cfg.nz,cfg.ny+1,cfg.nx)).copy()
        s.u[...] = xp.asarray(u);s.v[...] = xp.asarray(v);s.w[...] = xp.asarray(np.full((cfg.nz+1,cfg.ny,cfg.nx),4.,np.float32))
        for slot in TRACKER_WINDOW_SLOTS:s.scratch((cfg.ny,cfg.nx),slot)
        return s
    source=fresh();update_up_heli_max(source,cfg)
    up=source.existing_scratch('up_heli_max').to_numpy(); assert np.count_nonzero(up)>0
    for slot in TRACKER_WINDOW_SLOTS:assert np.array_equal(up.view(np.uint32),source.existing_scratch(slot).to_numpy().view(np.uint32))
    reset_up_heli_max(source)
    assert not np.count_nonzero(source.existing_scratch('up_heli_max').to_numpy())
    assert all(np.array_equal(up.view(np.uint32),source.existing_scratch(slot).to_numpy().view(np.uint32)) for slot in TRACKER_WINDOW_SLOTS)
    checkpoint=Path(directory)/'restart.npz';write_restart(checkpoint,source,cfg)
    restored=fresh();restore_restart(checkpoint,restored,cfg)
    update_up_heli_max(source,cfg);update_up_heli_max(restored,cfg)
    for slot in ('up_heli_max',*TRACKER_WINDOW_SLOTS):
        a=source.existing_scratch(slot).to_numpy();b=restored.existing_scratch(slot).to_numpy()
        bad=np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32));print(slot,'nonzero',int(np.count_nonzero(a)),'different',int(bad),'words',a.size,flush=True);assert bad==0
