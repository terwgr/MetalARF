"""Check classic RRTM's public adapter uses each model cell's Exner value."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from datetime import datetime,timezone
from types import SimpleNamespace
import numpy as np
from woof.core.rrtm_lw import RRTMLongwaveRadiation
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.wrf_exact import ENABLED
ap=argparse.ArgumentParser();ap.add_argument('--expect-mispacked',action='store_true');args=ap.parse_args()
assert not ENABLED
nz,ny,nx=5,2,3
shape=(nz,ny,nx)
with MetalContext() as context:
    context.batch_kernel_calls=False
    xp=MetalArrayModule(context)
    def grid(values,shape=shape):
        return xp.asarray(np.broadcast_to(np.asarray(values,dtype=np.float32),shape).copy())
    z=np.arange(nz+1,dtype=np.float32)[:,None,None]*np.float32(900)
    pw=np.linspace(100000,1200,nz+1,dtype=np.float32)[:,None,None]
    temperature=grid(np.linspace(295,235,nz,dtype=np.float32)[:,None,None]+np.arange(ny*nx,dtype=np.float32).reshape(1,ny,nx))
    atmosphere=dict(temperature=temperature,pressure=grid((pw[:-1]+pw[1:])*np.float32(.5)),dz=grid(900),qv=grid(.008),qc=grid(0),qi=grid(0),p_interface=grid(pw,(nz+1,ny,nx)),z_interface=grid(z,(nz+1,ny,nx)),exner=grid(1))
    state=SimpleNamespace(elapsed_seconds=0,fnm=xp.asarray(np.full(nz,.5,np.float32)),fnp=xp.asarray(np.full(nz,.5,np.float32)))
    fields=dict(tsk=grid(295,(ny,nx)),emiss=grid(.95,(ny,nx)))
    cfg=SimpleNamespace(mp_physics=0,bl_pbl_physics=0,icloud_bl=0,dt=1)
    adapter=RRTMLongwaveRadiation(datetime(2026,1,1,tzinfo=timezone.utc),np.zeros((ny,nx),np.float32),np.zeros((ny,nx),np.float32),p_top=1200,icloud=0)
    baseline=tuple(x.to_numpy().copy() for x in adapter.longwave_with_cloud_fraction(atmosphere=atmosphere,fields=fields,state=state,cfg=cfg))
    exner=np.linspace(.4,1.1,nz*ny*nx,dtype=np.float32).reshape(shape)
    atmosphere['exner']=xp.asarray(exner)
    actual=tuple(x.to_numpy().copy() for x in adapter.longwave_with_cloud_fraction(atmosphere=atmosphere,fields=fields,state=state,cfg=cfg))
    expected=baseline[0]/exner
    differences=int(np.count_nonzero(actual[0].view(np.uint32)!=expected.view(np.uint32)))
    for old,new in zip(baseline[1:],actual[1:]):assert np.array_equal(old.view(np.uint32),new.view(np.uint32))
    assert np.count_nonzero(baseline[0])==baseline[0].size
    assert bool(differences)==args.expect_mispacked,(differences,args.expect_mispacked)
    assert adapter.update_count==2
    print(json.dumps(dict(words=int(expected.size),different_words=differences,unchanged_flux_and_cloud=True,updates=adapter.update_count)))
