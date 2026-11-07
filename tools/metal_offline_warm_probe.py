"""Exercise the production offline warm-start with a synthetic checkpoint."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from woof.config import RunConfig
from woof.core.grid import make_base_state, make_vertical_coord
from woof.core.state import init_theta_perturbation
from woof.core.nest_relocation import relocatable_attrs
from woof.ingest.nest_init import RK_TIME_T_SEED_PAIRS
from woof.io.restart import write_restart, restore_restart
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.offline_child_warm import apply_warm_start, STRIP_ACCUMULATOR_HISTORY

cfg=RunConfig(nx=8,ny=7,nz=5,dx=1000.,dy=1000.,ztop=5000.,dt=1.,run_seconds=2.,moist=True)
coord=make_vertical_coord(cfg.nz);base=make_base_state(coord,lambda z:np.full_like(np.asarray(z,dtype=np.float64),300.),p_surf=cfg.p_surf,ztop=cfg.ztop)
def perturb(x,z):return np.zeros((cfg.nz,cfg.ny,cfg.nx),np.float32)
def expected(before,source,di,dj):
    out=before.copy();ny,nx=out.shape[-2:];sy,sx=source.shape[-2:]
    for j in range(ny):
        for i in range(nx):
            if 0<=j+dj<sy and 0<=i+di<sx:out[...,j,i]=source[...,j+dj,i+di]
    return out
with tempfile.TemporaryDirectory(prefix='woof-metal-warm-') as d, MetalContext() as context:
    context.batch_kernel_calls=False
    submit=context._submit_kernel_batch
    def sync(*a,**kw):
        r=submit(*a,**kw);r.wait();return r
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    def fresh():return init_theta_perturbation(cfg,coord,base,perturb,array_module=xp)
    donor=fresh()
    for name in ('u','v','w','thp','php','mup','qv','qc','qr'):
        a=getattr(donor,name);v=np.arange(a.size,dtype=np.float32).reshape(a.shape)*np.float32(1.e-6)
        if name=='qv':v+=np.float32(.005)
        a[...] = v
    checkpoint=Path(d)/'donor.npz';write_restart(checkpoint,donor,cfg)
    with np.load(checkpoint) as z:payload={n:z[n] for n in z.files}
    # The model checkpoint above is real; these explicit synthetic land and
    # accumulator members exercise its offline handoff with a minimal driver.
    land=np.arange(4*cfg.ny*cfg.nx,dtype=np.float32).reshape(4,cfg.ny,cfg.nx)+50
    payload['fields/tslb']=land
    slots={name:np.arange(cfg.ny*cfg.nx,dtype=np.float32).reshape(cfg.ny,cfg.nx)+100+i for i,name in enumerate(STRIP_ACCUMULATOR_HISTORY)}
    payload.update({f'scratch/{n}':v for n,v in slots.items()});np.savez(checkpoint,**payload)
    strip={name:np.full((cfg.ny,cfg.nx),200+i,np.float32) for i,name in enumerate(slots)}
    stripfile=Path(d)/'strip.npz';np.savez(stripfile,**strip)
    results=[]
    for di,dj in ((0,0),(2,-1),(-2,1)):
        child=fresh();driver=SimpleNamespace(fields={'tslb':xp.asarray(np.full_like(land,-9))})
        before={n:getattr(child,n).to_numpy() for n in relocatable_attrs(child) if f'state/{n}' in payload and getattr(child,n,None) is not None}
        for name in slots:child.scratch((cfg.ny,cfg.nx),name)[...]=np.float32(-7)
        receipt=apply_warm_start(child,cfg,driver,checkpoint=checkpoint,shift=(di,dj),strip=stripfile)
        bad=[];words=0
        for name,v in before.items():
            if name in ('p','al','alt','p_perturbation'):continue
            exp=expected(v,payload[f'state/{name}'],di,dj);got=getattr(child,name).to_numpy();words+=got.size
            if got.tobytes()!=exp.tobytes():bad.append(name)
        got=driver.fields['tslb'].to_numpy();words+=got.size
        if got.tobytes()!=expected(np.full_like(land,-9),land,di,dj).tobytes():bad.append('land')
        for name,src in slots.items():
            exp=expected(np.full_like(src,-7),src,di,dj)
            for j in range(cfg.ny):
                for i in range(cfg.nx):
                    sj,si=j+dj,i+di
                    if not (0<=sj<cfg.ny and 0<=si<cfg.nx) or sj in (0,cfg.ny-1) or si in (0,cfg.nx-1):exp[j,i]=strip[name][j,i]
            got=child.existing_scratch(name).to_numpy();words+=got.size
            if got.tobytes()!=exp.tobytes():bad.append(name)
        for current,seed in RK_TIME_T_SEED_PAIRS:
            a,b=getattr(child,current,None),getattr(child,seed,None)
            if a is not None and b is not None and a.to_numpy().tobytes()!=b.to_numpy().tobytes():bad.append(seed)
        for name in ('p','al','alt'):

            val=getattr(child,name).to_numpy()
            if not np.isfinite(val).all():bad.append(name+':'+str(np.count_nonzero(~np.isfinite(val))))
        path=Path(d)/f'warm-{di}-{dj}.npz';write_restart(path,child,cfg);restored=fresh();restore_restart(path,restored,cfg)
        for name in before:
            if getattr(child,name).to_numpy().tobytes()!=getattr(restored,name).to_numpy().tobytes():bad.append('restore/'+name)
        results.append({'shift':[di,dj],'words':words,'bad':bad,'receipt':receipt})
    print(json.dumps(results,indent=2))
    if any(r['bad'] for r in results):raise SystemExit(1)
