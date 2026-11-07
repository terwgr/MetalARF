"""Compare actual PWP and sea-ice callers against host-compiled selected source."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity import host_backend
host_backend.install()
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
from woof.ocean import seaice, mixed_layer

ice_source=(ROOT/'woof/ocean/seaice/thermodynamics.cu').read_text()
pwp_source='#define OCEAN_LEVELS 3\n'+(ROOT/'woof/ocean/mixed_layer/pwp.cu').read_text().split('// One-degree spatial bins')[0]
ice=host_backend.HostModule('seaice_pwp',ice_source)
pwp=host_backend.HostModule('pwp_coupled',pwp_source)
seaice._module=lambda:ice
mixed_layer.kernels=lambda levels:pwp
shape=(3,3);n=9;levels=3
F=lambda x:np.full(shape,x,np.float32)
fields={name:F(0) for name in set(seaice._FREEZE_COLUMN)|set(seaice.STATE)}
fields.update(ice_ocean_mask=np.array([[0,1,1],[1,1,1],[1,1,1]],np.float32),
              _pwp_land=np.array([[2,1,2],[2,2,2],[2,2,2]],np.float32),
              ice_ocean_coupled=np.array([[0,0,0],[1,1,0],[1,0,1]],np.float32),
              ice_ocean_heat=np.array([[0,0,-30],[-50,10,1],[-100,0,-2]],np.float32),
              ice_ocean_owed=np.array([[0,0,1],[.25,-.5,1],[2,0,20]],np.float32),
              xice=F(.3),ice_volume=F(.2),ice_skin=F(270),ice_thickness=F(.67),
              ice_snow_mass=F(3),ice_snow_depth=F(.03),sst=F(271.35),tsk=F(271.35),
              om_ml=np.array([[5,5,5],[5,10,15],[5,10,15]],np.float32))
fields['om_depth']=np.broadcast_to(np.array([5,10,20],np.float32)[:,None,None],(3,*shape)).copy()
fields['om_tmp']=np.broadcast_to(np.array([271.4,271.34,272],np.float32)[:,None,None],(3,*shape)).copy()
fields['om_tini']=fields['om_tmp'].copy()

def compare(host,device,names):
    return {k:int(np.count_nonzero(host[k].view(np.uint32)!=device[k].to_numpy().view(np.uint32))) for k in names}

def copy(x):return {k:v.copy() for k,v in x.items()}
with MetalContext() as ctx:
    ctx.batch_kernel_calls=False;submit=ctx._submit_kernel_batch
    def sync(*a,**kw):
        r=submit(*a,**kw);r.wait();return r
    ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
    h=copy(fields);d={k:xp.asarray(v) for k,v in fields.items()}
    seaice.seed_pwp_profile(h);seaice.seed_pwp_profile(d)
    seed=compare(h,d,('om_tmp','om_tini','tsk'))
    assert h['om_tmp'][1,1,1] == np.float32(271.35)
    assert h['om_tmp'][1,0,1] == fields['om_tmp'][1,0,1]
    # Start the heat case independently so it exercises supercooled layers.
    h=copy(fields);d={k:xp.asarray(v) for k,v in fields.items()}
    cfg=SimpleNamespace(sf_ocean_physics=2,seaice_new_ice_thickness=.1)
    for step in range(3):
        seaice.reconcile_ocean(h,cfg,dt=600.);seaice.reconcile_ocean(d,cfg,dt=600.)
        c=compare(h,d,tuple(fields))
        if any(c.values()):raise AssertionError(('freeze',step,c))
    freeze=c
    assert np.any(h['ice_growth'] > 0) and np.any(h['ice_ocean_owed'] != 0)
    assert np.array_equal(h['om_tmp'][:,0,1],fields['om_tmp'][:,0,1])
    # The full two-stage PWP caller exercises both ice-owned and open-water fluxes.
    def pwp_fields(coupled):
        out={k:v.copy() for k,v in fields.items()}
        for k in ('om_s','om_u','om_v','om_density','om_tini','om_sini','_pwp_w','_pwp_p'):
            out[k]=np.zeros((levels,*shape),np.float32)
        out['om_tmp'][:]=np.array([280,279,278],np.float32)[:,None,None]
        out['om_tini'][:]=out['om_tmp'];out['om_s'][:]=35;out['om_sini'][:]=35
        out['_pwp_base']=np.zeros((4*levels,*shape),np.float32);out['_pwp_tend']=out['_pwp_base'].copy()
        out['_pwp_status']=np.zeros(1,np.int32);out['_pwp_mx']=F(1);out['_pwp_my']=F(1)
        for k,v in {'hfx':50,'lh':80,'gsw':200,'glw':300,'qfx':0,'ust':0,'uair':1,'vair':0,'f':0}.items():out[k]=F(v)
        if not coupled:out.pop('ice_ocean_coupled')
        return out
    modes={}; outputs={}
    for coupled in (False,True):
        orig=pwp_fields(coupled);h=copy(orig);d={k:xp.asarray(v) for k,v in orig.items()}
        mixed_layer.step(h,dt=60.,dx=1000.,dy=1000.);mixed_layer.step(d,dt=60.,dx=1000.,dy=1000.)
        names=('om_tmp','om_s','om_u','om_v','om_density','om_ml','tsk','_pwp_tend','_pwp_status')
        modes[str(coupled)]=compare(h,d,names)
        outputs[coupled]=h['om_tmp'].copy()
    flux_difference=int(np.count_nonzero(outputs[False].view(np.uint32)!=outputs[True].view(np.uint32)))
    assert flux_difference > 0
    result={'ice_sha256':hashlib.sha256(ice_source.encode()).hexdigest(),
            'pwp_sha256':hashlib.sha256(pwp_source.encode()).hexdigest(),
            'host_flags':host_backend.CXX_FLAGS,'seed':seed,'freeze':freeze,'pwp':modes,
            'coupled_flux_differing_words':flux_difference}
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if any(seed.values()) or any(freeze.values()) or any(v for x in modes.values() for v in x.values()):raise SystemExit(1)
