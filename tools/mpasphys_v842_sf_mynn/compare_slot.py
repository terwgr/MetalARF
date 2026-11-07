"""Native sf_mynn source versus Metal pool publication across cold/warm calls."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.metal_mynn_word_probe import plain_arithmetic
from tools.mpasphys_v842_sf_mynn.compare_columns import fixture
from woof.hex.mpasphys.sf_mynn_v842 import (_cuda_source, FIELDS, CONSTANTS, OUTPUTS,
    _DIAG_INPUT, _DIAG_NAMES, _PROFILE, _SEA_HANDOFF, _SURFACE, CORRECT_MODE_SOURCE)
from woof.hex.mpasphys.metal_sf_mynn_v842 import MetalMpasSfMynn
from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def main():
    corrections=frozenset({'RP3'}) if '--rp3' in sys.argv else frozenset()
    host=HostModule('mpas_v842_sf_mynn',plain_arithmetic(_cuda_source(corrections)))
    host.get_function('sf_mynn_tables')((1,),(1,),())
    pre,from_mpas,col,to_mpas=(host.get_function(n) for n in
        ('sf_mynn_pre_columns','sf_mynn_from_mpas','sf_mynn_columns','sf_mynn_to_mpas'))
    n=8; initial,c=fixture(n); get=lambda name:initial[FIELDS.index(name)].copy()
    constants=dict(zip(CONSTANTS,c)); diag={name:get(name) for name in _DIAG_INPUT}
    diag['hpbl']=get('pblh')
    for name in OUTPUTS:
        if name not in ('wstar','qstar'):diag.setdefault(_DIAG_NAMES.get(name,name),np.zeros(n,np.float32))
    surface={'skintemp':get('tsk'),'xland':get('xland'),'snowh':get('snowh'),
             'sst':np.linspace(265,300,n,dtype=np.float32),
             'xice':np.array([0,.05,.15,.25,.5,.8,1,1.2],np.float32)}
    p0={name:get(field) for name,field in zip(_PROFILE,('u1d','v1d','t1d','qv1d','qv1d','p1d','rho1d','dz8w1d'))}
    profiles={name:np.stack((p0[name],p0[name]+np.float32(.01))) for name in _PROFILE}
    profiles['u3d'][1]=get('u1d2');profiles['v3d'][1]=get('v1d2');profiles['dz8w'][1]=get('dz2w1d')
    density=np.linspace(.7,1.4,n,dtype=np.float32);psfc=get('psfcpa');threshold=np.float32(.1)
    if corrections:
        # Recover/clamp ice skin temperature, both low-concentration safeguards,
        # and inactive/out-of-range cells; retain cold/warm and restart calls.
        surface['xice']=np.array([0,.01,.05,.15,.25,.5,1,1.2],np.float32)
        surface['skintemp']=np.array([260,220,230,240,250,270,275,280],np.float32)
        threshold=np.float32(.02)
    total=0
    with MetalContext(batch_kernel_calls=False) as context:
        original=context._submit_kernel_batch
        def sync(*a,**k):
            result=original(*a,**k);result.wait();return result
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
        for fraction in (False,True):
            hd={k:v.copy() for k,v in diag.items()}; ds={k:xp.asarray(v) for k,v in diag.items()}
            prepared={'psfc_p':xp.asarray(psfc),**{name+'_sea':xp.zeros((n,),np.float32) for name in _SEA_HANDOFF}}
            pools={'mesh':{'meshDensity':xp.asarray(density)},'diag_physics':ds,
                   'sfc_input':{k:xp.asarray(v) for k,v in surface.items()},
                   'atm_input':{k:xp.asarray(v) for k,v in profiles.items()},'prepared':prepared,
                   'constants':constants,'configs':{'config_len_disp':3000.,'config_do_restart':False},
                   'scalar_state':{'xice_threshold':float(threshold)}}
            for step in (1,2):
                # Reconstruct the operator after the first call: tables are
                # derived state, while all physical fields stay in the pools.
                op=MetalMpasSfMynn(context)
                packed=np.zeros((len(FIELDS),n),np.float32)
                for name in _DIAG_INPUT:packed[FIELDS.index(name)]=hd[name]
                packed[FIELDS.index('pblh')]=hd['hpbl']
                smap=dict(surface,psfc=psfc)
                for name,src in _SURFACE.items():packed[FIELDS.index(name)]=smap[src]
                sea=np.zeros_like(packed);ice=surface['xice'];hold=hd['regime'].copy()
                from_mpas((1,),(128,),(packed,sea,density,surface['sst'],ice,np.float32(3000),threshold,np.int32(fraction),np.int32(n)))
                def prepare(a):
                    qc=np.empty(n,np.float32);pattern=np.zeros(n,np.float32)
                    pre((1,),(128,),(a,*(profiles[k] for k in _PROFILE),pattern,qc,np.int32(step==1),np.int32(0),np.int32(n)))
                prepare(packed)
                args=lambda a:(a,c,np.int32(step==1),np.int32(1),np.int32(0),np.int32(0),np.int32(0),np.uint32(127),np.int32(n))
                col((1,),(128,),args(packed))
                sea_expected={}
                if fraction:
                    prepare(sea);col((1,),(128,),args(sea))
                    sea_expected={name:sea[FIELDS.index(name)].copy() for name in _SEA_HANDOFF}
                    to_mpas((1,),(128,),(packed,sea,ice,hold,threshold,np.int32(n)))
                for name in OUTPUTS:
                    if name not in ('wstar','qstar'):hd[_DIAG_NAMES.get(name,name)]=packed[FIELDS.index(name)].copy()
                call=SimpleNamespace(slot='surface_layer',scheme='sf_mynn',pools=pools,itimestep=step,
                                     options={'config_frac_seaice':fraction},corrections=corrections,backend='metal')
                registry=MetalSlotRegistry()
                registry.register_metal('surface_layer','sf_mynn',op.launch_slot,corrections=CORRECT_MODE_SOURCE)
                assert corrections <= registry.declared('surface_layer','sf_mynn','metal')
                registry.run(call)
                for name in OUTPUTS:
                    if name in ('wstar','qstar'):continue
                    name=_DIAG_NAMES.get(name,name); actual=ds[name].to_numpy();expected=hd[name]
                    bad=np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32));total+=bad
                    if bad:print('mismatch',fraction,step,name,int(bad),flush=True)
                for name,expected in sea_expected.items():
                    actual=prepared[name+'_sea'].to_numpy();bad=np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32));total+=bad
                    if bad:print('mismatch sea',step,name,int(bad),flush=True)
                print('corrections',sorted(corrections),'fraction',fraction,'step',step,'cumulative differences',total,flush=True)
    return int(total!=0)
if __name__=='__main__':raise SystemExit(main())
