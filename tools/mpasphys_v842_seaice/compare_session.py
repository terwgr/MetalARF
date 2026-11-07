"""Bind native sea ice through actual Registry views and replay its service state.

Other schemes and framework callbacks are isolated. The direct component has
separate selected-source comparisons; this checks descriptor transport and
whole-session checkpoint identity against that same component invocation.
"""
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
from tools.mpasphys_v842_metal.session_fixture import session_fixture
from tools.mpasphys_v842_rrtmg_lw.compare_session import same
from woof.hex.mpasphys.driver_v842 import resolve_namelist,PhysicsDriver,PhysicsManager
from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry
from woof.hex.mpasphys.metal_seaice_v842 import MetalMpasSeaice
from woof.hex.mpasphys.metal_transaction_v842 import MetalPhysicsTransaction
from woof.hex.mpasphys.seaice_v842 import _DIAG_FIELDS,_SFC_FIELDS,OPTIONAL_FIELDS,_SEA_FLUX_FIELDS
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    namelist="&physics\nconfig_physics_suite='convection_permitting'\nconfig_frac_seaice=.true.\n/\n"
    config=resolve_namelist(namelist)
    config=replace(config,backends={**config.backends,'seaice':'metal'})
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
        service=MetalMpasSeaice(context,module_locations=[11,12],i_start=1,j_index=1)
        with session_fixture(context,config,namelist,dt=60,slot_services={('seaice','sf_noah'):service}) as session:
            p=session.slot_pools;nc=p['nCellsSolve'];diag=p['diag_physics'];surface=p['sfc_input'];prepared=p['prepared']
            initial=dict(t3d=[255,265,270,272,275,268],qv3d=.002,p8w3d=99000,dz8w=30,
                glw=240,qgh=.003,gsw=100,snoalb2d=.8,xice=[0,.05,.2,.5,.8,1],rib=.1,chs=.015,
                emiss=.98,albedo=.4,z02d=.001,snow=15,tsk=265,snowc=.8,snowh2d=.06,
                cqs2=.014,qsfc=.002,cpm=1000,sr_input=.5,rainncv=.001,raincv=.002)
            def fill(target,value):
                target[...]=np.broadcast_to(np.asarray(value,np.float32),target.shape).copy()
            for source,target in _DIAG_FIELDS.items():fill(diag[source],initial.get(target,0))
            for name in OPTIONAL_FIELDS:fill(diag[name],0)
            for name,value in [('sr',.5),('rainncv',.001),('raincv',.002)]:fill(diag[name],value)
            for source,target in _SFC_FIELDS.items():
                if source!='tslb':fill(surface[source],initial.get(target,0))
            fill(surface['tslb'],np.linspace(263,270,4,dtype=np.float32)[:,None])
            for name,target in [('t3d','t_p'),('qv3d','qv_p'),('p8w3d','pres2_hyd_p'),('dz8w','dz_p')]:fill(prepared[target],initial[name])
            prepared['pres2_hyd_p'][1][...]=np.full(nc,97000,np.float32)
            for j,name in enumerate(_SEA_FLUX_FIELDS):fill(prepared[name],.01*(j+1))
            session.native.field_storage('diag_physics','hfx',level=1)[nc]=np.float32(12345)
            ref=SimpleNamespace(slot='seaice',scheme='sf_noah',options=config.options,dt=60,
                pools={'diag_physics':{name:xp.asarray(diag[name].to_numpy()) for name in set(_DIAG_FIELDS)|set(OPTIONAL_FIELDS)|{'sr','rainncv','raincv'}},
                       'sfc_input':{name:xp.asarray(surface[name].to_numpy()) for name in _SFC_FIELDS},
                       'seaice_shared':{name:xp.asarray(prepared[target].to_numpy()) for name,target in
                           {'t3d':'t_p','qv3d':'qv_p','p8w3d':'pres2_hyd_p','dz8w':'dz_p',**{name:name for name in _SEA_FLUX_FIELDS}}.items()}})
            reference=MetalMpasSeaice(context,module_locations=[11,12],i_start=1,j_index=1)
            reference.launch_slot(ref)
            registry=MetalSlotRegistry();session.register_slots(registry)
            for slot,scheme in config.selectors.items():
                if config.active(slot) and slot!='seaice':registry.register(slot,scheme,config.backends[slot],lambda call:None)
            class Hooks:
                def invoke(self,operation,call):pass
                def restart_state(self):return {}
                def restore_restart_state(self,state):assert not state
            def factory():return PhysicsDriver(config,PhysicsManager(config,60),registry,Hooks())
            driver=factory();driver.initialize(session)
            tx=MetalPhysicsTransaction(driver,session.pools,factory,stateful_services=session.transaction.stateful_services)
            before=tx.checkpoint()
            def advance():
                tx.begin_step(itimestep=1,time_seconds=0,driver_pools=p)
                calls=[call for call in tx.driver.calls if call['slot']=='seaice']
                assert len(calls)==1 and calls[0]['backend']=='metal'
                tx.sources(driver_pools=p);tx.finish_step(driver_pools=p);tx.commit()
                return tx.checkpoint()
            expected=advance()
            for pool,names in [('diag_physics',_DIAG_FIELDS),('sfc_input',_SFC_FIELDS)]:
                for name in names:same(p[pool][name].to_numpy(),ref.pools[pool][name].to_numpy())
            same(service.restart_state(),reference.restart_state())
            tx.restore(before);actual=advance();same(expected,actual)
            assert session.native.get_field('diag_physics','hfx',level=1)[nc]==np.float32(12345)
            print(json.dumps({'registry_fields':len(actual['pools']),'registry_values':sum(v.size for v in actual['pools'].values()),'source_component_binding':True,'driver_service_replay':True,'other_slots_isolated':True}),flush=True)


if __name__=='__main__':main()
