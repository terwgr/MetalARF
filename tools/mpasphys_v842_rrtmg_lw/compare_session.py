"""Exercise LW through Registry views, manager cadence and transaction replay."""
from dataclasses import replace
import json
import numpy as np
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import load_tables
from woof.hex.mpasphys.metal_rrtmg_lw_slot_v842 import MetalMpasLongwave
from woof.metal.runtime import MetalContext
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from tools.mpasphys_v842_metal import session_fixture as fixture
from woof.hex.mpasphys.driver_v842 import resolve_namelist,PhysicsDriver,PhysicsManager
from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry
from woof.hex.mpasphys.metal_transaction_v842 import MetalPhysicsTransaction


def same(a,b):
    if isinstance(a,np.ndarray):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


def main():
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        context._submit_kernel_batch=sync;tables=load_tables(context)
        for climate in (False,True):
            namelist="&physics\nconfig_physics_suite='convection_permitting'\nconfig_microp_scheme='mp_thompson'\nconfig_radtlw_interval='00:10:00'\nconfig_o3climatology="+('.true.' if climate else '.false.')+"\n/\n"
            config=resolve_namelist(namelist);config=replace(config,backends={**config.backends,'radiation_lw':'metal'})
            op=MetalMpasLongwave(context,tables)
            with fixture.session_fixture(context,config,namelist,slot_services={('radiation_lw','rrtmg_lw'):op}) as session:
                p=session.slot_pools;nc=p['nCellsSolve'];nz=p['nVertLevels'];prep=p['prepared'];diag=p['diag_physics'];surface=p['sfc_input']
                session.native.field_storage('diag_physics','glw',level=1)[nc]=np.float32(12345)
                def fill(target,value):target[...]=np.broadcast_to(np.asarray(value,dtype=np.float32),target.shape).copy()
                for name,value in [('pres_hyd_p',np.linspace(95000,5000,nz)),('t_p',np.linspace(280,220,nz)),('dz_p',np.full(nz,150)),('pi_p',np.linspace(1,.7,nz)),('qvrad_p',np.full(nz,.003)),('qcrad_p',np.full(nz,.0001)),('qirad_p',np.full(nz,.00003)),('qsrad_p',np.full(nz,.00002))]:fill(prep[name][:nz],value[:,None])
                p8=np.broadcast_to(np.linspace(100000,100,nz+1,dtype=np.float32)[:,None],(nz+1,nc)).copy();p8[-1,1::2]=600
                fill(prep['pres2_hyd_p'][:nz+1],p8)
                counts=session.pools.arrays.empty((nc,),np.int32)
                _launch('hexv842_mpas_rrtmg_lw_slotcounts_f32',nc,context,dict(p8w=prep['pres2_hyd_p'][:nz+1],counts=counts),dict(nc=np.int32(nc),nm=np.int32(nz)))
                assert counts.to_numpy().tolist()==[nz+1,nz+2]*3
                fill(prep['t2_p'][:nz+1],np.linspace(282,215,nz+1)[:,None]);fill(prep['xlat_p'],np.linspace(-.5,.5,nc))
                for name,value in [('cldfrac',.4),('re_cloud',12e-6),('re_ice',30e-6),('re_snow',50e-6),('sfc_emiss',.97)]:fill(diag[name],value)
                if climate:
                    fill(diag['o3clim'],6e-6);fill(p['atm_input']['pin'],np.linspace(1,1100,p['atm_input']['pin'].size))
                for name,value in [('skintemp',280),('xland',np.array([1,2,1,2,1,2])),('xice',0),('snow',1)]:fill(surface[name],value)
                registry=MetalSlotRegistry();session.register_slots(registry)
                for slot,selected in config.selectors.items():
                    if not config.active(slot):continue
                    if slot!='radiation_lw':registry.register(slot,selected,config.backends[slot],lambda call:None)
                class Hooks:
                    def invoke(self,operation,call):
                        if operation=='physics_timetracker_state':return session.calendar_callback(call)
                        return None
                    def restart_state(self):return {}
                    def restore_restart_state(self,state):assert not state
                def factory():return PhysicsDriver(config,PhysicsManager(config,300),registry,Hooks())
                driver=factory();driver.initialize(session)
                tx=MetalPhysicsTransaction(driver,session.pools,factory,stateful_services=session.transaction.stateful_services)
                def advance(step):
                    flags=tx.begin_step(itimestep=step,time_seconds=float(session.native.clock_state.elapsed),driver_pools=p)
                    calls=[c for c in tx.driver.calls if c['slot']=='radiation_lw'];assert len(calls)==int(flags.radiation_lw)
                    tx.sources(driver_pools=p);tx.finish_step(driver_pools=p);tx.commit();session.native.advance_clock();return flags.radiation_lw
                assert advance(1);middle=tx.checkpoint()
                held={name:diag[name].to_numpy().copy() for name in ('glw','olrtoa','lwupt','lwdnt')}
                assert not advance(2)
                for name,value in held.items():assert np.array_equal(diag[name].to_numpy().view(np.uint32),value.view(np.uint32)),name
                assert advance(3);expected=tx.checkpoint()
                tx.restore(middle);assert not advance(2);assert advance(3);got=tx.checkpoint()
                same(expected,got)
                assert session.native.get_field('diag_physics','glw',level=1)[nc]==np.float32(12345)
                assert np.any(diag['glw'].to_numpy()>0) and np.any(diag['olrtoa'].to_numpy()>0)
                if climate:assert np.any(diag['o3vmr'].to_numpy()>0)
                words=sum(v.size for v in got['pools'].values())
                print(json.dumps({'climatology':climate,'registry_fields':len(got['pools']),'registry_values':words,'cadence':[True,False,True],'driver_clock_replay':True}),flush=True)


if __name__=='__main__':main()
