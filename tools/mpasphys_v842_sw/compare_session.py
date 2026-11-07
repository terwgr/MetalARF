"""SW callback through actual Registry views, manager cadence and checkpoint state.

Other scheme slots and framework preparation are deliberately isolated. This
is a composed caller/restart check, not a complete native forecast oracle.
"""
from dataclasses import replace
import json
import numpy as np
from woof.hex.mpasphys.driver_v842 import resolve_namelist,PhysicsDriver,PhysicsManager
from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry
from woof.hex.mpasphys.metal_transaction_v842 import MetalPhysicsTransaction
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import initialize_tables
from woof.hex.mpasphys.metal_rrtmg_sw_slot_v842 import MetalMpasShortwave
from woof.hex.mpasphys.rrtmg_sw_v842 import SCALAR_OUTPUTS
from woof.metal.runtime import MetalContext
from tools.mpasphys_v842_metal.session_fixture import session_fixture


def main():
    with MetalContext(batch_kernel_calls=False) as context:
        original=context._submit_kernel_batch
        def sync(*a,**kw):
            result=original(*a,**kw);result.wait();return result
        context._submit_kernel_batch=sync;tables=initialize_tables(context)
        for aerosol in (False,True):
            scheme='mp_thompson_aerosols' if aerosol else 'mp_thompson'
            namelist="&physics\nconfig_physics_suite='convection_permitting'\nconfig_microp_scheme='"+scheme+"'\nconfig_radtsw_interval='00:10:00'\n/\n"
            config=resolve_namelist(namelist)
            config=replace(config,backends={**config.backends,'radiation_sw':'metal'})
            op=MetalMpasShortwave(context,tables)
            with session_fixture(context,config,namelist,slot_services={('radiation_sw','rrtmg_sw'):op}) as session:
                p=session.slot_pools;nc=p['nCellsSolve'];nz=p['nVertLevels']
                prepared=p['prepared'];diag=p['diag_physics'];surface=p['sfc_input'];mesh=p['mesh']
                def fill(target,value):target[...]=np.broadcast_to(np.asarray(value,dtype=np.float32),target.shape).copy()
                for name,value in [('pres_hyd_p',np.linspace(95000,20000,nz)),('t_p',np.linspace(280,230,nz)),('dz_p',np.full(nz,150)),('pi_p',np.linspace(1,.7,nz)),('qvrad_p',np.full(nz,.003)),('qcrad_p',np.full(nz,.0001)),('qirad_p',np.full(nz,.00003)),('qsrad_p',np.full(nz,.00002))]:
                    fill(prepared[name][:nz],value[:,None])
                fill(prepared['pres2_hyd_p'][:nz+1],np.linspace(100000,15000,nz+1)[:,None]);fill(prepared['t2_p'][:nz+1],np.linspace(282,220,nz+1)[:,None])
                if aerosol:
                    for name,value in [('qv_p',.003),('nwfa_p',1e8),('nifa_p',1e5)]:fill(prepared[name][:nz],value)
                for name,value in [('cldfrac',.4),('re_cloud',12e-6),('re_ice',30e-6),('re_snow',50e-6),('o3clim',6e-6),('sfc_albedo',.2)]:fill(diag[name],value)
                fill(p['atm_input']['pin'],np.linspace(1,1100,p['atm_input']['pin'].size))
                for name,value in [('skintemp',280),('xland',np.array([1,2,1,2,1,2])),('xice',0),('snow',1)]:fill(surface[name],value)
                fill(mesh['latCell'],0);fill(mesh['lonCell'],np.linspace(-180,180,nc)*p['constants']['degrad'])
                registry=MetalSlotRegistry();session.register_slots(registry)
                for slot,selected in config.selectors.items():
                    if not config.active(slot):continue
                    if slot!='radiation_sw':registry.register(slot,selected,config.backends[slot],lambda call:None)
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
                    time=float(session.native.clock_state.elapsed)
                    flags=tx.begin_step(itimestep=step,time_seconds=time,driver_pools=p)
                    calls=[c for c in tx.driver.calls if c['slot']=='radiation_sw']
                    assert len(calls)==int(flags.radiation_sw)
                    tx.sources(driver_pools=p);tx.finish_step(driver_pools=p);tx.commit();session.native.advance_clock()
                    return flags.radiation_sw
                assert advance(1);middle=tx.checkpoint()
                held={name:diag[name].to_numpy().copy() for name in SCALAR_OUTPUTS}
                assert not advance(2)
                for name,value in held.items():assert np.array_equal(diag[name].to_numpy().view(np.uint32),value.view(np.uint32)),name
                assert advance(3);expected=tx.checkpoint()
                tx.restore(middle);assert not advance(2);assert advance(3);got=tx.checkpoint()
                assert expected['pools'].keys()==got['pools'].keys()
                words=0
                for name,value in expected['pools'].items():
                    actual=got['pools'][name];words+=value.size
                    assert value.dtype==actual.dtype and value.shape==actual.shape and value.tobytes()==actual.tobytes(),name
                assert expected['driver']==got['driver']
                def same_state(a,b):
                    if isinstance(a,np.ndarray):
                        assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
                    elif isinstance(a,dict):
                        assert a.keys()==b.keys()
                        for key in a:same_state(a[key],b[key])
                    elif isinstance(a,(list,tuple)):
                        assert type(a) is type(b) and len(a)==len(b)
                        for x,y in zip(a,b):same_state(x,y)
                    else:assert a==b
                same_state(expected['services'],got['services'])
                assert np.any(diag['swdnb'].to_numpy()!=0) and np.any(diag['swdnb'].to_numpy()==0)
                if aerosol:assert np.any(diag['taod5502d'].to_numpy()>0)
                print(json.dumps({'aerosol':aerosol,'registry_fields':len(got['pools']),'registry_values':words,'cadence':[True,False,True],'driver_clock_replay':True}),flush=True)


if __name__=='__main__':main()
