"""MYNN through real Registry, scheduled driver and in-memory rollback.

The shared fixture uses the current Metal session metadata. Other schemes
and framework hooks are isolated; no public native provider is admitted.
"""
from dataclasses import replace
import json
import argparse
import numpy as np
from woof.metal.runtime import MetalContext

from tools.mpasphys_v842_metal import session_fixture as fixture
from woof.hex.mpasphys.driver_v842 import resolve_namelist,PhysicsDriver,PhysicsManager
from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry
from woof.hex.mpasphys.metal_transaction_v842 import MetalPhysicsTransaction
from woof.hex.mpasphys.metal_bl_mynn_slot_v842 import MetalMpasBlMynn


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exact',action='store_true');parser.add_argument('--rp7',action='store_true');parser.add_argument('--rp1',action='store_true');args=parser.parse_args();exact=args.exact
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            result=original(*a,**kw);result.wait();return result
        ctx._submit_kernel_batch=sync
        namelist="&physics\nconfig_physics_suite='convection_permitting'\nconfig_pbl_interval='00:10:00'\n/\n"
        config=resolve_namelist(namelist);corrections=({'RP6'} if not exact else set()) | ({'RP7'} if args.rp7 else set()) | ({'RP1'} if args.rp1 else set())
        config=replace(config,backends={**config.backends,'pbl':'metal'},mode='correct' if corrections else 'exact',corrections=frozenset(corrections))
        service=MetalMpasBlMynn(ctx)
        with fixture.session_fixture(ctx,config,namelist,slot_services={('pbl','bl_mynn'):service},nz=17) as session:
            p=session.slot_pools;nc=p['nCellsSolve'];nl=p['nVertLevels'];prepared=p['prepared'];diag=p['diag_physics'];surface=p['sfc_input'];mesh=p['mesh']
            def fill(target,value):target[...]=np.broadcast_to(np.asarray(value,dtype=np.float32),target.shape).copy()
            for name,value in (('dz_p',100),('u_p',5),('v_p',3),('w_p',0),('th_p',np.linspace(300,295,nl)[:,None]),('t_p',np.linspace(300,295,nl)[:,None]),('pres_hyd_p',np.linspace(100000,70000,nl)[:,None]),('pi_p',.96),('rho_p',1.1),('qv_p',.008),('qc_p',.0001),('qi_p',.00002),('qs_p',.00001),('nc_p',1e5),('ni_p',1e4),('nifa_p',1e4),('nwfa_p',1e5)):fill(prepared[name][:nl],value)
            fill(prepared['w_p'][nl:nl+1],3.5)
            fill(prepared['psfc_p'],99063.0)
            for name,value in (('el_pbl',10),('qke',.1),('cov',.001),('qsq',.001),('tsq',.001),('sh3d',.1),('sm3d',.1),('qc_bl',.0001),('qi_bl',.00002),('cldfrac_bl',.2),('ust',.4),('hfx',100),('qfx',.0001),('rmol',0),('wspd',5)):fill(diag[name],value)
            for name in ('edmf_a','edmf_w','edmf_qt','edmf_thl','edmf_ent','edmf_qc','sub_thl','sub_qv','det_thl','det_qv'):fill(diag[name],.3)
            fill(surface['skintemp'],305);fill(surface['xland'],np.array([1,2,1,2,1,2]));fill(mesh['meshDensity'],1)
            registry=MetalSlotRegistry();session.register_slots(registry)
            for slot,selected in config.selectors.items():
                if config.active(slot) and slot!='pbl':registry.register(slot,selected,config.backends[slot],lambda call:None)
            class Hooks:
                def invoke(self,operation,call):
                    if operation=='physics_timetracker_state':return session.calendar_callback(call)
                    return None
                def restart_state(self):return {}
                def restore_restart_state(self,state):assert not state
            def factory():return PhysicsDriver(config,PhysicsManager(config,300),registry,Hooks())
            driver=factory();driver.initialize(session);tx=MetalPhysicsTransaction(driver,session.pools,factory,stateful_services=session.transaction.stateful_services)
            def advance(step):
                time=float(session.native.clock_state.elapsed);flags=tx.begin_step(itimestep=step,time_seconds=time,driver_pools=p)
                due=tx.driver.manager.pbl_due;calls=[c for c in tx.driver.calls if c['slot']=='pbl'];assert len(calls)==int(due)
                tx.sources(driver_pools=p);tx.finish_step(driver_pools=p);tx.commit();session.native.advance_clock();return due
            assert advance(1)
            for name in ('edmf_a','edmf_w','edmf_qt','edmf_thl','edmf_ent','edmf_qc','sub_thl','sub_qv','det_thl','det_qv'):assert not np.any(diag[name].to_numpy()),name
            middle=tx.checkpoint();held=diag['qke'].to_numpy().copy()
            assert advance(2)==exact
            if not exact:assert np.array_equal(diag['qke'].to_numpy().view(np.uint32),held.view(np.uint32))
            assert advance(3);expected=tx.checkpoint();tx.restore(middle);assert advance(2)==exact;assert advance(3);got=tx.checkpoint()
            for name,value in expected['pools'].items():
                actual=got['pools'][name];assert value.dtype==actual.dtype and value.shape==actual.shape and value.tobytes()==actual.tobytes(),name
            assert expected['driver']==got['driver']
            def same_state(a,b):
                if isinstance(a,np.ndarray):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
                elif isinstance(a,dict):
                    assert a.keys()==b.keys()
                    for key in a:same_state(a[key],b[key])
                elif isinstance(a,(list,tuple)):
                    assert type(a) is type(b) and len(a)==len(b)
                    for x,y in zip(a,b):same_state(x,y)
                else:assert a==b
            same_state(expected['services'],got['services'])
            assert np.any(p['tend_physics']['rthblten'].to_numpy()!=0)
            print(json.dumps({'registry_fields':len(got['pools']),'cadence':[True,exact,True],'pool_driver_and_services_replay':True}),flush=True)
if __name__=='__main__':main()
