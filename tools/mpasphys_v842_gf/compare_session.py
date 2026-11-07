"""Exercise the GF callback with native Registry pools and driver cadence."""
from dataclasses import replace
import json
import numpy as np
from woof.hex.mpasphys import gf_mpas_v842 as native
from woof.hex.mpasphys.metal_gf_v842 import MetalMpasGrellFreitas
from woof.metal.runtime import MetalContext
from tools.gf_wrf461_oracle import gf_regime_columns as fixture
from tools.mpasphys_v842_metal.session_fixture import session_fixture
from woof.hex.mpasphys.driver_v842 import resolve_namelist,PhysicsDriver,PhysicsManager
from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry
from woof.hex.mpasphys.metal_transaction_v842 import MetalPhysicsTransaction


def same(a,b):
    if isinstance(a,np.ndarray):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


def main():
    text="&physics\nconfig_physics_suite='convection_permitting'\nconfig_convection_scheme='cu_grell_freitas'\nconfig_conv_interval='00:10:00'\nconfig_o3climatology=.false.\n/\n"
    config=resolve_namelist(text);config=replace(config,backends={**config.backends,'convection':'metal'})
    service=MetalMpasGrellFreitas()
    column=fixture.build_column(fixture.REGIMES[1],dx=3000,ishallow=1,ichoice=0,dt=300)
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        context._submit_kernel_batch=sync
        with session_fixture(context,config,text,nz=len(column.lev['t']),slot_services={('convection',native.SCHEME):service}) as session:
            p=session.slot_pools;nc=p['nCellsSolve'];nz=p['nVertLevels'];prep=p['prepared'];diag=p['diag_physics'];mesh=p['mesh'];surface=p['sfc_input'];tend=p['tend_physics']
            def fill(target,value):target[...]=np.broadcast_to(np.asarray(value,dtype=target.dtype),target.shape).copy()
            for name,source in native._PREPARED_FIELDS.items():
                values=column.p8w[:-1] if name=='p8w' else column.lev['qv' if name=='q' else name]
                fill(prep[source][:nz],values[:,None])
            for target,source in [('rthdynten','rthften'),('rqvdynten','rqvften'),('rthblten','rthblten'),('rqvblten','rqvblten'),('rthratenlw','rthraten')]:fill(tend[target],column.lev[source][:,None])
            fill(tend['rthratensw'],0)
            for name,value in [('hfx',column.sca['hfx']),('qfx',column.sca['qfx']),('gsw',200),('kpbl',column.kpbl)]:fill(diag[name],value)
            fill(mesh['meshDensity'],1);fill(mesh['areaCell'],3000**2);fill(mesh['zgrid'],column.sca['ht']);fill(surface['xland'],column.sca['xland']);fill(prep['cu_act_flag'],1)
            session.native.field_storage('diag_physics','raincv',level=1)[nc]=np.float32(12345)
            registry=MetalSlotRegistry();session.register_slots(registry)
            no_op=lambda call:None
            for slot,scheme in config.selectors.items():
                if slot=='convection' or not config.active(slot):continue
                key=(slot,scheme,config.backends[slot])
                if key in registry._callbacks:registry._callbacks[key]=no_op
                else:registry.register(slot,scheme,config.backends[slot],no_op)
            external={name:no_op for name in ('physics_tables_init','physics_registry_init','physics_run_init','physics_init','physics_update_surface','MPAS_to_physics','physics_get_tend','post_transport_prepare','atmphys_sfc_diagnostics','pre_microphysics','post_microphysics','snapshot_gwdo_tendencies','atmphys_finalize')}
            hooks=session.bind_hooks(external=external,save_fields=lambda:{},restore_fields=lambda x:None,after_call=lambda e,c:None)
            def factory():return PhysicsDriver(config,PhysicsManager(config,300),registry,hooks)
            driver=factory();driver.initialize(session)
            tx=MetalPhysicsTransaction(driver,session.pools,factory,stateful_services=session.transaction.stateful_services)
            def advance(step):
                flags=tx.begin_step(itimestep=step,time_seconds=float(session.native.clock_state.elapsed),driver_pools=p)
                tx.sources(driver_pools=p);tx.finish_step(driver_pools=p);tx.commit();session.native.advance_clock();return flags.convection
            assert advance(1);middle=tx.checkpoint()
            held={name:diag[name].to_numpy().copy() for name in ('raincv','xmb_total','xmb_shallow')}
            rain_before=diag['rainc'].to_numpy().copy()
            assert not advance(2)
            assert np.all(diag['rainc'].to_numpy()>rain_before)
            for name,value in held.items():assert np.array_equal(diag[name].to_numpy().view(np.uint32),value.view(np.uint32)),name
            assert advance(3);expected=tx.checkpoint();tx.restore(middle);assert not advance(2);assert advance(3);got=tx.checkpoint();same(expected,got)
            assert session.native.get_field('diag_physics','raincv',level=1)[nc]==np.float32(12345)
            assert np.any(diag['raincv'].to_numpy()>0) and np.any(diag['xmb_shallow'].to_numpy()>0)
            assert not np.any(prep['cu_act_flag'].to_numpy())
            assert any('convection' in key and value=={} for key,value in got['services'].items())
            print(json.dumps({'registry_fields':len(got['pools']),'registry_values':sum(x.size for x in got['pools'].values()),'cadence':[True,False,True],'checkpoint_replay':True}),flush=True)

if __name__=='__main__':main()
