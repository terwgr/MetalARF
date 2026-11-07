"""Exercise diagnostics-off Thompson through native Registry and transaction replay."""
from dataclasses import replace
import sys
import numpy as np
from tools.mpasphys_v842_metal import session_fixture as fixture
from woof.hex.mpasphys import thompson_v842 as native
from woof.hex.mpasphys.metal_thompson_tables_v842 import load_tables
from woof.hex.mpasphys.metal_thompson_column_v842 import bind_session,launch_slot,MetalMpasThompson
from woof.hex.mpasphys.driver_v842 import resolve_namelist,PhysicsDriver,PhysicsManager,SlotCall,StepFlags
from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry
from woof.hex.mpasphys.metal_transaction_v842 import MetalPhysicsTransaction
from woof.metal.runtime import MetalContext

def same(a,b):
    if isinstance(a,np.ndarray):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b

namelist="&physics\nconfig_physics_suite='convection_permitting'\nconfig_microp_scheme='mp_thompson'\n/\n"
config=resolve_namelist(namelist);config=replace(config,backends={**config.backends,'microphysics':'metal'})
with MetalContext(batch_kernel_calls=False) as context:
    tables=load_tables({'table_directory':sys.argv[1]},context)
    service=MetalMpasThompson(tables)
    with fixture.session_fixture(context,config,namelist,dt=2,nz=4,slot_services={('microphysics','mp_thompson'):service}) as session:
        session.native.call('allocate_microphysics',configs='configs')
        for name,value in [('dt_microp',np.float32(1)),('n_microp',np.int32(2)),('l_acrain',np.bool_(True)),('has_reqc',np.int32(1)),('has_reqi',np.int32(1)),('has_reqs',np.int32(1))]:
            session.native.put_variable(name,np.asarray(value))
        top=session.pools.field('tend_physics','mp_top_level');top[...]=np.int32(2)
        p=bind_session(session,tables)
        def fill(a,value):a[...]=np.broadcast_to(np.asarray(value,dtype=a.dtype),a.shape).copy()
        for name,value in [('qv',.006),('qc',.0001),('qr',.00008),('qi',.00001),('qs',.00001),('qg',.00001),('ni',900),('nr',2000),('theta_m',290),('rho_zz',1.15),('w',.1)]:fill(p['model'][name],value)
        fill(p['mesh']['zz'],.95);fill(p['mesh']['zgrid'],np.arange(5,dtype=np.float32)[:,None]*500)
        for name,value in [('exner',.98),('exner_base',.97),('pressure_base',90000),('pressure_p',100),('rtheta_base',300),('rtheta_p',10),('surface_pressure',98000)]:fill(p['diagnostics'][name],value)
        fill(p['surface']['ntc'],100e6)
        registry=MetalSlotRegistry();session.register_slots(registry)
        session.nest_look=object();rejected=bind_session(session,tables)
        assert rejected['nest_look'] is session.nest_look
        before=session.pools.snapshot()
        blocked=SlotCall(slot='microphysics',scheme='mp_thompson',backend='metal',itimestep=1,time_seconds=0,dt=2,flags=StepFlags(),pools=session.slot_pools,options=config.options)
        try:registry.run(blocked)
        except NotImplementedError:pass
        else:raise AssertionError('bound nest-look must be refused')
        same(before,session.pools.snapshot());session.nest_look=None
        for slot,selected in config.selectors.items():
            if config.active(slot) and slot!='microphysics':
                registry.register(slot,selected,config.backends[slot],lambda call:None)
        class Hooks:
            def invoke(self,operation,call):
                if operation=='physics_timetracker_state':return session.calendar_callback(call)
                return None
            def restart_state(self):return {}
            def restore_restart_state(self,state):assert not state
        def factory():return PhysicsDriver(config,PhysicsManager(config,2),registry,Hooks())
        driver=factory();driver.initialize(session)
        tx=MetalPhysicsTransaction(driver,session.pools,factory,stateful_services=session.transaction.stateful_services)
        def advance(i):
            flags=tx.begin_step(itimestep=i,time_seconds=float(session.native.clock_state.elapsed),driver_pools=session.slot_pools)
            assert not flags.diagnostics
            tx.sources(driver_pools=session.slot_pools);tx.finish_step(driver_pools=session.slot_pools)
            assert len([c for c in tx.driver.calls if c['slot']=='microphysics'])==1
            tx.commit();session.native.advance_clock()
        advance(1);middle=tx.checkpoint();advance(2);expected=tx.checkpoint();tx.restore(middle);advance(2);got=tx.checkpoint();same(expected,got)
        print('native Registry fields',len(got['pools']),'values',sum(v.size for v in got['pools'].values()),'driver checkpoint replay matched; other slots isolated')
