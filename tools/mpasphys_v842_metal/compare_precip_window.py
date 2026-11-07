"""Compare native RP6 window words and isolated real-driver surface handoff."""
from copy import deepcopy
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.mpasphys_v842_metal.session_fixture import session_fixture
from woof.hex.mpasphys.precip_window_v842 import SOURCE,WINDOW_ARRAYS,SNAPSHOT_PREFIX
from woof.hex.mpasphys.driver_v842 import resolve_namelist,CORRECTIONS_V842,PhysicsDriver,PhysicsManager,SlotRegistry
from woof.hex.mpasphys.metal_transaction_v842 import MetalPhysicsTransaction
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

FIELDS=('rainncv','raincv','snowncv','graupelncv','sr')
HOST=HostModule('mpas_native_precip_window',SOURCE)
ADD=HOST.get_function('precip_window_add');RESET=HOST.get_function('precip_window_reset')
def same(actual,expected):
    if isinstance(expected,np.ndarray):
        assert actual.dtype==expected.dtype and actual.shape==expected.shape
        assert actual.tobytes()==expected.tobytes()
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for name in expected:same(actual[name],expected[name])
    elif isinstance(expected,(list,tuple)):
        assert type(actual)==type(expected) and len(actual)==len(expected)
        for a,b in zip(actual,expected):same(a,b)
    else:assert actual==expected

def initial(n):
    tiny=np.resize(np.array([1,0x00100000,0x007fffff,0x00800001,0x3f000000,0x3f800000],np.uint32).view(np.float32),n)
    return {name:tiny.copy() if name!='sr' else np.resize(np.array([0,.25,.75,1],np.float32),n) for name in FIELDS}

def host_add(fields,window,steps,flags):
    n=steps.size
    ADD(((n+127)//128,),(128,),(np.int32(n),*map(np.int32,flags),*[fields[name] for name in FIELDS],*[window[name] for name in WINDOW_ARRAYS],steps))

def host_reset(window,steps):
    n=steps.size
    RESET(((n+127)//128,),(128,),(np.int32(n),*[window[name] for name in WINDOW_ARRAYS if name!='sr'],steps))

def components(context):
    xp=MetalArrayModule(context);n=257
    for flags in ((1,1,1),(1,0,0),(0,1,0),(0,0,0)):
        fields=initial(n);window={name:np.zeros(n,np.float32) for name in WINDOW_ARRAYS};steps=np.zeros(n,np.int32)
        df={name:xp.asarray(value) for name,value in fields.items()};dw={name:xp.asarray(value) for name,value in window.items()};ds=xp.asarray(steps)
        for _ in range(3):
            host_add(fields,window,steps,flags)
            _launch('hexv842_mpas_precip_window_add_f32',n,context,dict(df,nsteps_bl=ds,**{name+'_bl':value for name,value in dw.items()}),dict(n=np.int32(n),**dict(zip(('have_nc','have_c','have_sg'),map(np.int32,flags)))))
            for name in window:same(dw[name].to_numpy(),window[name])
            same(ds.to_numpy(),steps)
        host_reset(window,steps)
        _launch('hexv842_mpas_precip_window_reset_f32',n,context,dict(nsteps_bl=ds,**{name+'_bl':value for name,value in dw.items() if name!='sr'}),dict(n=np.int32(n)))
        for name in window:same(dw[name].to_numpy(),window[name])
        same(ds.to_numpy(),steps)
        print('native window presence',flags,'three accumulations and reset match',n*7,'source words',flush=True)

def session_check(context):
    text="&physics\nconfig_physics_suite='convection_permitting'\nconfig_pbl_interval='00:10:00'\nconfig_o3climatology=.false.\n/\n"
    config=resolve_namelist(text,physics_mode='correct',correct_skip=set(CORRECTIONS_V842)-{'RP6'})
    assert config.corrections==frozenset({'RP6'})
    with session_fixture(context,config,text,dt=300,nz=4) as session:
        service=session.precipitation_window;assert service is not None
        callbacks=session.bind_hooks(external={},save_fields=lambda:{},restore_fields=lambda state:None,after_call=lambda entry,call:None).external
        ops=set(service.operations());window={name:np.zeros(6,np.float32) for name in WINDOW_ARRAYS};steps=np.zeros(6,np.int32)
        observed=[]
        def land(call):
            observed.append(call.time_seconds)
            for name in service.present():same(session.pools.field('diag_physics',name).to_numpy(),window[name])
        registry=SlotRegistry()
        for slot,scheme in config.selectors.items():
            if config.active(slot):registry.register(slot,scheme,config.backends[slot],land if slot=='land_surface' else lambda call:None)
        class Hooks:
            def invoke(self,operation,call):
                if operation in ops or operation=='physics_timetracker_state':return callbacks[operation](call)
            def restart_state(self):return {}
            def restore_restart_state(self,state):assert not state
        def factory():return PhysicsDriver(config,PhysicsManager(config,300),registry,Hooks())
        driver=factory();driver.initialize(session)
        tx=MetalPhysicsTransaction(driver,session.pools,factory,stateful_services=session.transaction.stateful_services)
        def advance(i):
            fields=initial(6)
            for name in FIELDS:
                if name!='sr':fields[name]=fields[name]*np.float32(i)
                session.pools.field('diag_physics',name)[...]=fields[name]
            host_add(fields,window,steps,(1,1,1))
            tx.begin_step(itimestep=i,time_seconds=float(session.native.clock_state.elapsed),driver_pools=session.slot_pools)
            if driver.manager.pbl_due:host_reset(window,steps)
            for name in FIELDS:same(session.pools.field('diag_physics',name).to_numpy(),fields[name])
            tx.sources(driver_pools=session.slot_pools);tx.finish_step(driver_pools=session.slot_pools);tx.commit();session.native.advance_clock()
            for name in window:same(service.window[name].to_numpy(),window[name])
            same(service.steps.to_numpy(),steps)
        advance(1);advance(2);saved=tx.checkpoint();reference=deepcopy((window,steps));advance(3);expected=tx.checkpoint()
        tx.restore(saved);window,steps=deepcopy(reference);advance(3);same(tx.checkpoint(),expected)
        assert observed==[0.,600.,600.],observed
        print('native Registry RP6 due/held/due window handoff and all',len(expected['pools']),'pool/service/driver replay fields match; other schemes and startup isolated',flush=True)

if __name__=='__main__':
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        context._submit_kernel_batch=sync
        components(context);session_check(context)
