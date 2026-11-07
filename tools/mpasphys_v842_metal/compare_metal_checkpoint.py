"""Fresh-process native-carrier replay; unrelated physics is deliberately isolated.

Run under Anvil with the current session and its matching Metal bundle.
This exercises actual Registry/prepared words, Noah location state, ozone state,
RP6 window and manager alarms. Scheme computations and public forecast wiring
are outside this fixture; dycore carriers are synthetic. The Noah tables are
inert because its slot is not run.
"""
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction
import json, os, subprocess, sys, tempfile
import numpy as np
from woof.hex.mpasphys.checkpoint_v842 import CheckpointRoute
from woof.hex.mpasphys.moving_v842 import STATE_FIELDS, SAVED_FIELDS, SCHEMA, load_carrier
from woof.hex.mpasphys.metal_checkpoint_v842 import MetalCheckpointRoute
from woof.hex.mpasphys.driver_v842 import resolve_namelist, PhysicsDriver, PhysicsManager, SlotRegistry, CORRECTIONS_V842
from woof.hex.mpasphys.metal_noah_v842 import MetalMpasNoah
from woof.hex.mpasphys.noah_v842 import table_dtype
from woof.hex.mpasphys.metal_ozone_v842 import MetalOzoneService
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
from tools.mpasphys_v842_metal.session_fixture import session_fixture

TEXT="&physics\nconfig_physics_suite='convection_permitting'\nconfig_pbl_interval='00:10:00'\nconfig_o3climatology=.false.\n/\n"
ORIGIN='2024-06-28_12:00:00'
VALID='2024-06-28_12:10:00'

def same(a,b):
    if isinstance(b,np.ndarray):
        assert isinstance(a,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    elif isinstance(b,dict):
        assert a.keys()==b.keys()
        for k in b:same(a[k],b[k])
    elif isinstance(b,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b

def static():
    a=np.arange(1,8,dtype=np.float32)
    mesh=SimpleNamespace(arrays={'areaCell':a})
    vertical=SimpleNamespace(zgrid=a.copy(),dzw=a.copy())
    grid=SimpleNamespace(vertical_grid=vertical)
    reference=SimpleNamespace(**{n:a.copy() for n in ('rho_base','rho_theta_base','pressure_base','exner_base')})
    return mesh,vertical,(mesh,grid,None,None,reference,None,['qv','qc'])

def make_route(config, *, carrier=None, receipt=None, dt=300):
    route=MetalCheckpointRoute.from_route(CheckpointRoute())
    route.origin_time=ORIGIN;route.frame_from_cli=True;route.frame_seconds=600
    if carrier is not None:
        route.carrier=carrier;route.offset_seconds=carrier['time_seconds'];route.cli_restart=True
        route.restart_metadata=receipt['checkpoint']
    route.resolve_schedule(3600,dt,[])
    route.bind_inputs(static()[2],config)
    return route

def run(path, mode, phase, resume_dt):
    config=resolve_namelist(TEXT,physics_mode=mode,
                            correct_skip=set(CORRECTIONS_V842)-{'RP6'} if mode=='correct' else None)
    dt=300 if phase=='write' else resume_dt
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        context._submit_kernel_batch=sync
        with session_fixture(context,config,TEXT,dt=dt,nz=4) as session:
            if phase=='resume':session.native.clock(start=VALID,stop='2024-06-28_13:00:00',dt=dt)
            registry=SlotRegistry()
            for slot,scheme in config.selectors.items():
                if config.active(slot):registry.register(slot,scheme,config.backends[slot],lambda call:None)
            class IsolatedHooks:
                def invoke(self,operation,call):return None
                def restart_state(self):return {}
                def restore_restart_state(self,state):assert not state
            driver=PhysicsDriver(config,PhysicsManager(config,dt),registry,IsolatedHooks())
            driver.initialize(session);session.transaction.driver=driver
            xp=MetalArrayModule(context)
            noah=MetalMpasNoah(xp.asarray(np.zeros(table_dtype().itemsize//4,np.uint32)),
                               xp.asarray(np.array([7,11,13,17],np.int32)))
            session.transaction.stateful_services['land_surface/sf_noah']=noah
            ozone=MetalOzoneService(session,{})
            session.transaction.stateful_services['ozone']=ozone
            prepared=session.pools.prepared('qv_p')
            field=session.pools.field('diag_physics','rainncv')
            if phase=='write':
                prepared.fill(np.float32(.00125));field.fill(np.float32(.75))
                session.native.put_text_field('sfc_input','mminlu','MODIFIED_IGBP_MODIS_NOAH')
                session.native.put_variable('solcon',np.array(1361,np.float32))
                ozone.initialized=True
                if session.precipitation_window:
                    session.pools.field('diag_physics','sr').fill(np.float32(.25))
                    session.precipitation_window.accumulate()
                session.driver.manager.advance(0);session.driver.manager.advance(300)
                session.native.advance_clock();session.native.advance_clock()
                values={name:np.arange(1,8,dtype=np.float32) for name in STATE_FIELDS+SAVED_FIELDS}
                dycore=SimpleNamespace(atmosphere=SimpleNamespace(state=SimpleNamespace(**values),saved=SimpleNamespace(**values)),stochastic=None,chemistry=None)
                mesh,vertical,_=static();route=make_route(config)
                route.export(path,session=session,dycore=dycore,mesh=mesh,vertical=vertical,names=['qv','qc'],layout=session.layout,step=2,dt=300)
            else:
                receipt=json.loads(path.with_suffix(path.suffix+'.json').read_text());carrier=load_carrier(path)
                assert carrier['schema']==SCHEMA and carrier['route']=='mpas-suite'
                route=make_route(config,carrier=carrier,receipt=receipt,dt=dt)
                original=carrier['physics']['metal'];saved_clock=original['services']['clock']
                assert saved_clock['dt']=='300' and Fraction(saved_clock['elapsed'])==600
                before=session.driver.manager.restart_state()
                initialized_step={name:session.native.get_variable(name) for name in ('dt_dyn','n_cu','n_microp')}
                result=route.restore_physics(session,session.layout,None,static()[0])
                actual=session.transaction.checkpoint()
                same(actual['pools'],original['pools'])
                for name in original['services']:
                    if name!='clock':same(actual['services'][name],original['services'][name])
                same(actual['services']['clock']['text'],saved_clock['text'])
                assert session.native.clock_state.elapsed==0 and session.native.clock_state.dt==resume_dt
                for name,value in initialized_step.items():same(session.native.get_variable(name),value)
                assert noah.module_locations.to_numpy().tolist()==[7,11,13,17] and ozone.initialized
                assert float(session.native.get_variable('solcon')[0])==1361
                assert np.all(prepared.to_numpy()==np.float32(.00125))
                assert session.native.get_text_field('sfc_input','mminlu')=='MODIFIED_IGBP_MODIS_NOAH'
                if mode=='exact':
                    assert result['alarm_phase'] is None
                    same(session.driver.manager.restart_state()['alarms'],before['alarms'])
                    reference=PhysicsManager(config,dt)
                else:
                    assert result['alarm_phase'] is not None
                    for name,item in carrier['physics']['manager']['alarms'].items():
                        assert session.driver.manager.alarms[name].next_ring==Fraction(item['next_ring'])-600
                    assert session.precipitation_window is not None
                    assert np.all(session.precipitation_window.steps.to_numpy()==1)
                    reference=PhysicsManager(config,300)
                    reference.restore_restart_state(carrier['physics']['manager'])
                    reference.dt=Fraction(dt)
                for elapsed in range(0,601,dt):
                    actual_flags=session.driver.manager.advance(elapsed)
                    reference_flags=reference.advance(elapsed if mode=='exact' else 600+elapsed)
                    assert actual_flags==reference_flags
                    assert session.driver.manager.pbl_due==reference.pbl_due
                print(mode,'fresh-process carrier,',len(actual['pools']),'numeric fields, prepared, text, Noah, ozone and alarm replay; dt 300 ->',dt,flush=True)

if __name__=='__main__':
    if len(sys.argv)>1:run(Path(sys.argv[1]),sys.argv[2],sys.argv[3],int(sys.argv[4]))
    else:
        with tempfile.TemporaryDirectory(prefix='mpas-metal-file-checkpoint-') as temp:
            for mode in ('exact','correct'):
                for dt in (300,150):
                    path=Path(temp)/(mode+str(dt)+'.pkl')
                    for phase in ('write','resume'):
                        subprocess.run([sys.executable,__file__,str(path),mode,phase,str(dt)],check=True,env=os.environ.copy())
